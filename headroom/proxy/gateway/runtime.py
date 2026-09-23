"""Atomic policy publication, with process-lifetime mutable ownership."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from headroom.proxy.gateway.admission import AdmissionController
from headroom.proxy.gateway.auth import GatewayAuthenticator, GatewayAuthorizer
from headroom.proxy.gateway.config import GatewayConfigSnapshot, RouteConfig
from headroom.proxy.gateway.control import RedactedGatewayStatus
from headroom.proxy.gateway.credentials import CredentialBroker
from headroom.proxy.gateway.egress import EgressPolicy, build_managed_upstream_headers
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.models import AccountAvailability, CatalogSnapshot
from headroom.proxy.gateway.observability import GatewayObservability
from headroom.proxy.gateway.resources import ResourceBinding, ResourceRegistry
from headroom.proxy.gateway.routing import AccountRouter
from headroom.proxy.gateway.transport import http_client, private_transport, tls_context


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ReloadResult:
    applied: bool
    generation: int
    error: str | None = None


@dataclass(slots=True)
class RuntimeDependencies:
    """Explicit embedded/test seams; no request can supply these dependencies."""

    clock: Callable[[], float] = time.time
    egress_policy: EgressPolicy = field(default_factory=EgressPolicy)
    http_client: Any = None
    broker: Any = None
    metadata_reader: Callable[[Any, RouteConfig, str], Awaitable[tuple[str, ...]]] | None = None


@dataclass(frozen=True, slots=True)
class RuntimeGeneration:
    number: int
    snapshot: GatewayConfigSnapshot
    authenticator: GatewayAuthenticator
    authorizer: GatewayAuthorizer
    catalog: CatalogSnapshot
    broker: CredentialBroker
    config_digest: str
    authorities: tuple[tuple[str, str], ...]
    targets: tuple[tuple[str, str], ...]
    http_client: Any
    tls_context: Any
    egress_policy: EgressPolicy

    @property
    def models(self) -> CatalogSnapshot:
        return self.catalog

    def account_key(self, account_ref: str) -> str:
        return dict(self.authorities)[account_ref]

    def target_key(self, route_id: str) -> str:
        return dict(self.targets)[route_id]

    def validate_binding(self, binding: ResourceBinding) -> None:
        if binding.authority_fingerprint != self.account_key(
            binding.account_ref
        ) or binding.target_fingerprint != self.target_key(binding.route_id):
            raise GatewayAuthorizationError(
                status_code=404,
                code="gateway_resource_not_found",
                message="Stateful resource not found",
            )


class GatewayRuntime:
    def __init__(
        self,
        snapshot: GatewayConfigSnapshot,
        *,
        environ: Mapping[str, str],
        config_path: Path | None = None,
        dependencies: RuntimeDependencies | None = None,
    ) -> None:
        self._environ = environ
        self.config_path = config_path.resolve() if config_path is not None else None
        self.dependencies = dependencies or RuntimeDependencies()
        self._reload_lock = asyncio.Lock()
        self.admission = AdmissionController(
            max_concurrency=snapshot.admission.max_concurrency,
            budget_limit=None,
            queue_limit=snapshot.admission.queue_limit,
            unknown_cost_policy=snapshot.admission.unknown_cost_policy,
        )
        self.resources = ResourceRegistry(max_entries=snapshot.limits.max_resource_bindings)
        self.router = AccountRouter(
            available_accounts={c.id for c in snapshot.credentials if c.enabled}
        )
        self.observability = GatewayObservability()
        self.active_work: dict[str, Any] = {}
        self._refresh_tasks: dict[tuple[int, str, str], asyncio.Task[dict[str, object]]] = {}
        self._retired: list[RuntimeGeneration] = []
        self._pseudonyms: dict[str, str] = {}
        self._ready = True
        self._generation = self._build_generation(snapshot, 1)
        self.router.update_accounts(
            {c.id for c in snapshot.credentials if c.enabled}, dict(self._generation.authorities)
        )

    @property
    def generation(self) -> int:
        return self._generation.number

    @property
    def snapshot(self) -> GatewayConfigSnapshot:
        return self._generation.snapshot

    @property
    def authenticator(self) -> GatewayAuthenticator:
        return self._generation.authenticator

    @property
    def authorizer(self) -> GatewayAuthorizer:
        return self.capture().authorizer

    @property
    def models(self) -> CatalogSnapshot:
        return self.capture().catalog

    @property
    def broker(self) -> CredentialBroker:
        return self.capture().broker

    def capture(self) -> RuntimeGeneration:
        generation = self._generation
        catalog = CatalogSnapshot(
            generation.snapshot,
            accounts=generation.catalog.accounts,
            revision=generation.catalog.revision,
            generation=generation.number,
            now=self.dependencies.clock(),
        )
        return replace(
            generation,
            catalog=catalog,
            authorizer=GatewayAuthorizer(generation.snapshot, catalog),
            broker=self.dependencies.broker or generation.broker,
            http_client=self.dependencies.http_client or generation.http_client,
            egress_policy=self.dependencies.egress_policy,
        )

    async def reload(self, path: Path | None = None) -> ReloadResult:
        # The Python API accepts a trusted path; HTTP controls never accept one.
        selected = self.config_path if path is None else path
        if selected is None:
            return ReloadResult(False, self.generation, "reload_unavailable")
        try:
            snapshot = GatewayConfigSnapshot.load(selected)
            candidate = self._build_generation(snapshot, self.generation + 1)
        except (OSError, ValueError):
            return ReloadResult(False, self.generation, "gateway_configuration_invalid")
        async with self._reload_lock:
            candidate = replace(candidate, number=self.generation + 1)
            catalog = CatalogSnapshot(
                candidate.snapshot,
                accounts=candidate.catalog.accounts,
                revision=self._generation.catalog.revision + 1,
                generation=candidate.number,
                now=self.dependencies.clock(),
            )
            candidate = replace(
                candidate,
                catalog=catalog,
                authorizer=GatewayAuthorizer(candidate.snapshot, catalog),
            )
            self._retired.append(self._generation)
            self._generation = candidate
            self.router.update_accounts(
                {c.id for c in snapshot.credentials if c.enabled}, dict(candidate.authorities)
            )
        return ReloadResult(True, candidate.number)

    def status(self) -> RedactedGatewayStatus:
        generation = self.capture()
        return RedactedGatewayStatus(
            service="headroom",
            profile="gateway",
            generation=generation.number,
            ready=self._ready,
            route_count=len(generation.snapshot.routes),
            credential_count=len(generation.snapshot.credentials),
            config_digest=generation.config_digest,
            catalog_revision=generation.catalog.revision,
        )

    def account_status(self) -> list[dict[str, object]]:
        generation = self.capture()
        return [
            {
                "account": self._pseudonyms.setdefault(record.authority, secrets.token_hex(8)),
                "state": record.state(generation.catalog.captured_at),
                "source_state": record.source_state,
                "provenance": record.provenance,
                "entitlement": record.entitlement,
            }
            for record in generation.catalog.accounts
        ]

    async def revoke(
        self,
        *,
        principal_id: str | None = None,
        route_id: str | None = None,
        account_id: str | None = None,
    ) -> ReloadResult:
        if sum(value is not None for value in (principal_id, route_id, account_id)) != 1:
            raise ValueError("one revocation selector required")
        raw = self.snapshot.model_dump(mode="json")
        items = (
            raw["client_auth"]["principals"]
            if principal_id
            else raw["routes"]
            if route_id
            else raw["credentials"]
        )
        selector = principal_id or route_id or account_id
        if not any(item["id"] == selector for item in items):
            raise ValueError("unknown revocation selector")
        for item in items:
            if item["id"] == selector:
                item["enabled"] = False
        snapshot = GatewayConfigSnapshot.model_validate(raw)
        async with self._reload_lock:
            candidate = self._build_generation(snapshot, self.generation + 1)
            self._retired.append(self._generation)
            self._generation = candidate
            self.router.update_accounts(
                {c.id for c in snapshot.credentials if c.enabled}, dict(candidate.authorities)
            )
        return ReloadResult(True, candidate.number)

    async def refresh_catalog(self) -> dict[str, object]:
        generation = self.capture()
        key = (generation.number, "catalog", "complete")
        task = self._refresh_tasks.get(key)
        if task is None:
            task = asyncio.create_task(self._refresh_complete(generation))
            self._refresh_tasks[key] = task
            task.add_done_callback(lambda done: self._refresh_tasks.pop(key, None))
        return await asyncio.shield(task)

    async def _refresh_complete(self, generation: RuntimeGeneration) -> dict[str, object]:
        semaphore = asyncio.Semaphore(8)

        async def probe(
            route: RouteConfig, account: str
        ) -> tuple[str, str, tuple[str, ...] | None]:
            try:
                async with semaphore:
                    reader = self.dependencies.metadata_reader or self._read_metadata
                    models = await asyncio.wait_for(
                        reader(generation, route, account),
                        timeout=route.catalog.refresh_timeout_seconds,
                    )
                if (
                    not isinstance(models, tuple)
                    or len(models) > 10000
                    or any(not isinstance(model, str) or len(model) > 1024 for model in models)
                ):
                    raise ValueError("invalid metadata")
                return route.id, account, models
            except Exception:
                return route.id, account, None

        results = await asyncio.gather(
            *(
                probe(route, account)
                for route in generation.snapshot.routes
                if route.enabled and route.catalog.source == "provider"
                for account in route.credentials
                if any(
                    r.route_id == route.id
                    and r.account_ref == account
                    and r.source_state != "unavailable"
                    for r in generation.catalog.accounts
                )
            )
        )
        async with self._reload_lock:
            current = self._generation
            if current.number != generation.number:
                return {
                    "generation": current.number,
                    "refreshed": 0,
                    "failed": len(results),
                    "obsolete": True,
                }
            now = self.dependencies.clock()
            records = list(current.catalog.accounts)
            for route_id, account, models in results:
                route = next(r for r in current.snapshot.routes if r.id == route_id)
                for index, record in enumerate(records):
                    if record.route_id != route_id or record.account_ref != account:
                        continue
                    if models is not None:
                        records[index] = replace(
                            record,
                            metadata_available=route.upstream_model in models,
                            observed_at=now,
                            expires_at=now + route.catalog.ttl_seconds,
                            stale_until=now
                            + route.catalog.ttl_seconds
                            + route.catalog.stale_if_error_seconds,
                            refresh_failed=False,
                        )
                    else:
                        records[index] = replace(record, refresh_failed=True)
            if results:
                catalog = CatalogSnapshot(
                    current.snapshot,
                    accounts=tuple(records),
                    revision=current.catalog.revision + 1,
                    generation=current.number,
                    now=now,
                )
                self._generation = replace(
                    current,
                    catalog=catalog,
                    authorizer=GatewayAuthorizer(current.snapshot, catalog),
                )
        return {
            "generation": current.number,
            "catalog_revision": self._generation.catalog.revision,
            "refreshed": sum(models is not None for _route, _account, models in results),
            "failed": sum(models is None for _route, _account, models in results),
        }

    async def _read_metadata(
        self, generation: RuntimeGeneration, route: RouteConfig, account: str
    ) -> tuple[str, ...]:
        if route.provider not in {"openai", "compatible", "anthropic", "gemini"}:
            raise ValueError("provider metadata adapter unavailable")
        lease = await generation.broker.acquire(route, account_ref=account)
        target = (
            route.upstream_origin.rstrip("/") + route.upstream_path_prefix.rstrip("/") + "/models"
        )
        destination = await asyncio.to_thread(
            generation.egress_policy.authorize, lease, target, route=route
        )
        headers = build_managed_upstream_headers(
            {}, lease, target, resolved_addresses=destination.addresses, route=route
        )
        if route.provider == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        request = generation.http_client.build_request("GET", target, headers=headers)
        request.extensions["gateway_destination"] = destination
        with private_transport():
            response = await generation.http_client.send(
                request, stream=True, follow_redirects=False
            )
            try:
                if response.status_code != 200:
                    raise ValueError("metadata unavailable")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 1048576:
                        raise ValueError("metadata response too large")
                payload = json.loads(body)
            finally:
                await response.aclose()
        items = (
            payload.get("models" if route.provider == "gemini" else "data")
            if isinstance(payload, dict)
            else None
        )
        if not isinstance(items, list) or len(items) > 10000:
            raise ValueError("invalid metadata")
        models = []
        for item in items:
            value = (
                item.get("name" if route.provider == "gemini" else "id")
                if isinstance(item, dict)
                else None
            )
            if not isinstance(value, str) or not value or len(value) > 1024:
                raise ValueError("invalid metadata model")
            models.append(value.removeprefix("models/"))
        return tuple(models)

    async def shutdown(self) -> None:
        self._ready = False
        for task in self._refresh_tasks.values():
            task.cancel()
        await asyncio.gather(*tuple(self._refresh_tasks.values()), return_exceptions=True)
        await self.admission.shutdown()
        await self.resources.clear()
        for generation in [self._generation, *self._retired]:
            await generation.http_client.aclose()

    def _build_generation(self, snapshot: GatewayConfigSnapshot, number: int) -> RuntimeGeneration:
        authorities = tuple(
            (
                c.id,
                _fingerprint(
                    {
                        "id": c.id,
                        "provider": c.provider,
                        "source": c.source.model_dump(),
                        "origins": c.allowed_origins,
                        "paths": c.allowed_path_prefixes,
                        "owner": c.owner_group or c.id,
                        "billing": c.billing_group or c.id,
                    }
                ),
            )
            for c in snapshot.credentials
        )
        targets = tuple(
            (
                r.id,
                _fingerprint(
                    {
                        "provider": r.provider,
                        "origin": r.upstream_origin,
                        "path": r.upstream_path_prefix,
                        "model": r.upstream_model,
                        "protocols": r.native_protocols,
                        "contract": r.body_contract,
                    }
                ),
            )
            for r in snapshot.routes
        )
        credentials = {c.id: c for c in snapshot.credentials}
        records = tuple(
            AccountAvailability(
                r.id,
                account,
                dict(authorities)[account],
                ("available" if credentials[account].source.kind in {"env", "none"} else "unknown")
                if credentials[account].enabled
                else "unavailable",
                r.catalog.entitlements.get(account, "unknown"),
                r.catalog.source,
                metadata_available=r.catalog.source == "configured",
            )
            for r in snapshot.routes
            for account in r.credentials
        )
        catalog = CatalogSnapshot(
            snapshot, accounts=records, generation=number, now=self.dependencies.clock()
        )
        authenticator = GatewayAuthenticator(snapshot, self._environ)
        context = tls_context(snapshot)
        return RuntimeGeneration(
            number,
            snapshot,
            authenticator,
            GatewayAuthorizer(snapshot, catalog),
            catalog,
            CredentialBroker.from_snapshot(snapshot, environ=self._environ),
            _fingerprint(snapshot.redacted_dict()),
            authorities,
            targets,
            http_client(snapshot),
            context,
            self.dependencies.egress_policy,
        )
