"""Atomically published runtime state for the gateway profile."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from headroom.proxy.gateway.admission import AdmissionController
from headroom.proxy.gateway.auth import GatewayAuthenticator, GatewayAuthorizer
from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.control import RedactedGatewayStatus
from headroom.proxy.gateway.credentials import CredentialBroker
from headroom.proxy.gateway.models import ModelRegistry
from headroom.proxy.gateway.observability import GatewayObservability
from headroom.proxy.gateway.resources import ResourceRegistry
from headroom.proxy.gateway.routing import AccountRouter


@dataclass(frozen=True, slots=True)
class ReloadResult:
    applied: bool
    generation: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _RuntimeGeneration:
    number: int
    snapshot: GatewayConfigSnapshot
    authenticator: GatewayAuthenticator
    authorizer: GatewayAuthorizer
    models: ModelRegistry
    broker: CredentialBroker
    resources: ResourceRegistry
    router: AccountRouter


class GatewayRuntime:
    """Own gateway components and replace them only after complete validation."""

    def __init__(self, snapshot: GatewayConfigSnapshot, *, environ: Mapping[str, str]) -> None:
        self._environ = environ
        self._reload_lock = asyncio.Lock()
        self._generation = self._build_generation(snapshot, 1)
        self.admission = AdmissionController(
            budget_limit=None,
            max_concurrency=128,
            queue_limit=128,
            unknown_cost_policy="allow",
        )
        self.observability = GatewayObservability()

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
        return self._generation.authorizer

    @property
    def models(self) -> ModelRegistry:
        return self._generation.models

    @property
    def broker(self) -> CredentialBroker:
        return self._generation.broker

    @property
    def resources(self) -> ResourceRegistry:
        return self._generation.resources

    @property
    def router(self) -> AccountRouter:
        return self._generation.router

    async def reload(self, path: Path) -> ReloadResult:
        try:
            snapshot = GatewayConfigSnapshot.load(path)
            candidate = self._build_generation(snapshot, self.generation + 1)
        except (OSError, ValueError) as exc:
            return ReloadResult(False, self.generation, type(exc).__name__)
        async with self._reload_lock:
            if candidate.number != self.generation + 1:
                candidate = self._build_generation(snapshot, self.generation + 1)
            self._generation = candidate
            return ReloadResult(True, candidate.number)

    def status(self) -> RedactedGatewayStatus:
        generation = self._generation
        return RedactedGatewayStatus(
            service="headroom",
            profile="gateway",
            generation=generation.number,
            ready=True,
            route_count=len(generation.snapshot.routes),
            credential_count=len(generation.snapshot.credentials),
        )

    async def shutdown(self) -> None:
        await self.admission.shutdown()
        await self.resources.clear()

    def _build_generation(self, snapshot: GatewayConfigSnapshot, number: int) -> _RuntimeGeneration:
        return _RuntimeGeneration(
            number=number,
            snapshot=snapshot,
            authenticator=GatewayAuthenticator(snapshot, self._environ),
            authorizer=GatewayAuthorizer(snapshot),
            models=ModelRegistry(snapshot),
            broker=CredentialBroker.from_snapshot(snapshot, environ=self._environ),
            resources=ResourceRegistry(),
            router=AccountRouter(
                available_accounts={item.id for item in snapshot.credentials if item.enabled}
            ),
        )
