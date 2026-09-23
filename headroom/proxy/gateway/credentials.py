"""Opaque credential leases and single-flight acquisition."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from headroom.proxy.gateway.config import CredentialConfig, GatewayConfigSnapshot, RouteConfig
from headroom.proxy.gateway.errors import GatewayCredentialUnavailable


class SecretHandle:
    """A non-serializable secret value revealed only at the egress seam."""

    __slots__ = ("__value",)

    def __init__(self, value: Any) -> None:
        self.__value = value

    def _reveal_for_egress(self) -> Any:
        return self.__value

    def __repr__(self) -> str:
        return "SecretHandle(**redacted**)"


@dataclass(frozen=True, slots=True)
class CredentialLease:
    credential_id: str
    provider: str
    account_ref: str
    allowed_origins: tuple[str, ...]
    allowed_path_prefixes: tuple[str, ...]
    expires_at: float | None
    generation: int
    secret: SecretHandle = field(repr=False)
    source_kind: str = "env"
    project: str | None = None
    region: str | None = None

    def redacted_dict(self) -> dict[str, object]:
        return {
            "credential_id": self.credential_id,
            "provider": self.provider,
            "account_ref": self.account_ref,
            "allowed_origins": self.allowed_origins,
            "allowed_path_prefixes": self.allowed_path_prefixes,
            "expires_at": self.expires_at,
            "generation": self.generation,
        }

    def authorization_headers(self) -> dict[str, str]:
        secret = self.secret._reveal_for_egress()
        if not isinstance(secret, str):
            raise TypeError("credential requires provider-specific request signing")
        if self.provider == "compatible" and not secret:
            return {}
        if self.provider == "anthropic":
            return {"x-api-key": secret}
        if self.provider == "gemini":
            return {"x-goog-api-key": secret}
        return {"authorization": f"Bearer {secret}"}


class CredentialSource(Protocol):
    async def acquire(self, *, now: float) -> CredentialLease: ...

    async def invalidate(self, lease: CredentialLease, reason: str) -> None: ...


class CredentialBroker:
    """Acquire and cache one current immutable lease per configured identity."""

    def __init__(self, sources: Mapping[str, CredentialSource]) -> None:
        self._sources = dict(sources)
        self._leases: dict[str, CredentialLease] = {}
        self._generations: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {
            credential_id: asyncio.Lock() for credential_id in self._sources
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: GatewayConfigSnapshot,
        *,
        environ: Mapping[str, str],
    ) -> CredentialBroker:
        from headroom.proxy.gateway.credential_sources.aws import AwsChainCredentialSource
        from headroom.proxy.gateway.credential_sources.environment import (
            EnvironmentCredentialSource,
        )
        from headroom.proxy.gateway.credential_sources.gcp import GcpAdcCredentialSource

        sources: dict[str, CredentialSource] = {}
        for config in snapshot.credentials:
            config = CredentialConfig.model_validate(config.model_dump())
            if config.source.kind == "env":
                sources[config.id] = EnvironmentCredentialSource(config, environ)
            elif config.source.kind == "gcp-adc":
                sources[config.id] = GcpAdcCredentialSource(config)
            elif config.source.kind == "aws-chain":
                sources[config.id] = AwsChainCredentialSource(config)
            elif config.source.kind == "none":
                sources[config.id] = NoCredentialSource(config)
        return cls(sources)

    async def acquire(
        self,
        route: RouteConfig,
        account_ref: str | None = None,
    ) -> CredentialLease:
        unavailable = False
        for credential_id in route.credentials:
            source = self._sources.get(credential_id)
            if source is None:
                unavailable = True
                continue
            if account_ref is not None and credential_id != account_ref:
                continue
            async with self._locks[credential_id]:
                now = time.time()
                cached = self._leases.get(credential_id)
                if cached is not None and (
                    cached.expires_at is None or cached.expires_at > now + 30.0
                ):
                    return cached
                try:
                    lease = await source.acquire(now=now)
                except GatewayCredentialUnavailable:
                    unavailable = True
                    continue
                if lease.provider != route.provider or lease.credential_id != credential_id:
                    raise GatewayCredentialUnavailable(
                        status_code=503,
                        code="credential_unavailable",
                        message="Provider credential identity mismatch",
                    )
                generation = self._generations.get(credential_id, 0) + 1
                lease = replace(lease, generation=generation)
                self._generations[credential_id] = generation
                self._leases[credential_id] = lease
                return lease
        raise GatewayCredentialUnavailable(
            status_code=503,
            code="credential_unavailable",
            message="No configured provider credential is available",
        ) from (None if unavailable else None)

    async def invalidate(self, lease: CredentialLease, reason: str) -> None:
        source = self._sources.get(lease.credential_id)
        if source is None:
            return
        async with self._locks[lease.credential_id]:
            current = self._leases.get(lease.credential_id)
            if current is not lease:
                return
            self._leases.pop(lease.credential_id)
            await source.invalidate(lease, reason)


class NoCredentialSource:
    """Explicit no-auth source restricted by config validation to compatible routes."""

    def __init__(self, config: CredentialConfig) -> None:
        self._config = config

    async def acquire(self, *, now: float) -> CredentialLease:
        del now
        return CredentialLease(
            credential_id=self._config.id,
            provider=self._config.provider,
            account_ref=self._config.id,
            allowed_origins=self._config.allowed_origins,
            allowed_path_prefixes=self._config.allowed_path_prefixes,
            expires_at=None,
            generation=1,
            secret=SecretHandle(""),
        )

    async def invalidate(self, lease: CredentialLease, reason: str) -> None:
        del lease, reason
