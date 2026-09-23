"""Deterministic account selection and conservative retry classification."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import NoReturn

from headroom.proxy.gateway.config import RouteConfig
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.errors import GatewayCredentialUnavailable
from headroom.proxy.gateway.resources import ResourceBinding


@dataclass(frozen=True, slots=True)
class AccountSelection:
    account_ref: str
    sticky: bool


@dataclass(frozen=True, slots=True)
class TransportFailure:
    kind: str
    retry_after: float | None


@dataclass(frozen=True, slots=True)
class ProviderContract:
    max_attempts: int
    retryable_failures: frozenset[str]
    max_retry_after: float = 0.0


class RetryDecision:
    @staticmethod
    def classify(
        failure: TransportFailure,
        exposure: str,
        provider_contract: ProviderContract,
    ) -> bool:
        if provider_contract.max_attempts < 2 or exposure != "none":
            return False
        if failure.kind not in provider_contract.retryable_failures:
            return False
        return (
            failure.retry_after is None or failure.retry_after <= provider_contract.max_retry_after
        )


class AccountRouter:
    def update_accounts(
        self, available_accounts: set[str], identities: dict[str, str] | None = None
    ) -> None:
        self._available = frozenset(available_accounts)
        self._identities = identities or {}

    def __init__(self, *, available_accounts: set[str]) -> None:
        self._available = frozenset(available_accounts)
        self._identities = {}
        self._positions: dict[tuple[str, str], int] = {}
        self._cooldowns: dict[tuple[str, str], float] = {}

    def cool_down(self, account_ref: str, *, quota_key: str, until: float) -> None:
        self._cooldowns[(self._identities.get(account_ref, account_ref), quota_key)] = until

    def select(
        self,
        route: RouteConfig,
        principal: GatewayPrincipal,
        resource_binding: ResourceBinding | None = None,
        *,
        now: float | None = None,
        eligible_accounts: frozenset[str] | None = None,
        authority_keys: dict[str, str] | None = None,
        target_key: str | None = None,
    ) -> AccountSelection:
        if route.id not in principal.routes or "inference" not in principal.scopes:
            self._unavailable()
        available = (
            self._available if eligible_accounts is None else self._available & eligible_accounts
        )
        identities = self._identities if authority_keys is None else authority_keys
        if resource_binding is not None:
            if (
                resource_binding.principal_id != principal.id
                or resource_binding.route_id != route.id
                or resource_binding.account_ref not in route.credentials
                or resource_binding.account_ref not in available
            ):
                self._unavailable()
            return AccountSelection(resource_binding.account_ref, sticky=True)

        current = time.time() if now is None else now
        candidates = [
            account
            for account in route.credentials
            if account in available
            and all(
                deadline <= current
                for (cooled_account, _quota_key), deadline in self._cooldowns.items()
                if cooled_account == identities.get(account, account)
            )
        ]
        if not candidates:
            self._unavailable()
        key = (principal.id, target_key or route.id)
        position = self._positions.get(key, 0)
        account = candidates[position % len(candidates)]
        self._positions[key] = position + 1
        return AccountSelection(account, sticky=False)

    @staticmethod
    def _unavailable() -> NoReturn:
        raise GatewayCredentialUnavailable(
            status_code=503,
            code="credential_unavailable",
            message="No eligible gateway account is available",
        )
