"""Request-scoped gateway authorization state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from headroom.proxy.gateway.config import Protocol, RouteConfig

if TYPE_CHECKING:
    from headroom.proxy.gateway.routing import AccountSelection


@dataclass(frozen=True, slots=True)
class GatewayPrincipal:
    id: str
    scopes: frozenset[str]
    routes: frozenset[str]


@dataclass(frozen=True, slots=True)
class GatewayRequestContext:
    principal: GatewayPrincipal
    route: RouteConfig
    ingress_protocol: Protocol
    request_id: str
    snapshot_generation: int = 1
    account_selection: AccountSelection | None = None
