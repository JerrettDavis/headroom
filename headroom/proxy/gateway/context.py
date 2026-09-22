"""Request-scoped gateway authorization state."""

from __future__ import annotations

from dataclasses import dataclass

from headroom.proxy.gateway.config import Protocol, RouteConfig


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
