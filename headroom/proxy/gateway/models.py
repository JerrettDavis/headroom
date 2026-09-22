"""Principal-filtered gateway model catalog."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from headroom.proxy.gateway.config import GatewayConfigSnapshot, Protocol, RouteConfig
from headroom.proxy.gateway.context import GatewayPrincipal


class Capability(str, Enum):
    """Stable operations a published gateway route may expose."""

    GENERATE = "generate"
    STREAM = "stream"


@dataclass(frozen=True, slots=True)
class RouteCapabilities:
    protocols: tuple[Protocol, ...]
    operations: frozenset[Capability]


@dataclass(frozen=True, slots=True)
class PublishedRoute:
    id: str
    route_id: str
    protocols: tuple[Protocol, ...]
    body_contract: str


class ModelRegistry:
    def __init__(self, snapshot: GatewayConfigSnapshot) -> None:
        self._routes = snapshot.routes

    def visible_routes(self, principal: GatewayPrincipal) -> tuple[PublishedRoute, ...]:
        return tuple(
            PublishedRoute(
                id=route.public_model,
                route_id=route.id,
                protocols=route.ingress_protocols,
                body_contract=route.body_contract,
            )
            for route in sorted(self._routes, key=lambda item: item.public_model)
            if route.id in principal.routes and "models" in principal.scopes
        )

    def route_for_model(self, public_model: str) -> RouteConfig | None:
        return next((route for route in self._routes if route.public_model == public_model), None)

    def route_for_id(self, route_id: str) -> RouteConfig | None:
        return next((route for route in self._routes if route.id == route_id), None)
