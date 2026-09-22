from __future__ import annotations

from pathlib import Path

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.resources import ResourceBinding
from headroom.proxy.gateway.routing import AccountRouter

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


def test_account_router_round_robins_only_route_credentials() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    route = snapshot.routes[0].model_copy(update={"credentials": ("openai-api", "openai-api-2")})
    router = AccountRouter(available_accounts={"openai-api", "openai-api-2", "anthropic-api"})
    principal = GatewayPrincipal(
        id="principal-a",
        scopes=frozenset({"inference"}),
        routes=frozenset({route.id}),
    )

    assert router.select(route, principal).account_ref == "openai-api"
    assert router.select(route, principal).account_ref == "openai-api-2"
    assert router.select(route, principal).account_ref == "openai-api"


def test_resource_binding_forces_sticky_account_even_during_cooldown() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    route = snapshot.routes[0].model_copy(update={"credentials": ("openai-api", "openai-api-2")})
    router = AccountRouter(available_accounts={"openai-api", "openai-api-2"})
    router.cool_down("openai-api", quota_key="openai", until=200.0)
    binding = ResourceBinding(
        provider_id="resp_1",
        principal_id="principal-a",
        route_id=route.id,
        account_ref="openai-api",
        adapter="openai-responses",
        expires_at=None,
    )
    principal = GatewayPrincipal(
        id="principal-a",
        scopes=frozenset({"inference"}),
        routes=frozenset({route.id}),
    )

    selection = router.select(route, principal, resource_binding=binding, now=100.0)

    assert selection.account_ref == "openai-api"
    assert selection.sticky is True


def test_unbound_selection_skips_cooled_account() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    route = snapshot.routes[0].model_copy(update={"credentials": ("openai-api", "openai-api-2")})
    router = AccountRouter(available_accounts={"openai-api", "openai-api-2"})
    router.cool_down("openai-api", quota_key="openai", until=200.0)
    principal = GatewayPrincipal(
        id="principal-a",
        scopes=frozenset({"inference"}),
        routes=frozenset({route.id}),
    )

    assert router.select(route, principal, now=100.0).account_ref == "openai-api-2"
