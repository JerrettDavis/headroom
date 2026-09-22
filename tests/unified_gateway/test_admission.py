from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from headroom.proxy.gateway.admission import AdmissionController, AdmissionRequest
from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


@pytest.mark.asyncio
async def test_atomic_budget_reservation_allows_only_one_boundary_request() -> None:
    controller = AdmissionController(
        budget_limit=1.0,
        max_concurrency=2,
        queue_limit=0,
        unknown_cost_policy="block",
    )

    results = await asyncio.gather(
        controller.try_reserve(AdmissionRequest("principal-a", estimated_cost=0.75)),
        controller.try_reserve(AdmissionRequest("principal-a", estimated_cost=0.75)),
    )

    assert sorted(result.allowed for result in results) == [False, True]
    for result in results:
        if result.reservation is not None:
            await result.reservation.release()


@pytest.mark.asyncio
async def test_cancellation_and_double_release_never_leak_concurrency() -> None:
    controller = AdmissionController(
        budget_limit=None,
        max_concurrency=1,
        queue_limit=0,
        unknown_cost_policy="allow",
    )
    first = await controller.reserve(AdmissionRequest("principal-a", estimated_cost=0.1))
    denied = await controller.try_reserve(AdmissionRequest("principal-b", estimated_cost=0.1))
    assert denied.allowed is False

    await first.release()
    await first.release()
    second = await controller.reserve(AdmissionRequest("principal-b", estimated_cost=0.1))
    await second.finalize(actual_cost=0.2)

    assert controller.active_count == 0
    assert controller.committed_cost == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_strict_budget_rejects_unknown_estimate() -> None:
    controller = AdmissionController(
        budget_limit=5.0,
        max_concurrency=1,
        queue_limit=0,
        unknown_cost_policy="block",
    )

    result = await controller.try_reserve(AdmissionRequest("principal-a", estimated_cost=None))

    assert result.allowed is False
    assert result.reason == "unknown_cost"


def test_http_admission_denial_precedes_credential_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))
    app.state.gateway_admission = AdmissionController(
        budget_limit=0.0,
        max_concurrency=1,
        queue_limit=0,
        unknown_cost_policy="block",
    )

    class Broker:
        async def acquire(self, *_args, **_kwargs):
            raise AssertionError("credential acquisition must not run after admission denial")

    app.state.gateway_credential_broker = Broker()

    response = TestClient(app).post(
        "/v1/responses",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
        json={"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "input": "hello"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "gateway_admission_unknown_cost"
