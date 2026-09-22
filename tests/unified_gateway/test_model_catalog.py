from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

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


def test_model_catalog_contains_only_principal_granted_routes(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))

    response = TestClient(app).get(
        "/v1/models",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == [
        "REPLACE_WITH_ENABLED_ANTHROPIC_MODEL",
        "REPLACE_WITH_ENABLED_GEMINI_MODEL",
        "REPLACE_WITH_ENABLED_OPENAI_MODEL",
    ]
