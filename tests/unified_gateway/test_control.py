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


def test_gateway_readyz_is_local_and_non_secret(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret-sentinel")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))

    response = TestClient(app).get("/readyz", headers={"host": "127.0.0.1:8787"})

    assert response.status_code == 200
    assert response.json()["service"] == "headroom"
    assert response.json()["profile"] == "gateway"
    assert "sentinel" not in response.text


def test_gateway_disables_runtime_mutation_and_settings(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    client = TestClient(create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE))))
    headers = {"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"}

    assert client.post("/admin/runtime-env", headers=headers, json={}).status_code == 404
    assert client.post("/settings", headers=headers, json={}).status_code == 404
