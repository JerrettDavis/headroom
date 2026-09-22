from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import websockets
from fastapi.testclient import TestClient

from headroom.proxy.gateway.auth import GatewayAuthorizer
from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.websocket import authorize_response_create_frame
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


def test_every_websocket_generation_turn_is_reauthorized() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    authorizer = GatewayAuthorizer(snapshot)
    principal = GatewayPrincipal(
        id="local-app",
        scopes=frozenset({"inference"}),
        routes=frozenset({"openai-native"}),
    )
    first = json.dumps(
        {
            "type": "response.create",
            "response": {"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "input": "one"},
        }
    )

    route, payload = authorize_response_create_frame(first, principal, authorizer)

    assert route.id == "openai-native"
    assert payload["input"] == "one"
    with pytest.raises(GatewayAuthorizationError, match="(?i)model"):
        authorize_response_create_frame(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"model": "ungranted-model", "input": "two"},
                }
            ),
            principal,
            authorizer,
            expected_route_id=route.id,
        )


@pytest.mark.parametrize(
    "frame",
    [
        "not-json",
        json.dumps({"type": "response.create", "response": {"input": "missing model"}}),
        json.dumps({"type": "session.update", "session": {}}),
    ],
)
def test_invalid_generation_frame_fails_before_upstream(frame: str) -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    principal = GatewayPrincipal(
        id="local-app",
        scopes=frozenset({"inference"}),
        routes=frozenset({"openai-native"}),
    )

    with pytest.raises(GatewayAuthorizationError):
        authorize_response_create_frame(frame, principal, GatewayAuthorizer(snapshot))


def test_gateway_websocket_reauthorizes_second_turn_before_forwarding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    sent: list[str] = []

    class FakeUpstream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, frame: str) -> None:
            sent.append(frame)

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Future()

    fake = FakeUpstream()
    monkeypatch.setattr(websockets, "connect", lambda *_args, **_kwargs: fake)
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))
    first = {
        "type": "response.create",
        "response": {"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "input": "one"},
    }
    second = {
        "type": "response.create",
        "response": {"model": "ungranted-model", "input": "two"},
    }

    with TestClient(app).websocket_connect(
        "/v1/responses",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
    ) as socket:
        socket.send_json(first)
        socket.send_json(second)
        error = socket.receive_json()

    assert error["type"] == "error"
    assert error["error"]["code"] == "gateway_model_unavailable"
    assert sent == [json.dumps(first, separators=(",", ":"))]
