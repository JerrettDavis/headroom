from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from headroom.proxy.gateway.config import GatewayConfigSnapshot

PROPOSAL = Path(__file__).parents[2] / "docs" / "proposals" / "unified-api-gateway"


def _example() -> dict[str, object]:
    return json.loads((PROPOSAL / "examples" / "gateway.api-keys.json").read_text())


def test_loads_version_one_example_as_immutable_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(_example()), encoding="utf-8")

    snapshot = GatewayConfigSnapshot.load(path)

    assert snapshot.version == 1
    assert snapshot.runtime.profile == "gateway"
    assert snapshot.routes[0].ingress_protocols == ("openai-chat", "openai-responses")
    with pytest.raises(ValidationError):
        snapshot.runtime.port = 9999  # type: ignore[misc]


def test_rejects_unknown_security_setting(tmp_path: Path) -> None:
    raw = _example()
    raw["client_auth"]["allow_anonymous"] = True  # type: ignore[index]
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="allow_anonymous"):
        GatewayConfigSnapshot.load(path)


def test_rejects_duplicate_public_model_and_missing_credential_reference(tmp_path: Path) -> None:
    raw = _example()
    routes = raw["routes"]  # type: ignore[index]
    routes.append({**routes[0], "id": "duplicate-route", "credentials": ["missing"]})
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="duplicate public_model|unknown credential"):
        GatewayConfigSnapshot.load(path)


def test_redacted_dict_contains_references_not_resolved_secret_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret-sentinel")
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(_example()), encoding="utf-8")

    rendered = json.dumps(GatewayConfigSnapshot.load(path).redacted_dict())

    assert "OPENAI_API_KEY" in rendered
    assert "provider-secret-sentinel" not in rendered
