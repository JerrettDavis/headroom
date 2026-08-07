from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from headroom.proxy import session_registry as sr
from headroom.proxy.session_registry import ActiveSessionRegistry, ClusterConfig


def test_active_session_registry_writes_local_and_cluster_manifests(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry(
        agent_type="codex",
        session_id="sess-1",
        instance_id="inst-1",
        local_sessions_dir=tmp_path / "sessions",
        cluster=ClusterConfig(
            enabled=True,
            cluster_id="team-gamma",
            cluster_dir=tmp_path / "cluster",
        ),
    )

    payload = registry.heartbeat({"requests": 2, "tokens_saved": 50})

    assert payload["session_id"] == "sess-1"
    local = json.loads((tmp_path / "sessions" / "sess-1" / "session.json").read_text())
    cluster = json.loads(
        (
            tmp_path / "cluster" / "team-gamma" / "sessions" / "inst-1" / "sess-1" / "session.json"
        ).read_text()
    )
    assert local["metrics"]["tokens_saved"] == 50
    assert cluster["cluster"]["cluster_id"] == "team-gamma"
    assert "cluster_dir" not in cluster["cluster"]
    assert str(tmp_path) not in json.dumps(cluster)

    sessions = sr.list_active_sessions(tmp_path / "sessions")
    assert [item["session_id"] for item in sessions] == ["sess-1"]
    assert sr.aggregate_sessions(sessions)["totals"]["tokens_saved"] == 50

    registry.close()
    assert not registry.local_manifest_path.exists()
    assert registry.cluster_manifest_path is not None
    assert not registry.cluster_manifest_path.exists()


def test_heartbeat_tolerates_unavailable_manifest_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_write(path: Path, payload: dict[str, object]) -> None:
        raise PermissionError(path)

    monkeypatch.setattr(sr, "_atomic_write_json", fail_write)
    registry = ActiveSessionRegistry(
        session_id="sess-unwritable",
        instance_id="inst-unwritable",
        local_sessions_dir=tmp_path / "sessions",
        cluster=ClusterConfig(
            enabled=True,
            cluster_id="team-gamma",
            cluster_dir=tmp_path / "cluster",
        ),
    )

    payload = registry.heartbeat({"requests": 3})

    assert payload["session_id"] == "sess-unwritable"
    assert payload["metrics"]["requests"] == 3
    assert registry.snapshot({"requests": 4})["metrics"]["requests"] == 4


def test_manifest_keeps_only_numeric_aggregate_metrics(tmp_path: Path) -> None:
    registry = ActiveSessionRegistry(local_sessions_dir=tmp_path / "sessions")

    payload = registry.heartbeat(
        {
            "requests": 3,
            "tokens_saved": 12.5,
            "savings_storage_path": "/Users/example/.headroom/savings.json",
            "arbitrary_label": "private",
            "failed_requests": True,
        }
    )

    assert payload["metrics"] == {"requests": 3, "tokens_saved": 12.5}
    assert "/Users/example" not in registry.local_manifest_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("cluster_id", ["../outside", "../../outside", ".", "..", "a/b", "a\\b"])
def test_cluster_id_rejects_path_traversal(cluster_id: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single path-safe component"):
        ClusterConfig(enabled=True, cluster_id=cluster_id, cluster_dir=tmp_path)


def test_close_tolerates_unavailable_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ActiveSessionRegistry(
        local_sessions_dir=tmp_path / "sessions",
        cluster=ClusterConfig(True, "team-gamma", tmp_path / "cluster"),
    )
    registry.heartbeat({"requests": 1})
    real_unlink = Path.unlink

    def fail_manifest_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.name == "session.json":
            raise PermissionError(path)
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_manifest_unlink)

    registry.close()


@pytest.mark.asyncio
async def test_periodic_heartbeat_keeps_idle_session_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(sr, "_utc_now", lambda: now)
    registry = ActiveSessionRegistry(
        local_sessions_dir=tmp_path / "sessions",
        stale_after_seconds=6,
    )
    registry.heartbeat({"requests": 1})
    sleeps = 0

    async def advance_clock(_seconds: float) -> None:
        nonlocal now, sleeps
        sleeps += 1
        now += timedelta(seconds=2)
        if sleeps > 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(sr.asyncio, "sleep", advance_clock)

    with pytest.raises(asyncio.CancelledError):
        await registry.run_heartbeat_loop(lambda: {"requests": 1}, interval_seconds=2)

    assert now - registry.started_at > timedelta(seconds=6)
    assert [item["session_id"] for item in sr.list_active_sessions(tmp_path / "sessions")] == [
        registry.session_id
    ]


def test_list_active_sessions_prunes_stale_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 4, 16, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(sr, "_utc_now", lambda: now)
    registry = ActiveSessionRegistry(
        session_id="fresh",
        instance_id="inst",
        local_sessions_dir=tmp_path / "sessions",
    )
    registry.heartbeat({"requests": 1})

    stale_dir = tmp_path / "sessions" / "stale"
    stale_dir.mkdir(parents=True)
    stale_payload = {
        "session_id": "stale",
        "last_heartbeat_at": (now - timedelta(seconds=300)).isoformat().replace("+00:00", "Z"),
        "metrics": {"requests": 99},
    }
    (stale_dir / "session.json").write_text(json.dumps(stale_payload), encoding="utf-8")

    sessions = sr.list_active_sessions(
        tmp_path / "sessions",
        stale_after_seconds=120,
        prune_stale=True,
    )

    assert [item["session_id"] for item in sessions] == ["fresh"]
    assert not (stale_dir / "session.json").exists()
