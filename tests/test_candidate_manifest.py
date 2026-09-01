"""Tests for fail-closed immutable candidate manifest handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.candidate_manifest import main

SHA = "a" * 40
PRODUCER_SHA = "b" * 40


def _rollout(path: Path, *, eligible: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_version": "1",
                "channel": "stable",
                "registry_digest": "sha256:" + "c" * 64,
                "snapshot_digest": "sha256:" + "d" * 64,
                "unsafe_override": not eligible,
                "qualification_eligible": eligible,
                **(
                    {}
                    if eligible
                    else {"qualification_ineligible_reason": "unsafe_rollout_override_active"}
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


def _create(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "headroom-candidate.tar"
    artifact.write_bytes(b"immutable candidate bytes")
    manifest = tmp_path / "candidate-manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "candidate_manifest.py",
            "create",
            "--artifact",
            str(artifact),
            "--output",
            str(manifest),
            "--source-sha",
            SHA,
            "--producer-sha",
            PRODUCER_SHA,
            "--repository",
            "headroomlabs-ai/headroom",
            "--package",
            "headroom-ai",
            "--version",
            "0.37.0rc1",
            "--workflow",
            "candidate-artifact.yml",
            "--run-id",
            "123",
            "--run-attempt",
            "1",
            "--rollout",
            str(_rollout(tmp_path / "rollout.json")),
            "--created-at",
            "2026-09-01T12:00:00Z",
        ],
    )
    main()
    return artifact, manifest


def test_create_is_canonical_and_verify_accepts_exact_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact, manifest = _create(monkeypatch, tmp_path)
    first = manifest.read_bytes()
    assert first.endswith(b"\n")
    assert b" " not in first

    monkeypatch.setattr(
        "sys.argv",
        [
            "candidate_manifest.py",
            "verify",
            "--artifact",
            str(artifact),
            "--manifest",
            str(manifest),
            "--source-sha",
            SHA,
            "--producer-sha",
            PRODUCER_SHA,
        ],
    )
    main()


@pytest.mark.parametrize("mutation", ["bytes", "filename", "size"])
def test_verify_rejects_changed_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    artifact, manifest = _create(monkeypatch, tmp_path)
    if mutation == "bytes":
        artifact.write_bytes(b"mutable candidate bytes!!")
    elif mutation == "filename":
        artifact = artifact.rename(tmp_path / "renamed.tar")
    else:
        artifact.write_bytes(b"short")
    monkeypatch.setattr(
        "sys.argv",
        [
            "candidate_manifest.py",
            "verify",
            "--artifact",
            str(artifact),
            "--manifest",
            str(manifest),
        ],
    )
    with pytest.raises(ValueError, match="candidate verification failed"):
        main()


def test_create_rejects_ineligible_rollout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = tmp_path / "candidate.tar"
    artifact.write_bytes(b"candidate")
    monkeypatch.setattr(
        "sys.argv",
        [
            "candidate_manifest.py",
            "create",
            "--artifact",
            str(artifact),
            "--output",
            str(tmp_path / "manifest.json"),
            "--source-sha",
            SHA,
            "--producer-sha",
            PRODUCER_SHA,
            "--repository",
            "headroomlabs-ai/headroom",
            "--package",
            "headroom-ai",
            "--version",
            "0.37.0rc1",
            "--workflow",
            "candidate-artifact.yml",
            "--run-id",
            "123",
            "--run-attempt",
            "1",
            "--rollout",
            str(_rollout(tmp_path / "rollout.json", eligible=False)),
        ],
    )
    with pytest.raises(ValueError, match="qualification eligible"):
        main()


def test_cli_rejects_short_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "candidate_manifest.py",
            "verify",
            "--manifest",
            "x",
            "--artifact",
            "y",
            "--source-sha",
            "abc",
        ],
    )
    with pytest.raises(SystemExit):
        main()
