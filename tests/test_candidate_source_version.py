"""Candidate version assignment must depend only on the selected source tree."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_historical_candidate_version_ignores_future_tags_and_worktree(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "--quiet")
    git("config", "user.email", "candidate-test@example.invalid")
    git("config", "user.name", "Candidate regression")
    manifest = tmp_path / "pyproject.toml"
    manifest.write_text('[project]\nversion = "0.38.0"\n', encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "--quiet", "-m", "chore: first source")
    source = git("rev-parse", "HEAD")
    git("tag", "v0.38.0")
    manifest.write_text('[project]\nversion = "0.39.0"\n', encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "--quiet", "-m", "feat: later source")
    git("tag", "v0.39.0")

    env = os.environ.copy()
    for key in ("PYTHONPATH", "GITHUB_OUTPUT", "MANUAL_VER", "LEVEL"):
        env.pop(key, None)
    env["CANDIDATE_SOURCE_SHA"] = source

    def resolve():
        result = subprocess.run(
            [sys.executable, str(ROOT / "headroom/release_version.py")],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    before = resolve()
    git("tag", "v99.0.0")
    after = resolve()
    assert before["version"] == after["version"] == "0.38.0"
    assert before["npm_version"] == after["npm_version"] == "0.38.0"
    assert before["canonical"] == after["canonical"] == "0.38.0"


def test_candidate_version_is_resolved_before_calling_shared_build():
    import yaml

    candidate = yaml.safe_load(
        (ROOT / ".github/workflows/candidate-artifact.yml").read_text(encoding="utf-8")
    )
    validation = candidate["jobs"]["validate-source"]
    assert validation["outputs"]["version"] == "${{ steps.version.outputs.version }}"
    version = next(step for step in validation["steps"] if step.get("id") == "version")
    assert version["env"]["CANDIDATE_SOURCE_SHA"] == "${{ inputs.source_sha }}"
    assert version["run"] == "python headroom/release_version.py"
    assert candidate["jobs"]["build-and-smoke"]["with"]["resolved_version"] == (
        "${{ needs.validate-source.outputs.version }}"
    )


@pytest.mark.parametrize("source_sha", ["", "main", "A" * 40, "a" * 39])
def test_invalid_candidate_identity_cannot_fall_back_to_release_detection(tmp_path, source_sha):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CANDIDATE_SOURCE_SHA"] = source_sha
    result = subprocess.run(
        [sys.executable, str(ROOT / "headroom/release_version.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Candidate source must be a lowercase full commit SHA" in result.stderr
    assert "version=" not in result.stdout
