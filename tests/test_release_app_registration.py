"""Credential-boundary checks for the local App registration helper."""

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "register_release_app", ROOT / "scripts/register-release-app.py"
)
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


@pytest.mark.parametrize(
    "query",
    [
        "code=abc",
        "state=wrong&code=abc",
        "state=expected&code=a&code=b",
        "state=expected&code=../bad",
    ],
)
def test_rejects_untrusted_callback(query):
    with pytest.raises(ValueError):
        app.callback_code(query, "expected")


def test_accepts_single_state_bound_code():
    assert app.callback_code("state=expected&code=abc_123", "expected") == "abc_123"


def test_configuration_keeps_key_out_of_arguments():
    with patch.object(
        app.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
    ) as run:
        app.configure_repository("JerrettDavis/headroom", {"pem": "PRIVATE KEY", "id": 123})
    calls = run.call_args_list
    assert calls[0].kwargs["input"] == "PRIVATE KEY"
    assert "PRIVATE KEY" not in str(calls[0].args)
    assert calls[1].kwargs["input"] == "123"


def test_configuration_failure_does_not_disclose_stderr():
    with patch.object(
        app.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, stderr="PRIVATE KEY")
    ):
        with pytest.raises(RuntimeError) as error:
            app.configure_repository("JerrettDavis/headroom", {"pem": "PRIVATE KEY", "id": 123})
    assert "PRIVATE KEY" not in str(error.value)


def test_manifest_permissions_are_minimal():
    manifest = json.loads((ROOT / "scripts/release-app-manifest.json").read_text())
    assert manifest["default_permissions"] == {"contents": "write", "pull_requests": "write"}
    assert manifest["default_events"] == []
    assert manifest["hook_attributes"]["active"] is False
    assert manifest["hook_attributes"]["url"].startswith("https://")


def test_workflow_writers_require_repository_scoped_app_tokens():
    import yaml

    for name, job in [
        ("release-please.yml", "release-please"),
        ("release-metadata-sync.yml", "sync"),
    ]:
        workflow = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        steps = workflow["jobs"][job]["steps"]
        mint = next(step for step in steps if step.get("id") == "app-token")
        assert "continue-on-error" not in mint
        assert "if" not in mint
        assert mint["with"]["repositories"] == "${{ github.event.repository.name }}"
        assert mint["with"]["permission-contents"] == "write"
        if job == "release-please":
            assert mint["with"]["permission-pull-requests"] == "write"
            writer = next(
                step
                for step in steps
                if step.get("uses", "").startswith("googleapis/release-please-action@")
            )
            assert writer["with"]["token"] == "${{ steps.app-token.outputs.token }}"
        else:
            writer = next(step for step in steps if "SYNC_TOKEN" in step.get("env", {}))
            assert writer["env"]["SYNC_TOKEN"] == "${{ steps.app-token.outputs.token }}"
