"""Tests for Docker-bridge wrap preparation flows."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from headroom.cli.main import main


def _set_test_home(monkeypatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)


def test_wrap_claude_prepare_only_skips_host_binary_lookup() -> None:
    runner = CliRunner()

    with patch("headroom.cli.wrap._prepare_wrap_rtk") as prepare_rtk:
        with patch("headroom.cli.wrap.shutil.which") as which_mock:
            result = runner.invoke(main, ["wrap", "claude", "--prepare-only"])

    assert result.exit_code == 0, result.output
    prepare_rtk.assert_called_once()
    which_mock.assert_not_called()


def test_wrap_codex_prepare_only_updates_config(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=None):
        result = runner.invoke(main, ["wrap", "codex", "--prepare-only", "--port", "8787"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".codex" / "config.toml"
    assert config_file.exists()
    assert 'model_provider = "headroom"' in config_file.read_text()
    assert 'base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text()


def test_wrap_aider_prepare_only_injects_conventions(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=Path("rtk")):
            result = runner.invoke(main, ["wrap", "aider", "--prepare-only"])

        assert result.exit_code == 0, result.output
        conventions = Path("CONVENTIONS.md")
        assert conventions.exists()
        assert "headroom:rtk-instructions" in conventions.read_text()


def test_wrap_cursor_prepare_only_injects_cursorrules(monkeypatch, tmp_path: Path) -> None:
    _set_test_home(monkeypatch, tmp_path)
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=str(tmp_path)):
        with patch("headroom.cli.wrap._ensure_rtk_binary", return_value=Path("rtk")):
            result = runner.invoke(main, ["wrap", "cursor", "--prepare-only"])

        assert result.exit_code == 0, result.output
        cursorrules = Path(".cursorrules")
        assert cursorrules.exists()
        assert "headroom:rtk-instructions" in cursorrules.read_text()


def test_wrap_openclaw_prepare_only_emits_config_without_python_default() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "wrap",
            "openclaw",
            "--prepare-only",
            "--gateway-provider-id",
            "codex",
            "--gateway-provider-id",
            "anthropic",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["enabled"] is True
    assert payload["config"]["proxyPort"] == 8787
    assert payload["config"]["gatewayProviderIds"] == ["codex", "anthropic"]
    assert "pythonPath" not in payload["config"]


def test_unwrap_openclaw_prepare_only_preserves_unmanaged_config() -> None:
    runner = CliRunner()
    existing_entry = json.dumps(
        {
            "enabled": True,
            "config": {
                "pythonPath": "C:\\Python312\\python.exe",
                "proxyPort": 8787,
                "customFlag": True,
            },
        }
    )

    result = runner.invoke(
        main,
        [
            "unwrap",
            "openclaw",
            "--prepare-only",
            "--existing-entry-json",
            existing_entry,
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"enabled": False, "config": {"customFlag": True}}
