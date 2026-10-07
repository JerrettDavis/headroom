from unittest.mock import Mock

import pytest
from headroom_claude_mod import startup


def test_existing_companion_does_not_spawn(monkeypatch):
    spawn = Mock()
    monkeypatch.setattr(startup.subprocess, "Popen", spawn)
    startup.start_proxy("http://127.0.0.1:8787", Mock())
    spawn.assert_not_called()


def test_start_proxy_waits_for_companion(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(startup.importlib.util, "find_spec", lambda _: object())
    child = Mock()
    child.poll.return_value = None
    spawn = Mock(return_value=child)
    monkeypatch.setattr(startup.subprocess, "Popen", spawn)
    check = Mock(side_effect=[ValueError("offline"), {}])
    startup.start_proxy("http://127.0.0.1:8787", check)
    args = spawn.call_args.args[0]
    assert args == [
        startup.sys.executable,
        "-c",
        "from headroom.cli import main; main()",
        "proxy",
        "--host",
        "127.0.0.1",
        "--port",
        "8787",
        "--proxy-extension",
        "claude_mod",
    ]
    assert spawn.call_args.kwargs["stdin"] == startup.subprocess.DEVNULL
    assert spawn.call_args.kwargs["close_fds"]


def test_start_failure_reports_log(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(startup.importlib.util, "find_spec", lambda _: object())
    child = Mock()
    child.poll.return_value = 1
    monkeypatch.setattr(startup.subprocess, "Popen", Mock(return_value=child))
    with pytest.raises(ValueError, match="could not start.*proxy.log"):
        startup.start_proxy("http://127.0.0.1:8787", Mock(side_effect=ValueError("offline")))
