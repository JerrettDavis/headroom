"""Isolated real-process gateway harness."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(slots=True)
class GatewayProcess:
    process: subprocess.Popen[str]
    base_url: str

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def start_gateway_process(tmp_path: Path) -> GatewayProcess:
    repository = Path(__file__).parents[3]
    example = (
        repository
        / "docs"
        / "proposals"
        / "unified-api-gateway"
        / "examples"
        / "gateway.api-keys.json"
    )
    raw = json.loads(example.read_text(encoding="utf-8"))
    port = _unused_loopback_port()
    raw["runtime"]["port"] = port
    config_path = tmp_path / "gateway.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    private_home = tmp_path / "home"
    private_home.mkdir()
    allowed_names = {
        "COMSPEC",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "WINDIR",
    }
    environment = {name: value for name, value in os.environ.items() if name.upper() in allowed_names}
    environment.update(
        {
            "HOME": str(private_home),
            "USERPROFILE": str(private_home),
            "HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret",
            "OPENAI_API_KEY": "provider-secret-openai",
            "ANTHROPIC_API_KEY": "provider-secret-anthropic",
            "GEMINI_API_KEY": "provider-secret-gemini",
            "HEADROOM_REQUIRE_RUST_CORE": "false",
            "HEADROOM_SKIP_UPDATE_CHECK": "1",
            "HEADROOM_SKIP_UPSTREAM_CHECK": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "headroom.cli",
            "proxy",
            "--gateway",
            "--gateway-config",
            str(config_path),
        ],
        cwd=repository,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    gateway = GatewayProcess(process=process, base_url=f"http://127.0.0.1:{port}")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise RuntimeError(f"gateway process exited ({process.returncode}):\n{output}")
        try:
            if httpx.get(f"{gateway.base_url}/readyz", timeout=0.5).status_code == 200:
                return gateway
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    gateway.close()
    raise RuntimeError("gateway process did not become ready")


def _unused_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
