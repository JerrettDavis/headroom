from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.unified_gateway.process.harness import GatewayProcess, start_gateway_process


@pytest.fixture(scope="module")
def gateway_process(tmp_path_factory: pytest.TempPathFactory) -> Iterator[GatewayProcess]:
    path: Path = tmp_path_factory.mktemp("gateway-process")
    process = start_gateway_process(path)
    try:
        yield process
    finally:
        process.close()
