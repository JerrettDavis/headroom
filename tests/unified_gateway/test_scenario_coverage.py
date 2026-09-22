from __future__ import annotations

import subprocess
import sys

from tests.unified_gateway.scenario_coverage import load_scenario_coverage


def test_every_scenario_has_an_executable_or_explicit_external_cell() -> None:
    coverage = load_scenario_coverage()
    assert set(coverage) == {f"T{index:03d}" for index in range(1, 101)}
    assert all(cell.status in {"local_test", "external_not_run"} for cell in coverage.values())
    assert all(cell.test_nodes for cell in coverage.values() if cell.status == "local_test")


def test_every_local_coverage_node_is_collectable() -> None:
    nodes = sorted(
        {
            node
            for cell in load_scenario_coverage().values()
            if cell.status == "local_test"
            for node in cell.test_nodes
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *nodes],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
