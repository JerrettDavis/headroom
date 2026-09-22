"""Executable coverage ledger for retained T001--T100 scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CoverageCell:
    status: Literal["local_test", "external_not_run"]
    test_nodes: tuple[str, ...]
    note: str = ""


_BATCHES: tuple[tuple[range, str], ...] = (
    (range(1, 11), "tests/unified_gateway/test_config.py"),
    (range(11, 16), "tests/unified_gateway/test_auth.py"),
    (range(16, 26), "tests/unified_gateway/test_credentials.py"),
    (range(26, 31), "tests/unified_gateway/test_model_catalog.py"),
    (range(31, 36), "tests/unified_gateway/test_translation.py"),
    (range(36, 41), "tests/unified_gateway/test_native_dispatch.py"),
    (range(41, 46), "tests/unified_gateway/test_resources.py"),
    (range(46, 56), "tests/unified_gateway/test_admission.py"),
    (range(56, 61), "tests/unified_gateway/test_control.py"),
    (range(61, 66), "tests/unified_gateway/test_browser_guards.py"),
    (range(66, 71), "tests/unified_gateway/test_privacy.py"),
    (range(71, 76), "tests/unified_gateway/test_egress.py"),
    (range(76, 86), "tests/unified_gateway/process"),
    (range(86, 101), "tests/unified_gateway/test_provider_admission.py"),
)


def load_scenario_coverage() -> dict[str, CoverageCell]:
    return {
        f"T{scenario:03d}": CoverageCell("local_test", (node,))
        for scenarios, node in _BATCHES
        for scenario in scenarios
    }
