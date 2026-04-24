"""Regression tests for SmartCrusher bugs.

Bug 1: _crush_number_array mixes types (string summary + numbers),
       violating the schema-preserving guarantee.
Bug 2: _current_field_semantics is shared instance state, creating
       a race condition when crushing concurrently.
"""

from __future__ import annotations

import builtins
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

from headroom import SmartCrusherConfig
from headroom.transforms import smart_crusher as smart_crusher_module
from headroom.transforms.smart_crusher import (
    CompressionStrategy,
    FieldStats,
    SmartAnalyzer,
    SmartCrusher,
    _calculate_string_entropy,
    _compress_text_within_items,
    _detect_error_items_for_preservation,
    _detect_id_field_statistically,
    _detect_items_by_learned_semantics,
    _detect_rare_status_values,
    _detect_score_field_statistically,
    _detect_sequential_pattern,
    _detect_structural_outliers,
    _get_preserve_field_values,
    _get_within_compressor,
    _hash_field_name,
    _is_uuid_format,
    _item_has_preserve_field_match,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_crusher(max_items: int = 10, min_items: int = 3) -> SmartCrusher:
    config = SmartCrusherConfig(
        enabled=True,
        min_items_to_analyze=min_items,
        min_tokens_to_crush=0,
        max_items_after_crush=max_items,
        variance_threshold=2.0,
    )
    return SmartCrusher(config=config)


# ---------------------------------------------------------------------------
# Bug 1: Number array type mixing
# ---------------------------------------------------------------------------


class TestNumberArraySchemaPreservation:
    """_crush_number_array must return only original numeric values.

    Previously it prepended a stats summary string, producing
    [string, int, int, ...] which violates the schema-preserving
    guarantee and breaks type-aware JSON consumers.
    """

    def test_crushed_number_array_contains_only_numbers(self) -> None:
        """Every element of the crushed array must be int or float."""
        crusher = _make_crusher(max_items=10)
        numbers = list(range(50))  # 0..49, well above the n<=8 passthrough
        crushed, strategy = crusher._crush_number_array(numbers)

        for i, item in enumerate(crushed):
            assert isinstance(item, int | float), (
                f"Item {i} is {type(item).__name__} = {item!r}, expected int/float. "
                f"Schema-preserving guarantee violated."
            )

    def test_crushed_number_array_subset_of_original(self) -> None:
        """Every value in the crushed array must exist in the original."""
        crusher = _make_crusher(max_items=10)
        numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        crushed, _ = crusher._crush_number_array(numbers)

        original_set = set(numbers)
        for item in crushed:
            assert item in original_set, (
                f"Value {item!r} not in original array — generated content detected"
            )

    def test_stats_summary_in_strategy_not_in_array(self) -> None:
        """Statistics should be communicated via strategy string, not array content."""
        crusher = _make_crusher(max_items=5)
        numbers = list(range(100))
        crushed, strategy = crusher._crush_number_array(numbers)

        # Strategy should contain stats info
        assert "number:" in strategy

        # Array should not contain any strings
        strings_in_result = [x for x in crushed if isinstance(x, str)]
        assert strings_in_result == [], f"Found string(s) in numeric array: {strings_in_result}"

    def test_number_array_passthrough_for_small(self) -> None:
        """Arrays with n <= 8 should pass through unchanged."""
        crusher = _make_crusher()
        small = [1, 2, 3, 4, 5]
        crushed, strategy = crusher._crush_number_array(small)
        assert crushed == small
        assert strategy == "number:passthrough"

    def test_number_array_preserves_outliers(self) -> None:
        """Outlier values should be preserved in the crushed output."""
        crusher = _make_crusher(max_items=10)
        # Normal range + extreme outlier
        numbers = [10] * 20 + [10000]
        crushed, strategy = crusher._crush_number_array(numbers)
        assert 10000 in crushed, "Outlier value 10000 was dropped"

    def test_number_array_preserves_boundaries(self) -> None:
        """First and last values should always be kept."""
        crusher = _make_crusher(max_items=5)
        numbers = list(range(100))
        crushed, strategy = crusher._crush_number_array(numbers)
        assert crushed[0] == 0, "First value not preserved"
        assert numbers[-1] in crushed, "Last value not preserved"

    def test_non_finite_passthrough(self) -> None:
        """All-NaN/Inf arrays should return unchanged."""
        crusher = _make_crusher()
        nans = [float("nan")] * 10
        crushed, strategy = crusher._crush_number_array(nans)
        assert strategy == "number:no_finite"
        assert len(crushed) == 10

    def test_full_crush_pipeline_number_array_types(self) -> None:
        """End-to-end: crushing a JSON number array via the public API."""
        crusher = _make_crusher(max_items=10)
        content = json.dumps(list(range(50)))
        result, was_modified, info = crusher._smart_crush_content(content)

        if was_modified:
            parsed = json.loads(result)
            assert isinstance(parsed, list)
            for item in parsed:
                assert isinstance(item, int | float), (
                    f"Public API returned non-numeric item {item!r} in number array"
                )


# ---------------------------------------------------------------------------
# Bug 2: Race condition on _current_field_semantics
# ---------------------------------------------------------------------------


class TestFieldSemanticsThreadSafety:
    """_current_field_semantics must not leak between concurrent crushes.

    Previously it was stored as instance state (self._current_field_semantics)
    which created a race condition when the same SmartCrusher instance
    was used from multiple threads.
    """

    def test_concurrent_crushes_no_cross_contamination(self) -> None:
        """Two concurrent crushes must not share field_semantics state."""
        crusher = _make_crusher(max_items=5)

        # Two different array payloads
        payload_a = json.dumps([{"name": f"item_{i}", "value": i} for i in range(20)])
        payload_b = json.dumps([{"key": f"k_{i}", "score": i * 0.1} for i in range(20)])

        results: dict[str, str] = {}
        errors: list[Exception] = []

        def crush_task(label: str, content: str) -> None:
            try:
                result, modified, info = crusher._smart_crush_content(content)
                results[label] = result
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            # Run many concurrent crushes to increase race probability
            for i in range(20):
                futures.append(executor.submit(crush_task, f"a_{i}", payload_a))
                futures.append(executor.submit(crush_task, f"b_{i}", payload_b))
            for f in as_completed(futures):
                f.result()  # Re-raise exceptions

        assert not errors, f"Concurrent crushes raised errors: {errors}"

        # After all crushes, thread-local state must be clean
        tl = getattr(crusher, "_thread_local", None)
        if tl is not None:
            semantics = getattr(tl, "field_semantics", None)
            assert semantics is None, f"field_semantics leaked in thread-local: {semantics}"


# ---------------------------------------------------------------------------
# Issue 7: Recursion depth limit
# ---------------------------------------------------------------------------


class TestRecursionDepthLimit:
    """_process_value must not crash on deeply nested JSON."""

    def test_deeply_nested_json_does_not_crash(self) -> None:
        """Nesting deeper than _MAX_PROCESS_DEPTH should return value unchanged."""
        crusher = _make_crusher()
        # Build a 100-level nested structure
        nested: dict = {"leaf": "value"}
        for _i in range(100):
            nested = {"level": nested}

        content = json.dumps(nested)
        result, was_modified, info = crusher._smart_crush_content(content)
        # Should not raise RecursionError
        parsed = json.loads(result)
        # The deep structure should be preserved (returned as-is past depth limit)
        assert isinstance(parsed, dict)

    def test_deeply_nested_list_does_not_crash(self) -> None:
        """Deeply nested lists should also be handled safely."""
        crusher = _make_crusher()
        nested: list = ["leaf"]
        for _i in range(100):
            nested = [nested]

        content = json.dumps(nested)
        result, was_modified, info = crusher._smart_crush_content(content)
        parsed = json.loads(result)
        assert isinstance(parsed, list)


class TestHelperCoverage:
    def test_get_within_compressor_caches_available_module(self, monkeypatch) -> None:
        class _FakeKompress:
            pass

        fake_module = SimpleNamespace(
            KompressCompressor=lambda: _FakeKompress(),
            is_kompress_available=lambda: True,
        )

        monkeypatch.setattr(smart_crusher_module, "_within_compressor", None)
        monkeypatch.setattr(smart_crusher_module, "_within_compressor_checked", False)
        monkeypatch.setitem(sys.modules, "headroom.transforms.kompress_compressor", fake_module)

        first = _get_within_compressor()
        second = _get_within_compressor()

        assert isinstance(first, _FakeKompress)
        assert second is first

    def test_get_within_compressor_returns_none_when_import_fails(self, monkeypatch) -> None:
        monkeypatch.setattr(smart_crusher_module, "_within_compressor", None)
        monkeypatch.setattr(smart_crusher_module, "_within_compressor_checked", False)
        monkeypatch.delitem(sys.modules, "headroom.transforms.kompress_compressor", raising=False)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            if name.endswith("kompress_compressor"):
                raise ImportError("missing optional dependency")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert _get_within_compressor() is None
        assert smart_crusher_module._within_compressor_checked is True

    def test_compress_text_within_items_only_replaces_helpful_fields(self, monkeypatch) -> None:
        class _FakeCompressor:
            def compress(self, value, context=""):  # noqa: ANN001, ANN201
                if "raise" in value:
                    raise RuntimeError("boom")
                if "compress me" in value:
                    return SimpleNamespace(compressed="shortened")
                return SimpleNamespace(compressed=value)

        monkeypatch.setattr(
            smart_crusher_module, "_get_within_compressor", lambda: _FakeCompressor()
        )
        long_text = "compress me " + ("x" * 220)
        error_text = "raise " + ("y" * 220)
        items = [
            {"id": 1, "body": long_text, "short": "ok"},
            {"id": 2, "body": error_text},
            {"id": 3, "body": "z" * 220},
        ]

        result = _compress_text_within_items(items, context="ctx")

        assert result is not items
        assert result[0]["body"] == "shortened"
        assert result[1] is items[1]
        assert result[2] is items[2]

    def test_compress_text_within_items_passthroughs_without_compressor(self, monkeypatch) -> None:
        items = [{"body": "x" * 250}]
        monkeypatch.setattr(smart_crusher_module, "_get_within_compressor", lambda: None)

        assert _compress_text_within_items(items) is items

    def test_preserve_field_helpers_match_hashed_field_values(self) -> None:
        item = {"ticket": "ABC-123", "count": 7}
        preserve_hashes = [_hash_field_name("ticket")]

        matches = _get_preserve_field_values(item, preserve_hashes)

        assert matches == [("ticket", "ABC-123")]
        assert _item_has_preserve_field_match(item, preserve_hashes, "show abc-123 details") is True
        assert _item_has_preserve_field_match(item, preserve_hashes, "different query") is False
        assert _item_has_preserve_field_match(item, preserve_hashes, "") is False


class TestStatisticalHelperCoverage:
    def test_uuid_entropy_and_sequential_helpers_cover_key_branches(self) -> None:
        assert _is_uuid_format("550e8400-e29b-41d4-a716-446655440000") is True
        assert _is_uuid_format("not-a-uuid") is False
        assert _calculate_string_entropy("aaaaaaaaaa") < 0.3
        assert _calculate_string_entropy("a1b2c3d4e5f6") > 0.7

        assert _detect_sequential_pattern([1, 2, 3, 4, 5]) is True
        assert _detect_sequential_pattern([5, 4, 3, 2, 1]) is False
        assert _detect_sequential_pattern(["10", "11", "12", "13", "14"], check_order=False) is True

    def test_id_and_score_detectors_handle_uuid_numeric_and_ranked_values(self) -> None:
        uuid_stats = FieldStats(
            name="id",
            field_type="string",
            count=5,
            unique_count=5,
            unique_ratio=1.0,
            is_constant=False,
        )
        uuid_values = [
            "550e8400-e29b-41d4-a716-446655440000",
            "550e8400-e29b-41d4-a716-446655440001",
            "550e8400-e29b-41d4-a716-446655440002",
            "550e8400-e29b-41d4-a716-446655440003",
            "550e8400-e29b-41d4-a716-446655440004",
        ]
        assert _detect_id_field_statistically(uuid_stats, uuid_values) == (True, 0.95)

        numeric_id_stats = FieldStats(
            name="ticket_id",
            field_type="numeric",
            count=6,
            unique_count=6,
            unique_ratio=1.0,
            is_constant=False,
            min_val=1001,
            max_val=1006,
        )
        is_id, confidence = _detect_id_field_statistically(
            numeric_id_stats, [1001, 1002, 1003, 1004, 1005, 1006]
        )
        assert is_id is True
        assert confidence >= 0.85

        score_stats = FieldStats(
            name="score",
            field_type="numeric",
            count=5,
            unique_count=5,
            unique_ratio=1.0,
            is_constant=False,
            min_val=0.71,
            max_val=0.99,
        )
        ranked_items = [
            {"score": 0.99},
            {"score": 0.94},
            {"score": 0.89},
            {"score": 0.81},
            {"score": 0.71},
        ]
        is_score, score_confidence = _detect_score_field_statistically(score_stats, ranked_items)
        assert is_score is True
        assert score_confidence >= 0.4

        ascending_score_items = [
            {"score": 1},
            {"score": 2},
            {"score": 3},
            {"score": 4},
            {"score": 5},
        ]
        ascending_stats = FieldStats(
            name="score",
            field_type="numeric",
            count=5,
            unique_count=5,
            unique_ratio=1.0,
            is_constant=False,
            min_val=1,
            max_val=5,
        )
        assert _detect_score_field_statistically(ascending_stats, ascending_score_items) == (
            False,
            0.0,
        )

    def test_outlier_error_and_learned_semantics_helpers_preserve_important_items(self) -> None:
        items = [{"status": "ok", "kind": "task", "payload": {"n": i}} for i in range(8)] + [
            {"status": "ok", "kind": "task", "extra_debug": True},
            {"status": "failed", "kind": "task", "message": "critical timeout"},
        ]

        outliers = _detect_structural_outliers(items)
        assert 8 in outliers
        assert 9 in outliers
        assert _detect_rare_status_values(items, {"status", "kind"}) == [9]
        assert _detect_error_items_for_preservation(items) == [9]
        item_strings = [json.dumps(item) for item in items]
        assert _detect_error_items_for_preservation(items, item_strings=item_strings) == [9]

        important_status_hash = smart_crusher_module.hashlib.sha256(b"failed").hexdigest()[:8]
        important_payload_hash = smart_crusher_module.hashlib.sha256(
            json.dumps({"n": 3}, sort_keys=True).encode()
        ).hexdigest()[:8]
        field_semantics = {
            _hash_field_name("status"): SimpleNamespace(
                confidence=0.9,
                inferred_type="status",
                is_value_important=lambda value_hash: value_hash == important_status_hash,
            ),
            _hash_field_name("payload"): SimpleNamespace(
                confidence=0.9,
                inferred_type="object",
                is_value_important=lambda value_hash: value_hash == important_payload_hash,
            ),
            _hash_field_name("ignored"): SimpleNamespace(
                confidence=0.1,
                inferred_type="status",
                is_value_important=lambda value_hash: True,
            ),
        }

        assert _detect_items_by_learned_semantics(items, field_semantics) == [3, 9]


class TestAnalyzerHelperCoverage:
    def test_analyze_field_handles_null_unknown_and_overflow_paths(self, monkeypatch) -> None:
        analyzer = SmartAnalyzer(SmartCrusherConfig())

        null_stats = analyzer._analyze_field("missing", [{"other": 1}, {"other": 2}])
        assert null_stats.field_type == "null"
        assert null_stats.is_constant is True

        class _CustomValue:
            pass

        unknown_stats = analyzer._analyze_field("custom", [{"custom": _CustomValue()}])
        assert unknown_stats.field_type == "unknown"

        original_mean = smart_crusher_module.statistics.mean

        def fake_mean(values):  # noqa: ANN001, ANN201
            if list(values) == [1.0, 2.0]:
                raise OverflowError("boom")
            return original_mean(values)

        monkeypatch.setattr(smart_crusher_module.statistics, "mean", fake_mean)
        overflow_stats = analyzer._analyze_field("value", [{"value": 1.0}, {"value": 2.0}])
        assert overflow_stats.min_val is None
        assert overflow_stats.max_val is None
        assert overflow_stats.mean_val is None
        assert overflow_stats.change_points == []

    def test_pattern_temporal_strategy_and_reduction_helpers(self) -> None:
        analyzer = SmartAnalyzer(SmartCrusherConfig(min_items_to_analyze=3))

        log_pattern_field_stats = {
            "message": FieldStats(
                name="message",
                field_type="string",
                count=10,
                unique_count=8,
                unique_ratio=0.8,
                is_constant=False,
                avg_length=40.0,
            ),
            "level": FieldStats(
                name="level",
                field_type="string",
                count=10,
                unique_count=3,
                unique_ratio=0.05,
                is_constant=False,
                avg_length=5.0,
            ),
        }
        assert (
            analyzer._detect_pattern(log_pattern_field_stats, [{"message": "x", "level": "INFO"}])
            == "logs"
        )

        log_strategy_field_stats = {
            **log_pattern_field_stats,
            "message": FieldStats(
                name="message",
                field_type="string",
                count=10,
                unique_count=4,
                unique_ratio=0.4,
                is_constant=False,
                avg_length=40.0,
            ),
        }
        assert (
            analyzer._select_strategy(log_strategy_field_stats, "logs", 10, None)
            == CompressionStrategy.CLUSTER_SAMPLE
        )

        search_items = [
            {"score": 0.99},
            {"score": 0.93},
            {"score": 0.88},
            {"score": 0.82},
            {"score": 0.77},
        ]
        search_field_stats = {
            "score": FieldStats(
                name="score",
                field_type="numeric",
                count=5,
                unique_count=5,
                unique_ratio=1.0,
                is_constant=False,
                min_val=0.77,
                max_val=0.99,
                variance=0.01,
            )
        }
        assert analyzer._detect_pattern(search_field_stats, search_items) == "search_results"
        assert (
            analyzer._select_strategy(search_field_stats, "search_results", 5, None)
            == CompressionStrategy.TOP_N
        )

        temporal_string_stats = {
            "ts": FieldStats(
                name="ts",
                field_type="string",
                count=2,
                unique_count=2,
                unique_ratio=1.0,
                is_constant=False,
                avg_length=20.0,
            )
        }
        assert (
            analyzer._detect_temporal_field(
                temporal_string_stats,
                [{"ts": "2025-01-01T12:00:00Z"}, {"ts": "2025-01-02T12:00:00Z"}],
            )
            is True
        )

        temporal_numeric_stats = {
            "epoch": FieldStats(
                name="epoch",
                field_type="numeric",
                count=2,
                unique_count=2,
                unique_ratio=1.0,
                is_constant=False,
                min_val=1_700_000_000,
                max_val=1_700_000_100,
            )
        }
        assert (
            analyzer._detect_temporal_field(temporal_numeric_stats, [{"epoch": 1_700_000_000}])
            is True
        )
        assert (
            analyzer._estimate_reduction(
                log_strategy_field_stats, CompressionStrategy.CLUSTER_SAMPLE, 10
            )
            == 0.8
        )

    def test_analyze_crushability_distinguishes_unique_and_signaled_cases(self) -> None:
        analyzer = SmartAnalyzer(SmartCrusherConfig())

        repetitive_items = [{"id": 1000 + i, "active": True} for i in range(10)]
        repetitive_analysis = analyzer.analyze_array(repetitive_items)
        assert repetitive_analysis.crushability is not None
        assert repetitive_analysis.crushability.reason == "repetitive_content_with_ids"
        assert repetitive_analysis.crushability.crushable is True

        unique_items = [{"id": 2000 + i, "name": f"user-{i}"} for i in range(10)]
        unique_analysis = analyzer.analyze_array(unique_items)
        assert unique_analysis.crushability is not None
        assert unique_analysis.crushability.reason == "unique_entities_no_signal"
        assert unique_analysis.recommended_strategy == CompressionStrategy.SKIP

        signaled_items = [
            {"id": 3000 + i, "name": f"user-{i}", "score": 1.0 - (i * 0.02)} for i in range(10)
        ]
        signaled_analysis = analyzer.analyze_array(signaled_items)
        assert signaled_analysis.crushability is not None
        assert signaled_analysis.crushability.reason == "unique_entities_with_signal"
        assert signaled_analysis.recommended_strategy == CompressionStrategy.TOP_N
