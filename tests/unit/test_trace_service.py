"""Unit tests for TraceService -- JSONL trace file parsing and querying.

Tests verify:
- Loading from JSONL files (valid, malformed, empty, missing)
- Filtering by trace_type
- Reverse-chronological ordering
- Limit enforcement
- get_trace by ID
- get_stage_timings extraction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trace(
    trace_id: str = "t1",
    trace_type: str = "ingestion",
    started_at: str = "2026-01-15T10:00:00Z",
    total_elapsed_ms: float = 500.0,
    stages: list | None = None,
    metadata: dict | None = None,
) -> Dict[str, Any]:
    """Build a trace dict matching TraceContext.to_dict() format."""
    return {
        "trace_id": trace_id,
        "trace_type": trace_type,
        "started_at": started_at,
        "finished_at": "2026-01-15T10:00:01Z",
        "total_elapsed_ms": total_elapsed_ms,
        "stages": stages or [],
        "metadata": metadata or {},
    }


def _write_jsonl(path: Path, traces: list[Dict[str, Any]]) -> None:
    """Write traces as JSONL."""
    with path.open("w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")


@pytest.fixture()
def traces_file(tmp_path: Path) -> Path:
    """Return path to a JSONL file with sample traces."""
    path = tmp_path / "traces.jsonl"
    traces = [
        _make_trace("t1", "ingestion", "2026-01-15T10:00:00Z", 500.0,
                     metadata={"source_path": "/tmp/a.pdf"}),
        _make_trace("t2", "query", "2026-01-15T11:00:00Z", 120.0,
                     metadata={"query": "hello"}),
        _make_trace("t3", "ingestion", "2026-01-15T12:00:00Z", 800.0,
                     metadata={"source_path": "/tmp/b.pdf"}),
        _make_trace("t4", "query", "2026-01-15T09:00:00Z", 50.0,
                     metadata={"query": "world"}),
    ]
    _write_jsonl(path, traces)
    return path


# ===================================================================
# Tests: list_traces
# ===================================================================

class TestListTraces:
    def test_returns_all_traces(self, traces_file: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        svc = TraceService(traces_path=traces_file)
        result = svc.list_traces()

        assert len(result) == 4

    def test_filter_by_type_ingestion(self, traces_file: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        svc = TraceService(traces_path=traces_file)
        result = svc.list_traces(trace_type="ingestion")

        assert len(result) == 2
        assert all(t["trace_type"] == "ingestion" for t in result)

    def test_filter_by_type_query(self, traces_file: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        svc = TraceService(traces_path=traces_file)
        result = svc.list_traces(trace_type="query")

        assert len(result) == 2
        assert all(t["trace_type"] == "query" for t in result)

    def test_reverse_chronological_order(self, traces_file: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        svc = TraceService(traces_path=traces_file)
        result = svc.list_traces()

        timestamps = [t["started_at"] for t in result]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_limit(self, traces_file: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        svc = TraceService(traces_path=traces_file)
        result = svc.list_traces(limit=2)

        assert len(result) == 2
        # Should be the 2 newest
        assert result[0]["trace_id"] == "t3"
        assert result[1]["trace_id"] == "t2"

    def test_empty_file(self, tmp_path: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        path = tmp_path / "empty.jsonl"
        path.write_text("")

        svc = TraceService(traces_path=path)
        assert svc.list_traces() == []

    def test_missing_file(self, tmp_path: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        path = tmp_path / "nonexistent.jsonl"
        svc = TraceService(traces_path=path)
        assert svc.list_traces() == []

    def test_malformed_lines_skipped(self, tmp_path: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        path = tmp_path / "mixed.jsonl"
        with path.open("w") as fh:
            fh.write(json.dumps(_make_trace("t1")) + "\n")
            fh.write("NOT VALID JSON\n")
            fh.write("{bad json\n")
            fh.write(json.dumps(_make_trace("t2")) + "\n")
            fh.write("\n")  # blank line

        svc = TraceService(traces_path=path)
        result = svc.list_traces()
        assert len(result) == 2

    def test_filter_nonexistent_type_returns_empty(self, traces_file: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        svc = TraceService(traces_path=traces_file)
        result = svc.list_traces(trace_type="evaluation")
        assert result == []


# ===================================================================
# Tests: get_trace
# ===================================================================

class TestGetTrace:
    def test_found(self, traces_file: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        svc = TraceService(traces_path=traces_file)
        result = svc.get_trace("t2")

        assert result is not None
        assert result["trace_id"] == "t2"
        assert result["trace_type"] == "query"

    def test_not_found(self, traces_file: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        svc = TraceService(traces_path=traces_file)
        assert svc.get_trace("nonexistent") is None

    def test_empty_file_returns_none(self, tmp_path: Path):
        from src.observability.dashboard.services.trace_service import TraceService

        path = tmp_path / "empty.jsonl"
        path.write_text("")
        svc = TraceService(traces_path=path)
        assert svc.get_trace("any") is None


# ===================================================================
# Tests: get_stage_timings
# ===================================================================

class TestGetStageTimings:
    def test_extracts_timings(self):
        from src.observability.dashboard.services.trace_service import TraceService

        trace = _make_trace(stages=[
            {"stage": "load", "timestamp": "2026-01-01T00:00:00Z",
             "data": {"text_length": 5000}, "elapsed_ms": 100.5},
            {"stage": "split", "timestamp": "2026-01-01T00:00:01Z",
             "data": {"chunk_count": 10}, "elapsed_ms": 50.2},
            {"stage": "embed", "timestamp": "2026-01-01T00:00:02Z",
             "data": {"dense_vector_count": 10}, "elapsed_ms": 200.0},
        ])

        svc = TraceService()
        timings = svc.get_stage_timings(trace)

        assert len(timings) == 3
        assert timings[0]["stage_name"] == "load"
        assert timings[0]["elapsed_ms"] == 100.5
        assert timings[0]["data"]["text_length"] == 5000
        assert timings[1]["stage_name"] == "split"
        assert timings[2]["stage_name"] == "embed"

    def test_empty_stages(self):
        from src.observability.dashboard.services.trace_service import TraceService

        trace = _make_trace(stages=[])
        svc = TraceService()
        assert svc.get_stage_timings(trace) == []

    def test_missing_stages_key(self):
        from src.observability.dashboard.services.trace_service import TraceService

        trace = {"trace_id": "t1"}  # no "stages" key
        svc = TraceService()
        assert svc.get_stage_timings(trace) == []

    def test_non_dict_data_defaults_to_empty(self):
        from src.observability.dashboard.services.trace_service import TraceService

        trace = _make_trace(stages=[
            {"stage": "load", "data": "not a dict", "elapsed_ms": 10},
        ])
        svc = TraceService()
        timings = svc.get_stage_timings(trace)

        assert len(timings) == 1
        assert timings[0]["data"] == {}

    def test_missing_elapsed_ms_defaults_to_zero(self):
        from src.observability.dashboard.services.trace_service import TraceService

        trace = _make_trace(stages=[
            {"stage": "load", "data": {}},
        ])
        svc = TraceService()
        timings = svc.get_stage_timings(trace)

        assert timings[0]["elapsed_ms"] == 0

    def test_preserves_stage_order(self):
        from src.observability.dashboard.services.trace_service import TraceService

        stages = [
            {"stage": "load", "data": {}, "elapsed_ms": 10},
            {"stage": "split", "data": {}, "elapsed_ms": 20},
            {"stage": "transform", "data": {}, "elapsed_ms": 30},
            {"stage": "embed", "data": {}, "elapsed_ms": 40},
            {"stage": "upsert", "data": {}, "elapsed_ms": 50},
        ]
        trace = _make_trace(stages=stages)
        svc = TraceService()
        timings = svc.get_stage_timings(trace)

        names = [t["stage_name"] for t in timings]
        assert names == ["load", "split", "transform", "embed", "upsert"]
