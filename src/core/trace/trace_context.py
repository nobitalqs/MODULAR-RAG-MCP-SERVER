"""Trace context for observability across pipeline stages.

Provides trace_id, trace_type (query/ingestion), per-stage timing,
finish() lifecycle, and to_dict() serialisation for JSON Lines output.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class TraceContext:
    """Request-scoped trace context that records pipeline stages and timing.

    Attributes:
        trace_id: Unique identifier for this trace.
        trace_type: Either ``"query"`` or ``"ingestion"``.
        started_at: ISO-8601 timestamp when the trace was created.
        finished_at: ISO-8601 timestamp when ``finish()`` was called, or None.
        stages: Ordered list of recorded stage dicts.
        metadata: Arbitrary key/value pairs attached to the trace.
    """

    trace_type: Literal["query", "ingestion"] = "query"
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = field(default=None)
    stages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # internal monotonic clock for accurate elapsed calculation
    _start_mono: float = field(default_factory=time.monotonic, repr=False)
    _finish_mono: float | None = field(default=None, repr=False)
    _stage_timings: dict[str, float] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── recording ─────────────────────────────────────────────────────

    def record_stage(
        self,
        stage_name: str,
        data: dict[str, Any],
        elapsed_ms: float | None = None,
    ) -> None:
        """Record data from a pipeline stage.

        Args:
            stage_name: Name of the stage (e.g. ``"dense_retrieval"``).
            data: Stage-specific payload (method, provider, details …).
            elapsed_ms: Pre-computed elapsed time in ms.  If *None* the
                caller should measure externally, or leave it to the
                ``stage_timer`` context-manager.
        """
        entry: dict[str, Any] = {
            "stage": stage_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        if elapsed_ms is not None:
            entry["elapsed_ms"] = round(elapsed_ms, 2)
        with self._lock:
            if elapsed_ms is not None:
                self._stage_timings[stage_name] = elapsed_ms
            self.stages.append(entry)

    # ── lifecycle ─────────────────────────────────────────────────────

    def finish(self) -> None:
        """Mark the trace as finished and record wall-clock end time."""
        self._finish_mono = time.monotonic()
        self.finished_at = datetime.now(timezone.utc).isoformat()

    # ── timing helpers ────────────────────────────────────────────────

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """Return elapsed time in milliseconds.

        Args:
            stage_name: If given, return the elapsed time recorded for
                that stage.  If *None*, return the total trace elapsed
                time (start → finish, or start → now if not yet
                finished).

        Returns:
            Elapsed milliseconds.

        Raises:
            KeyError: If *stage_name* was provided but not found.
        """
        if stage_name is not None:
            if stage_name not in self._stage_timings:
                raise KeyError(f"Stage '{stage_name}' has no recorded timing")
            return self._stage_timings[stage_name]

        end = self._finish_mono if self._finish_mono is not None else time.monotonic()
        return (end - self._start_mono) * 1000.0

    # ── serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise the trace to a plain dict suitable for ``json.dumps``.

        Returns:
            Dictionary with all trace data.
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": round(self.elapsed_ms(), 2),
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }

    # ── backwards-compat helper used in C5 / C6 ───────────────────────

    def get_stage_data(self, stage_name: str) -> dict[str, Any] | None:
        """Retrieve recorded data for a specific stage.

        Searches stages list (last-write-wins for duplicate names).

        Args:
            stage_name: Name of the stage to retrieve.

        Returns:
            The ``data`` dict of the matching stage, or *None*.
        """
        for entry in reversed(self.stages):
            if entry.get("stage") == stage_name:
                return entry.get("data")
        return None
