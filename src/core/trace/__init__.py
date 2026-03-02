"""
Trace Module - Request-level tracing infrastructure.

Components:
- TraceContext: Per-request trace with stage recording
- TraceCollector: Trace persistence to JSON Lines
"""

from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext

__all__ = ["TraceContext", "TraceCollector"]
