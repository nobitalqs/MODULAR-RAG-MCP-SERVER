"""Query Traces page -- browse query trace history with stage waterfall.

Layout:
1. Optional keyword search filter
2. Trace list (reverse-chronological, filtered to trace_type=="query")
3. Detail view: stage waterfall + Dense vs Sparse comparison + Rerank delta
4. Per-stage detail tabs: Query Processing / Dense / Sparse / Fusion / Rerank
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


def render() -> None:
    """Render the Query Traces page."""
    st.header("Query Traces")

    svc = TraceService()
    traces = svc.list_traces(trace_type="query")

    if not traces:
        st.info("No query traces recorded yet. Run a query first!")
        return

    # -- Keyword filter ------------------------------------------------
    keyword = st.text_input(
        "Search by query keyword",
        value="",
        key="qt_keyword",
    )
    if keyword.strip():
        kw = keyword.strip().lower()
        traces = [
            t
            for t in traces
            if kw in str(t.get("metadata", {})).lower()
            or kw in str(t.get("stages", [])).lower()
        ]

    st.subheader(f"Query History ({len(traces)})")

    for idx, trace in enumerate(traces):
        started = trace.get("started_at", "-")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "-"
        meta = trace.get("metadata", {})
        query_text = meta.get("query", "")
        source = meta.get("source", "unknown")

        # Expander title: truncated query text
        query_preview = (
            query_text[:40] + "..." if len(query_text) > 40 else query_text
        ) if query_text else "-"
        expander_title = (
            f'"{query_preview}" | {total_label} | {started[:19]}'
        )

        with st.expander(expander_title, expanded=(idx == 0)):
            # -- 1. Query overview ----------------------------------------
            st.markdown("#### Query")
            col_q, col_meta = st.columns([3, 1])
            with col_q:
                st.markdown(f"> {query_text}")
            with col_meta:
                st.markdown(f"**Source:** `{source}`")
                st.markdown(f"**Top-K:** `{meta.get('top_k', '-')}`")
                st.markdown(f"**Collection:** `{meta.get('collection', '-')}`")

            st.divider()

            # -- 2. Overview metrics --------------------------------------
            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            dense_d = stages_by_name.get("dense_retrieval", {}).get("data") or {}
            sparse_d = stages_by_name.get("sparse_retrieval", {}).get("data") or {}
            fusion_d = stages_by_name.get("fusion", {}).get("data") or {}
            rerank_d = stages_by_name.get("rerank", {}).get("data") or {}

            dense_count = dense_d.get("result_count", 0)
            sparse_count = sparse_d.get("result_count", 0)
            fusion_count = fusion_d.get("result_count", 0)
            rerank_count = rerank_d.get("output_count", 0)

            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            with rc1:
                st.metric("Dense Hits", dense_count)
            with rc2:
                st.metric("Sparse Hits", sparse_count)
            with rc3:
                st.metric("Fused", fusion_count or (dense_count + sparse_count))
            with rc4:
                st.metric("After Rerank", rerank_count if rerank_d else "-")
            with rc5:
                st.metric("Total Time", total_label)

            st.divider()

            # -- 3. Stage timing waterfall --------------------------------
            main_stage_names = (
                "query_processing",
                "dense_retrieval",
                "sparse_retrieval",
                "fusion",
                "rerank",
            )
            main_timings = [
                t for t in timings if t["stage_name"] in main_stage_names
            ]
            if main_timings:
                st.markdown("#### Stage Timings")
                chart_data = {
                    t["stage_name"]: t["elapsed_ms"] for t in main_timings
                }
                st.bar_chart(chart_data, horizontal=True)
                st.table(
                    [
                        {
                            "Stage": t["stage_name"],
                            "Elapsed (ms)": round(t["elapsed_ms"], 2),
                        }
                        for t in main_timings
                    ]
                )

            st.divider()

            # -- 4. Per-stage detail tabs ---------------------------------
            st.markdown("#### Stage Details")

            tab_defs: List[tuple[str, str]] = []
            if "query_processing" in stages_by_name:
                tab_defs.append(("Query Processing", "query_processing"))
            if "dense_retrieval" in stages_by_name:
                tab_defs.append(("Dense Retrieval", "dense_retrieval"))
            if "sparse_retrieval" in stages_by_name:
                tab_defs.append(("Sparse Retrieval", "sparse_retrieval"))
            if "fusion" in stages_by_name:
                tab_defs.append(("Fusion (RRF)", "fusion"))
            if "rerank" in stages_by_name:
                tab_defs.append(("Rerank", "rerank"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"{elapsed:.1f} ms")

                        if key == "query_processing":
                            _render_query_processing_stage(data)
                        elif key == "dense_retrieval":
                            _render_retrieval_stage(data, "Dense")
                        elif key == "sparse_retrieval":
                            _render_retrieval_stage(data, "Sparse")
                        elif key == "fusion":
                            _render_fusion_stage(data)
                        elif key == "rerank":
                            _render_rerank_stage(data)
            else:
                st.info("No stage details available.")

            # NOTE: Ragas evaluate button will be added in Phase H4.


# ===================================================================
# Per-stage renderers
# ===================================================================


def _render_query_processing_stage(data: Dict[str, Any]) -> None:
    """Render Query Processing stage: original query -> keywords."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Query**")
        st.info(data.get("original_query", "-"))
    with c2:
        st.markdown("**Method**")
        st.code(data.get("method", "-"))

    keywords = data.get("keywords", [])
    if keywords:
        st.markdown("**Extracted Keywords**")
        st.markdown(" ".join(f"`{kw}`" for kw in keywords))
    else:
        st.warning("No keywords extracted.")


def _render_retrieval_stage(data: Dict[str, Any], label: str) -> None:
    """Render Dense or Sparse retrieval stage: method, counts, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "-"))
    with c2:
        extra = data.get("provider", data.get("keyword_count", "-"))
        extra_label = "Provider" if "provider" in data else "Keywords"
        st.metric(extra_label, extra)
    with c3:
        st.metric("Results", data.get("result_count", 0))

    st.markdown(f"**Top-K requested:** `{data.get('top_k', '-')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(
            chunks, prefix=f"{label.lower().replace(' ', '_')}_chunk"
        )
    else:
        st.info(f"No {label.lower()} results returned.")


def _render_fusion_stage(data: Dict[str, Any]) -> None:
    """Render Fusion (RRF) stage: input lists, fused result count, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "rrf"))
    with c2:
        st.metric("Input Lists", data.get("input_lists", "-"))
    with c3:
        st.metric("Fused Results", data.get("result_count", 0))

    st.markdown(f"**Top-K:** `{data.get('top_k', '-')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix="fusion_chunk")
    else:
        st.info("No fusion results.")


def _render_rerank_stage(data: Dict[str, Any]) -> None:
    """Render Rerank stage: method, input/output counts, reranked chunk list."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Method", data.get("method", "-"))
    with c2:
        st.metric("Provider", data.get("provider", "-"))
    with c3:
        st.metric("Input", data.get("input_count", "-"))
    with c4:
        st.metric("Output", data.get("output_count", "-"))

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix="rerank_chunk")
    else:
        st.info("No reranked results.")


def _render_chunk_list(
    chunks: List[Dict[str, Any]], prefix: str = "chunk"
) -> None:
    """Render a list of chunk dicts as expandable cards with score indicators.

    Score colour coding:
    - >= 0.8: [HIGH]
    - >= 0.5: [MED]
    - <  0.5: [LOW]
    """
    for ci, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        title = chunk.get("title", "")

        # Score indicator
        if score >= 0.8:
            score_bar = "[HIGH]"
        elif score >= 0.5:
            score_bar = "[MED]"
        else:
            score_bar = "[LOW]"

        header = f"{score_bar} #{ci + 1} -- Score: `{score:.4f}`"
        if title:
            header += f" -- {title}"

        with st.expander(header, expanded=False):
            cols = st.columns([2, 3])
            with cols[0]:
                st.caption(f"Chunk ID: `{chunk_id}`")
            with cols[1]:
                if source:
                    st.caption(f"Source: `{source}`")

            if text:
                st.text_area(
                    f"{prefix}_{ci}",
                    value=text,
                    height=max(80, min(len(text) // 2, 400)),
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.caption("(No text available)")
