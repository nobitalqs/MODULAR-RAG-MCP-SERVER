"""Evaluation Panel page -- run evaluations and view metrics.

Layout:
1. Evaluation config summary (provider, metrics, enabled status)
2. Golden test set selection + top-k slider
3. Run evaluation button
4. Aggregate metrics display (metric cards)
5. Per-query detail expanders with individual metrics
6. Optional: export report as JSON
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_SET = "tests/fixtures/golden_test_set.json"
REPORTS_DIR = Path("logs/eval_reports")


def render() -> None:
    """Render the Evaluation Panel page."""
    st.header("Evaluation Panel")

    # -- 1. Load settings and show config summary -------------------------
    settings = _load_settings_safe()
    if settings is None:
        st.error("Failed to load settings. Check config/settings.yaml.")
        return

    eval_cfg = settings.evaluation
    _render_config_summary(eval_cfg)

    st.divider()

    # -- 2. Evaluation controls -------------------------------------------
    st.subheader("Run Evaluation")

    col_file, col_topk = st.columns([3, 1])
    with col_file:
        golden_path = st.text_input(
            "Golden Test Set Path",
            value=DEFAULT_GOLDEN_SET,
            key="eval_golden_path",
        )
    with col_topk:
        top_k = st.slider(
            "Top-K", min_value=1, max_value=50, value=10, key="eval_top_k"
        )

    collection = st.text_input(
        "Collection filter (optional)",
        value="",
        key="eval_collection",
    )

    # -- 3. Run button ----------------------------------------------------
    run_clicked = st.button("Run Evaluation", type="primary", key="eval_run_btn")

    if run_clicked:
        _run_evaluation(settings, golden_path, top_k, collection or None)

    # -- 4. Show previous results from session state ----------------------
    if "eval_report" in st.session_state:
        st.divider()
        _render_report(st.session_state["eval_report"])


# =====================================================================
# Config summary
# =====================================================================


def _render_config_summary(eval_cfg: Any) -> None:
    """Display evaluation configuration as metric cards."""
    c1, c2, c3 = st.columns(3)
    with c1:
        status = "Enabled" if eval_cfg.enabled else "Disabled"
        st.metric("Evaluation", status)
    with c2:
        st.metric("Provider", eval_cfg.provider)
    with c3:
        st.metric("Metrics", ", ".join(eval_cfg.metrics) if eval_cfg.metrics else "-")


# =====================================================================
# Run evaluation
# =====================================================================


def _run_evaluation(
    settings: Any,
    golden_path: str,
    top_k: int,
    collection: Optional[str],
) -> None:
    """Execute EvalRunner and store report in session state."""
    from src.libs.evaluator import EvaluatorFactory
    from src.observability.evaluation.eval_runner import EvalRunner

    path = Path(golden_path)
    if not path.exists():
        st.error(f"Golden test set not found: {golden_path}")
        return

    with st.spinner("Building evaluator..."):
        try:
            evaluator = EvaluatorFactory.create(settings)
        except Exception as exc:
            st.error(f"Failed to create evaluator: {exc}")
            return

    hybrid_search = _build_hybrid_search_safe(settings)

    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
    )

    with st.spinner("Running evaluation..."):
        try:
            report = runner.run(
                test_set_path=golden_path,
                top_k=top_k,
                collection=collection,
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            return

    st.session_state["eval_report"] = report.to_dict()
    st.success(
        f"Evaluation complete: {len(report.query_results)} queries, "
        f"{report.total_elapsed_ms:.0f} ms"
    )


# =====================================================================
# Report rendering
# =====================================================================


def _render_report(report_dict: Dict[str, Any]) -> None:
    """Render an evaluation report from its dict representation."""
    st.subheader("Evaluation Results")

    # -- Aggregate metrics ------------------------------------------------
    st.markdown("#### Aggregate Metrics")
    agg = report_dict.get("aggregate_metrics", {})
    if agg:
        cols = st.columns(len(agg))
        for col, (name, value) in zip(cols, agg.items()):
            with col:
                st.metric(name, f"{value:.4f}")
    else:
        st.info("No aggregate metrics available.")

    # -- Summary info -----------------------------------------------------
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Evaluator", report_dict.get("evaluator_name", "-"))
    with sc2:
        st.metric("Queries", report_dict.get("query_count", 0))
    with sc3:
        elapsed = report_dict.get("total_elapsed_ms", 0)
        st.metric("Total Time", f"{elapsed:.0f} ms")

    st.divider()

    # -- Per-query details ------------------------------------------------
    st.markdown("#### Per-Query Results")
    query_results = report_dict.get("query_results", [])
    for idx, qr in enumerate(query_results):
        query_text = qr.get("query", "")
        preview = query_text[:50] + "..." if len(query_text) > 50 else query_text
        metrics = qr.get("metrics", {})
        elapsed_ms = qr.get("elapsed_ms", 0)

        metrics_str = ", ".join(
            f"{k}: {v:.4f}" for k, v in metrics.items()
        ) if metrics else "no metrics"

        with st.expander(
            f"#{idx + 1} | {preview} | {elapsed_ms:.0f} ms | {metrics_str}",
            expanded=False,
        ):
            st.markdown(f"**Query:** {query_text}")

            # Metrics row
            if metrics:
                mcols = st.columns(len(metrics))
                for mc, (mname, mval) in zip(mcols, metrics.items()):
                    with mc:
                        st.metric(mname, f"{mval:.4f}")

            # Retrieved chunks
            chunk_ids = qr.get("retrieved_chunk_ids", [])
            if chunk_ids:
                st.markdown(f"**Retrieved chunks ({len(chunk_ids)}):**")
                st.code(", ".join(chunk_ids))
            else:
                st.caption("No chunks retrieved (HybridSearch not configured).")

            # Generated answer
            answer = qr.get("generated_answer")
            if answer:
                st.markdown("**Generated Answer:**")
                st.text_area(
                    f"answer_{idx}",
                    value=answer,
                    height=80,
                    disabled=True,
                    label_visibility="collapsed",
                )

    st.divider()

    # -- Export button -----------------------------------------------------
    st.markdown("#### Export Report")
    report_json = json.dumps(report_dict, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Report (JSON)",
        data=report_json,
        file_name="eval_report.json",
        mime="application/json",
        key="eval_download_btn",
    )


# =====================================================================
# Helpers
# =====================================================================


def _load_settings_safe() -> Any:
    """Load settings, returning None on failure."""
    try:
        from src.core.settings import load_settings
        return load_settings()
    except Exception as exc:
        logger.error("Failed to load settings: %s", exc)
        return None


def _build_hybrid_search_safe(settings: Any) -> Any:
    """Build HybridSearch from settings, returning None on failure."""
    try:
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.query_processor import QueryProcessor
        from src.libs.embedding import EmbeddingFactory
        from src.libs.vector_store import VectorStoreFactory
        from src.ingestion.storage.bm25_indexer import BM25Indexer

        embedding = EmbeddingFactory.create(settings)
        vector_store = VectorStoreFactory.create(settings)
        bm25_indexer = BM25Indexer(settings=settings)

        dense = create_dense_retriever(
            settings=settings,
            embedding=embedding,
            vector_store=vector_store,
        )
        sparse = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
        )
        query_processor = QueryProcessor(settings=settings)

        return create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense,
            sparse_retriever=sparse,
        )
    except Exception as exc:
        logger.warning(
            "HybridSearch not available (will evaluate without retrieval): %s", exc
        )
        return None
