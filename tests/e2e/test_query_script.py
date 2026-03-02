"""End-to-End tests for scripts/query.py.

Behavioural tests (no API keys / ingested data needed):
- ``--help`` flag
- Missing config file
- Missing ``--query`` argument

Integration tests (require ingested data + API keys):
- Single query with results
- Verbose mode intermediate output
- No-rerank flag

Run behavioural tests:
    pytest tests/e2e/test_query_script.py -v -m "not integration"

Run all (including integration):
    pytest tests/e2e/test_query_script.py -v -m integration -s
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = str(PROJECT_ROOT / "scripts" / "query.py")


# ── Helpers ──────────────────────────────────────────────────────────


def _run(
    query: str = "test query",
    collection: str = "default",
    top_k: int = 5,
    config: str | None = None,
    no_rerank: bool = False,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run query.py as a subprocess."""
    cmd = [
        sys.executable, SCRIPT,
        "--query", query,
        "--collection", collection,
        "--top-k", str(top_k),
    ]

    if config:
        cmd.extend(["--config", config])
    if no_rerank:
        cmd.append("--no-rerank")
    if verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
        env=env,
        encoding="utf-8",
        errors="replace",
    )


# ── Behavioural tests (no API / data needed) ────────────────────────


class TestQueryCLI:
    """Tests that verify CLI behaviour without requiring API access."""

    def test_help_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "--query" in result.stdout
        assert "--collection" in result.stdout
        assert "--top-k" in result.stdout
        assert "--no-rerank" in result.stdout
        assert "--verbose" in result.stdout

    def test_missing_query_arg(self) -> None:
        result = subprocess.run(
            [sys.executable, SCRIPT],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 2
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_invalid_config(self) -> None:
        result = _run(
            query="test",
            config="/nonexistent/config.yaml",
        )
        assert result.returncode == 2
        assert "not found" in result.stdout.lower()

    def test_parse_args_defaults(self) -> None:
        """Verify parse_args returns correct defaults."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.query import parse_args

        args = parse_args(["--query", "hello world"])
        assert args.query == "hello world"
        assert args.collection == "default"
        assert args.top_k == 10
        assert args.no_rerank is False
        assert args.verbose is False

    def test_parse_args_all_flags(self) -> None:
        """Verify parse_args handles all flags correctly."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.query import parse_args

        args = parse_args([
            "--query", "azure配置",
            "--collection", "tech_docs",
            "--top-k", "5",
            "--no-rerank",
            "--verbose",
        ])
        assert args.query == "azure配置"
        assert args.collection == "tech_docs"
        assert args.top_k == 5
        assert args.no_rerank is True
        assert args.verbose is True


# ── Unit tests for formatting helpers ────────────────────────────────


class TestFormattingHelpers:
    """Tests for _format_filters and _print_results."""

    def test_format_filters_empty(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.query import _format_filters

        assert _format_filters({}) == "(none)"

    def test_format_filters_single(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.query import _format_filters

        result = _format_filters({"collection": "docs"})
        assert "collection=docs" in result

    def test_format_filters_multiple(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.query import _format_filters

        result = _format_filters({"collection": "docs", "type": "pdf"})
        assert "collection=docs" in result
        assert "type=pdf" in result

    def test_print_results_empty(self, capsys) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.query import _print_results

        _print_results([], top_k=5, title="TEST")
        captured = capsys.readouterr()
        assert "TEST" in captured.out
        assert "returned=0" in captured.out

    def test_print_results_with_items(self, capsys) -> None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.query import _print_results
        from src.core.types import RetrievalResult

        results = [
            RetrievalResult(
                chunk_id="c1",
                score=0.95,
                text="Hello world content",
                metadata={"source_path": "doc.pdf", "page_num": 1},
            ),
            RetrievalResult(
                chunk_id="c2",
                score=0.80,
                text="Another chunk",
                metadata={"source_path": "doc.pdf"},
            ),
        ]

        _print_results(results, top_k=5, title="RESULTS")
        captured = capsys.readouterr()
        assert "RESULTS" in captured.out
        assert "returned=2" in captured.out
        assert "#01" in captured.out
        assert "0.9500" in captured.out
        assert "source_path=doc.pdf" in captured.out
        assert "page_num=1" in captured.out
        assert "#02" in captured.out
        assert "0.8000" in captured.out


# ── Integration tests (require ingested data) ────────────────────────


@pytest.mark.integration
class TestQueryIntegration:
    """Tests that run actual queries — need API keys and ingested data."""

    def test_query_default_collection(self) -> None:
        result = _run(query="test query", verbose=True)
        # Should either return results or friendly empty message
        assert result.returncode in (0, 1)
        assert "Modular RAG Query Script" in result.stdout

    def test_query_no_rerank(self) -> None:
        result = _run(query="test query", no_rerank=True, verbose=True)
        assert result.returncode in (0, 1)
        if "Reranking disabled" in result.stdout:
            pass  # expected when reranker disabled
        # Should not crash

    def test_query_custom_top_k(self) -> None:
        result = _run(query="test query", top_k=3)
        assert result.returncode in (0, 1)
