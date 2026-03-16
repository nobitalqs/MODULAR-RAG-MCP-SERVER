"""Unit tests for RetrievalTextGenerator transform."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.retrieval_text_generator import RetrievalTextGenerator
from src.libs.llm.base_llm import BaseLLM

# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def mock_settings_enabled():
    """Settings with retrieval_text_generator enabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.retrieval_text_generator = {
        "enabled": True,
        "target_doc_types": ["source_code"],
        "max_chunk_length": 2000,
    }
    return settings


@pytest.fixture
def mock_settings_disabled():
    """Settings with retrieval_text_generator disabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.retrieval_text_generator = {"enabled": False}
    return settings


@pytest.fixture
def mock_settings_no_config():
    """Settings with no retrieval_text_generator section."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.retrieval_text_generator = None
    return settings


@pytest.fixture
def mock_llm():
    """Mock LLM that returns a summary."""
    llm = Mock(spec=BaseLLM)
    response = Mock()
    response.content = "This function calculates the invariant mass from two four-vectors using the relativistic energy-momentum relation."
    llm.chat.return_value = response
    return llm


@pytest.fixture
def mock_llm_failing():
    """Mock LLM that raises an exception."""
    llm = Mock(spec=BaseLLM)
    llm.chat.side_effect = RuntimeError("LLM unavailable")
    return llm


@pytest.fixture
def code_chunk():
    """A source_code chunk."""
    return Chunk(
        id="chunk_001",
        text="def compute_mass(p1, p2):\n    e = p1.E + p2.E\n    return math.sqrt(e**2 - px**2)",
        metadata={
            "source_path": "analysis.py",
            "doc_type": "source_code",
            "brief": "Invariant mass calculation",
            "title": "compute_mass",
            "tags": ["compute_mass", "invariant_mass"],
        },
        source_ref="doc_001",
    )


@pytest.fixture
def pdf_chunk():
    """A PDF (non-code) chunk."""
    return Chunk(
        id="chunk_002",
        text="The attention mechanism allows the model to focus on relevant parts of the input.",
        metadata={
            "source_path": "paper.pdf",
            "doc_type": "pdf",
        },
        source_ref="doc_002",
    )


@pytest.fixture
def markdown_chunk():
    """A Markdown chunk."""
    return Chunk(
        id="chunk_003",
        text="# API Reference\n\nThis section describes the REST API endpoints.",
        metadata={
            "source_path": "docs.md",
            "doc_type": "markdown",
        },
        source_ref="doc_003",
    )


# ── Test: LLM summary for source_code ─────────────────────────────


class TestLLMSummary:
    def test_source_code_gets_retrieval_text(self, mock_settings_enabled, mock_llm, code_chunk):
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=mock_llm)
        result = gen.transform([code_chunk])

        assert len(result) == 1
        assert "retrieval_text" in result[0].metadata
        assert result[0].metadata["retrieval_text_by"] == "llm"
        # Original text preserved
        assert result[0].text == code_chunk.text

    def test_llm_receives_truncated_text(self, mock_settings_enabled, mock_llm):
        """Code longer than max_chunk_length is truncated before LLM call."""
        long_code = "x = 1\n" * 500  # ~3000 chars, exceeds 2000
        chunk = Chunk(
            id="long_001",
            text=long_code,
            metadata={"source_path": "big.py", "doc_type": "source_code"},
        )
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=mock_llm)
        gen.transform([chunk])

        # Check the text sent to LLM was truncated
        call_args = mock_llm.chat.call_args
        prompt_text = call_args[0][0][0].content
        assert len(prompt_text) < len(long_code) + 500  # prompt + truncated code


# ── Test: non-code passthrough ─────────────────────────────────────


class TestPassthrough:
    def test_pdf_chunk_unchanged(self, mock_settings_enabled, mock_llm, pdf_chunk):
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=mock_llm)
        result = gen.transform([pdf_chunk])

        assert len(result) == 1
        assert "retrieval_text" not in result[0].metadata
        assert result[0].text == pdf_chunk.text
        mock_llm.chat.assert_not_called()

    def test_markdown_chunk_unchanged(self, mock_settings_enabled, mock_llm, markdown_chunk):
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=mock_llm)
        result = gen.transform([markdown_chunk])

        assert "retrieval_text" not in result[0].metadata
        mock_llm.chat.assert_not_called()

    def test_mixed_chunks_only_code_processed(
        self, mock_settings_enabled, mock_llm, code_chunk, pdf_chunk
    ):
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=mock_llm)
        result = gen.transform([code_chunk, pdf_chunk])

        assert "retrieval_text" in result[0].metadata  # code
        assert "retrieval_text" not in result[1].metadata  # pdf
        assert mock_llm.chat.call_count == 1


# ── Test: LLM failure fallback ─────────────────────────────────────


class TestFallback:
    def test_llm_failure_uses_rule_based(self, mock_settings_enabled, mock_llm_failing, code_chunk):
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=mock_llm_failing)
        result = gen.transform([code_chunk])

        assert len(result) == 1
        assert "retrieval_text" in result[0].metadata
        assert result[0].metadata["retrieval_text_by"] == "rule"
        # Rule-based should contain brief
        assert "Invariant mass calculation" in result[0].metadata["retrieval_text"]

    def test_no_llm_uses_rule_based(self, mock_settings_enabled, code_chunk):
        """When no LLM is provided, falls back to rule-based."""
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=None)
        result = gen.transform([code_chunk])

        assert result[0].metadata["retrieval_text_by"] == "rule"

    def test_rule_based_extracts_signatures(self, mock_settings_enabled):
        """Rule-based fallback extracts function/class signatures."""
        chunk = Chunk(
            id="sig_001",
            text="class MyParser:\n    def parse(self, text):\n        pass\n\ndef helper():\n    pass",
            metadata={
                "source_path": "parser.py",
                "doc_type": "source_code",
                "brief": "Text parser module",
                "tags": ["MyParser"],
            },
        )
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=None)
        result = gen.transform([chunk])
        rt = result[0].metadata["retrieval_text"]

        assert "Text parser module" in rt
        assert "MyParser" in rt or "parse" in rt or "helper" in rt


# ── Test: disabled config ──────────────────────────────────────────


class TestDisabled:
    def test_disabled_passes_through(self, mock_settings_disabled, mock_llm, code_chunk):
        gen = RetrievalTextGenerator(mock_settings_disabled, llm=mock_llm)
        result = gen.transform([code_chunk])

        assert "retrieval_text" not in result[0].metadata
        mock_llm.chat.assert_not_called()

    def test_no_config_passes_through(self, mock_settings_no_config, mock_llm, code_chunk):
        gen = RetrievalTextGenerator(mock_settings_no_config, llm=mock_llm)
        result = gen.transform([code_chunk])

        assert "retrieval_text" not in result[0].metadata


# ── Test: edge cases ───────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_chunk_list(self, mock_settings_enabled, mock_llm):
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=mock_llm)
        result = gen.transform([])
        assert result == []

    def test_immutability(self, mock_settings_enabled, mock_llm, code_chunk):
        """Original chunk must not be mutated."""
        original_metadata = code_chunk.metadata.copy()
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=mock_llm)
        result = gen.transform([code_chunk])

        assert code_chunk.metadata == original_metadata
        assert result[0] is not code_chunk

    def test_trace_context_recorded(self, mock_settings_enabled, mock_llm, code_chunk):
        trace = TraceContext(trace_type="test")
        gen = RetrievalTextGenerator(mock_settings_enabled, llm=mock_llm)
        gen.transform([code_chunk], trace=trace)

        # TraceContext.stages is list[dict], each with "stage" key
        assert any(s.get("stage") == "retrieval_text_generator" for s in trace.stages)

    def test_llm_returns_empty_uses_fallback(self, mock_settings_enabled, code_chunk):
        """If LLM returns empty string, fall back to rule-based."""
        llm = Mock(spec=BaseLLM)
        response = Mock()
        response.content = ""
        llm.chat.return_value = response

        gen = RetrievalTextGenerator(mock_settings_enabled, llm=llm)
        result = gen.transform([code_chunk])

        assert result[0].metadata["retrieval_text_by"] == "rule"
