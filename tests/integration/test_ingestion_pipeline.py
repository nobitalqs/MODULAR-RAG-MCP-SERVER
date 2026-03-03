"""Integration tests for the Ingestion Pipeline.

These tests require real API keys and PDF fixtures.  They are marked
with ``@pytest.mark.integration`` and are skipped in normal runs.

Run explicitly with:
    pytest -m integration tests/integration/test_ingestion_pipeline.py -v -s

Test Data (place in tests/fixtures/sample_documents/):
- complex_technical_doc.pdf  : multi-chapter, images, tables (~21 KB)
- simple.pdf                 : basic single-page PDF for regression
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline, PipelineResult

# All tests in this module require real API access
pytestmark = pytest.mark.integration

FIXTURES = Path("tests/fixtures/sample_documents")


@pytest.fixture
def settings():
    """Load settings from config file."""
    return load_settings("config/settings.yaml")


@pytest.fixture
def complex_pdf_path():
    path = FIXTURES / "complex_technical_doc.pdf"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return str(path)


@pytest.fixture
def simple_pdf_path():
    path = FIXTURES / "simple.pdf"
    if not path.exists():
        pytest.skip(f"Test fixture not found: {path}")
    return str(path)


class TestIngestionPipeline:
    """Full pipeline integration tests (requires Azure APIs + real PDFs)."""

    def test_pipeline_with_complex_technical_doc(
        self, settings, complex_pdf_path,
    ) -> None:
        """Run full pipeline on complex_technical_doc.pdf.

        Validates:
        - SHA256 integrity check
        - PDF loading with image extraction
        - Document chunking
        - All three transforms (refine, enrich, caption)
        - Dense + sparse encoding
        - Vector upsert + BM25 index + image registration
        """
        pipeline = IngestionPipeline(
            settings=settings, collection="test_complex", force=True,
        )
        try:
            result = pipeline.run(complex_pdf_path)

            assert result.success, f"Pipeline failed: {result.error}"
            assert result.doc_id is not None
            assert result.chunk_count > 0
            assert len(result.vector_ids) == result.chunk_count

            # All 6 stages should be recorded
            for stage in ("integrity", "load", "split", "transform", "embed", "upsert"):
                assert stage in result.stages
                assert result.stages[stage]["elapsed_ms"] >= 0

            # Integrity should NOT be skipped (force=True)
            assert result.stages["integrity"]["skipped"] is False

            # Storage should match chunk count
            assert result.stages["upsert"]["vector_count"] == result.chunk_count
            assert result.stages["upsert"]["bm25_terms"] == result.chunk_count

        finally:
            pipeline.close()

    def test_pipeline_skip_already_processed(
        self, settings, simple_pdf_path,
    ) -> None:
        """Second run (no force) should skip an already-processed file."""
        collection = "test_skip"

        # First run — force process
        p1 = IngestionPipeline(settings, collection=collection, force=True)
        try:
            r1 = p1.run(simple_pdf_path)
            assert r1.success
            assert r1.chunk_count > 0
        finally:
            p1.close()

        # Second run — should skip
        p2 = IngestionPipeline(settings, collection=collection, force=False)
        try:
            r2 = p2.run(simple_pdf_path)
            assert r2.success
            assert r2.stages["integrity"]["skipped"] is True
            assert r2.chunk_count == 0  # nothing processed
        finally:
            p2.close()

    def test_pipeline_force_reprocess(
        self, settings, simple_pdf_path,
    ) -> None:
        """force=True should reprocess even if already ingested."""
        collection = "test_force"

        p1 = IngestionPipeline(settings, collection=collection, force=True)
        try:
            r1 = p1.run(simple_pdf_path)
            assert r1.success
            count1 = r1.chunk_count
        finally:
            p1.close()

        p2 = IngestionPipeline(settings, collection=collection, force=True)
        try:
            r2 = p2.run(simple_pdf_path)
            assert r2.success
            assert r2.chunk_count == count1
            assert r2.stages["integrity"]["skipped"] is False
        finally:
            p2.close()


class TestPipelineComponents:
    """Component smoke tests (require real API keys)."""

    def test_settings_loads_correctly(self, settings) -> None:
        assert settings.llm is not None
        assert settings.embedding is not None
        assert settings.ingestion is not None

    def test_embedding_creates_vectors(self, settings) -> None:
        from src.libs.embedding import (
            AzureEmbedding,
            EmbeddingFactory,
            OllamaEmbedding,
            OpenAIEmbedding,
        )

        factory = EmbeddingFactory()
        factory.register_provider("azure", AzureEmbedding)
        factory.register_provider("openai", OpenAIEmbedding)
        factory.register_provider("ollama", OllamaEmbedding)
        embedding = factory.create_from_settings(settings.embedding)

        vectors = embedding.embed(["Hello world", "Testing"])
        assert len(vectors) == 2
        assert len(vectors[0]) > 0

    def test_llm_responds(self, settings) -> None:
        from src.libs.llm import AzureLLM, LLMFactory, Message, OllamaLLM, OpenAILLM

        factory = LLMFactory()
        factory.register_provider("azure", AzureLLM)
        factory.register_provider("openai", OpenAILLM)
        factory.register_provider("ollama", OllamaLLM)
        llm = factory.create_from_settings(settings.llm)

        response = llm.chat([Message(role="user", content="Say hello.")])
        assert response is not None
        assert len(response.content) > 0
