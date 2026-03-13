"""Ingestion pipeline orchestrator.

Wires together all ingestion stages: integrity check -> load -> split
-> transform -> encode -> store.  Executes sequentially with clear
error reporting and optional progress callbacks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.settings import Settings, load_settings
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk, Document

# Stage 3: Chunking
from src.ingestion.chunking.document_chunker import DocumentChunker

# Document lifecycle
from src.ingestion.document_manager import DocumentManager

# Stage 5: Encoding
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder

# Stage 6: Storage
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import VectorUpserter

# Stage 4: Transform
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.transform.metadata_enricher import MetadataEnricher

# Factories
from src.libs.embedding import (
    AzureEmbedding,
    EmbeddingFactory,
    OllamaEmbedding,
    OpenAIEmbedding,
)
from src.libs.llm import (
    AzureLLM,
    AzureVisionLLM,
    DeepSeekLLM,
    LLMFactory,
    OllamaLLM,
    OpenAILLM,
)

# Stage 1: Integrity
from src.libs.loader.file_integrity import SQLiteIntegrityChecker

# Stage 2: Loading (factory-based multi-format)
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.source_code_loader import SourceCodeLoader
from src.libs.splitter import RecursiveSplitter, SplitterFactory
from src.libs.vector_store import ChromaStore, VectorStoreFactory

logger = logging.getLogger(__name__)

# Stage names in execution order
STAGE_NAMES = ("integrity", "load", "split", "transform", "embed", "upsert")
TOTAL_STAGES = len(STAGE_NAMES)


# ── Result ──────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Immutable result of a single pipeline run."""

    success: bool
    file_path: str
    doc_id: str | None = None
    chunk_count: int = 0
    image_count: int = 0
    vector_ids: list[str] = field(default_factory=list)
    error: str | None = None
    old_version_cleaned: bool = False
    old_chunks_deleted: int = 0
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "success": self.success,
            "file_path": self.file_path,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "vector_ids": self.vector_ids,
            "error": self.error,
            "old_version_cleaned": self.old_version_cleaned,
            "old_chunks_deleted": self.old_chunks_deleted,
            "stages": self.stages,
        }


# ── Pipeline ────────────────────────────────────────────────────────


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline.

    Executes 6 stages sequentially:
        1. Integrity check  (SHA256 skip / force)
        2. Document loading  (PDF -> Document)
        3. Chunking          (Document -> List[Chunk])
        4. Transform         (refine -> enrich -> caption)
        5. Encoding          (dense + sparse vectors)
        6. Storage           (vector upsert + BM25 index + images)
    """

    def __init__(
        self,
        settings: Settings,
        collection: str = "default",
        force: bool = False,
    ) -> None:
        self.settings = settings
        self.collection = collection
        self.force = force

        # Stage 1: Integrity checker
        self.integrity_checker = SQLiteIntegrityChecker(
            db_path="data/db/file_integrity.db",
        )

        # Stage 2: Loader factory (multi-format)
        self._table_extraction = None
        self._formula_extraction = None
        if settings.ingestion:
            self._table_extraction = settings.ingestion.table_extraction
            self._formula_extraction = settings.ingestion.formula_extraction
        self.loader_factory = LoaderFactory()
        self.loader_factory.register_provider(".pdf", PdfLoader)
        self.loader_factory.register_provider(".md", MarkdownLoader)
        self.loader_factory.register_provider(".markdown", MarkdownLoader)
        self.loader_factory.register_provider(".c", SourceCodeLoader)
        self.loader_factory.register_provider(".cpp", SourceCodeLoader)
        self.loader_factory.register_provider(".cxx", SourceCodeLoader)
        self.loader_factory.register_provider(".cc", SourceCodeLoader)
        self.loader_factory.register_provider(".h", SourceCodeLoader)
        self.loader_factory.register_provider(".hxx", SourceCodeLoader)
        self.loader_factory.register_provider(".py", SourceCodeLoader)

        # Stage 3: Chunking — use SplitterFactory to create from settings
        splitter_factory = SplitterFactory()
        splitter_factory.register_provider("recursive", RecursiveSplitter)
        splitter = splitter_factory.create_from_settings(settings.ingestion)
        self.chunker = DocumentChunker(splitter)

        # Stage 4: Transforms — create LLM/VisionLLM (graceful on failure)
        llm = self._create_llm(settings)
        vision_llm = self._create_vision_llm(settings)

        self.chunk_refiner = ChunkRefiner(settings, llm=llm)
        self.metadata_enricher = MetadataEnricher(settings, llm=llm)
        self.image_captioner = ImageCaptioner(settings, vision_llm=vision_llm)

        # Stage 5: Encoding — create embedding from factory
        embedding = self._create_embedding(settings)
        batch_size = settings.ingestion.batch_size if settings.ingestion else 100
        dense_encoder = DenseEncoder(embedding, batch_size=batch_size)
        sparse_encoder = SparseEncoder()
        self.batch_processor = BatchProcessor(
            dense_encoder,
            sparse_encoder,
            batch_size=batch_size,
        )

        # Stage 6: Storage — create vector store from factory
        vector_store = self._create_vector_store(settings)
        self.vector_upserter = VectorUpserter(vector_store)
        self.bm25_indexer = BM25Indexer(index_dir="data/db/bm25")
        self.image_storage = ImageStorage(
            db_path="data/db/image_index.db",
            images_root="data/images",
        )

        # Document lifecycle manager (for old version cleanup)
        self.document_manager = DocumentManager(
            chroma_store=vector_store,
            bm25_indexer=self.bm25_indexer,
            image_storage=self.image_storage,
            file_integrity=self.integrity_checker,
        )

        logger.info(
            "IngestionPipeline initialized: collection=%s, force=%s",
            collection,
            force,
        )

    # ── factory helpers ─────────────────────────────────────────────

    @staticmethod
    def _create_llm(settings: Settings):
        """Create text LLM from settings.  Returns None on failure."""
        try:
            factory = LLMFactory()
            factory.register_provider("openai", OpenAILLM)
            factory.register_provider("azure", AzureLLM)
            factory.register_provider("deepseek", DeepSeekLLM)
            factory.register_provider("ollama", OllamaLLM)
            return factory.create_from_settings(settings.llm)
        except Exception:
            logger.warning(
                "LLM creation failed; transforms will use rule-only mode",
            )
            return None

    @staticmethod
    def _create_vision_llm(settings: Settings):
        """Create vision LLM from settings.  Returns None on failure."""
        try:
            from src.libs.llm.openai_vision_llm import OpenAIVisionLLM

            factory = LLMFactory()
            factory.register_vision_provider("azure", AzureVisionLLM)
            factory.register_vision_provider("openai", OpenAIVisionLLM)
            return factory.create_vision_llm_from_settings(settings.vision_llm)
        except Exception:
            logger.warning(
                "Vision LLM creation failed; image captioning disabled",
            )
            return None

    @staticmethod
    def _create_embedding(settings: Settings):
        """Create embedding model from settings."""
        factory = EmbeddingFactory()
        factory.register_provider("openai", OpenAIEmbedding)
        factory.register_provider("azure", AzureEmbedding)
        factory.register_provider("ollama", OllamaEmbedding)
        return factory.create_from_settings(settings.embedding)

    @staticmethod
    def _create_vector_store(settings: Settings):
        """Create vector store from settings."""
        factory = VectorStoreFactory()
        factory.register_provider("chroma", ChromaStore)
        return factory.create_from_settings(settings.vector_store)

    # ── public API ──────────────────────────────────────────────────

    def run(
        self,
        file_path: str | Path,
        trace: TraceContext | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> PipelineResult:
        """Execute the full ingestion pipeline for a single file.

        Args:
            file_path: Path to the document to ingest.
            trace: Optional TraceContext for observability.
            on_progress: Callback ``(stage_name, current, total)``.

        Returns:
            PipelineResult with success/failure details.
        """
        file_path = str(Path(file_path).resolve())
        stages: dict[str, dict[str, Any]] = {}

        def _progress(stage_name: str, step: int) -> None:
            if on_progress is not None:
                on_progress(stage_name, step, TOTAL_STAGES)

        try:
            # ── Stage 1: Integrity ──────────────────────────────────
            _progress("integrity", 1)
            t0 = time.monotonic()
            file_hash = self.integrity_checker.compute_sha256(file_path)

            if not self.force and self.integrity_checker.should_skip(file_hash):
                elapsed = (time.monotonic() - t0) * 1000
                stages["integrity"] = {
                    "skipped": True,
                    "file_hash": file_hash,
                    "elapsed_ms": elapsed,
                }
                if trace:
                    trace.record_stage("integrity", stages["integrity"], elapsed)
                logger.info("Skipping already-processed file: %s", file_path)
                return PipelineResult(
                    success=True,
                    file_path=file_path,
                    stages=stages,
                )

            elapsed = (time.monotonic() - t0) * 1000
            stages["integrity"] = {
                "skipped": False,
                "file_hash": file_hash,
                "elapsed_ms": elapsed,
            }
            if trace:
                trace.record_stage("integrity", stages["integrity"], elapsed)

            # ── Old version cleanup ───────────────────────────────
            old_version_cleaned = False
            old_chunks_deleted = 0
            old_hash = self.integrity_checker.lookup_by_path(
                file_path,
                self.collection,
            )
            if old_hash is not None and old_hash != file_hash:
                logger.info(
                    "Detected modified document, cleaning old version: %s (old=%s, new=%s)",
                    file_path,
                    old_hash[:12],
                    file_hash[:12],
                )
                try:
                    del_result = self.document_manager.delete_document(
                        source_path=file_path,
                        collection=self.collection,
                        source_hash=old_hash,
                    )
                    old_version_cleaned = True
                    old_chunks_deleted = del_result.chunks_deleted
                    if del_result.errors:
                        logger.warning(
                            "Partial cleanup errors: %s",
                            del_result.errors,
                        )
                except Exception as e:
                    logger.warning("Failed to clean old version: %s", e)

            stages["integrity"]["old_version_cleaned"] = old_version_cleaned
            stages["integrity"]["old_chunks_deleted"] = old_chunks_deleted

            # ── Stage 2: Load ───────────────────────────────────────
            _progress("load", 2)
            t0 = time.monotonic()
            loader = self.loader_factory.create_for_file(
                file_path,
                extract_images=True,
                image_storage_dir="data/images",
                table_extraction=self._table_extraction,
                formula_extraction=self._formula_extraction,
            )
            document: Document = loader.load(file_path)
            elapsed = (time.monotonic() - t0) * 1000

            image_count = len(document.metadata.get("images", []))
            stages["load"] = {
                "doc_id": document.id,
                "page_count": document.metadata.get("page_count", 0),
                "image_count": image_count,
                "text_length": len(document.text),
                "elapsed_ms": elapsed,
            }
            if trace:
                trace.record_stage("load", stages["load"], elapsed)
            logger.info(
                "Loaded document: %s (pages=%d, images=%d)",
                document.id,
                document.metadata.get("page_count", 0),
                image_count,
            )

            # ── Stage 3: Split ──────────────────────────────────────
            _progress("split", 3)
            t0 = time.monotonic()
            chunks: list[Chunk] = self.chunker.split_document(document)
            elapsed = (time.monotonic() - t0) * 1000

            stages["split"] = {
                "chunk_count": len(chunks),
                "elapsed_ms": elapsed,
            }
            if trace:
                trace.record_stage("split", stages["split"], elapsed)
            logger.info("Split into %d chunks", len(chunks))

            # ── Stage 4: Transform ──────────────────────────────────
            _progress("transform", 4)
            t0 = time.monotonic()
            chunks = self.chunk_refiner.transform(chunks, trace=trace)
            chunks = self.metadata_enricher.transform(chunks, trace=trace)
            chunks = self.image_captioner.transform(chunks, trace=trace)
            elapsed = (time.monotonic() - t0) * 1000

            stages["transform"] = {
                "chunk_count": len(chunks),
                "elapsed_ms": elapsed,
            }
            if trace:
                trace.record_stage("transform", stages["transform"], elapsed)
            logger.info("Transforms complete: %d chunks", len(chunks))

            # ── Stage 5: Encode ─────────────────────────────────────
            _progress("embed", 5)
            t0 = time.monotonic()
            batch_result = self.batch_processor.process(chunks, trace=trace)
            elapsed = (time.monotonic() - t0) * 1000

            stages["embed"] = {
                "dense_count": len(batch_result.dense_vectors),
                "sparse_count": len(batch_result.sparse_stats),
                "batch_count": batch_result.batch_count,
                "successful_chunks": batch_result.successful_chunks,
                "failed_chunks": batch_result.failed_chunks,
                "elapsed_ms": elapsed,
            }
            if trace:
                trace.record_stage("embed", stages["embed"], elapsed)
            logger.info(
                "Encoding complete: %d dense, %d sparse",
                len(batch_result.dense_vectors),
                len(batch_result.sparse_stats),
            )

            # ── Stage 6: Store ──────────────────────────────────────
            _progress("upsert", 6)
            t0 = time.monotonic()

            # 6a: Vector upsert
            vector_ids = self.vector_upserter.upsert(
                chunks,
                batch_result.dense_vectors,
                trace=trace,
            )

            # 6b: BM25 index — remap chunk_id to vector IDs for consistency
            term_stats = [
                {
                    "chunk_id": vid,
                    "term_frequencies": sparse["term_frequencies"],
                    "doc_length": sparse["doc_length"],
                }
                for vid, sparse in zip(vector_ids, batch_result.sparse_stats)
            ]
            self.bm25_indexer.build(
                term_stats,
                collection=self.collection,
                trace=trace,
            )

            # 6c: Image registration
            images_registered = 0
            for img_info in document.metadata.get("images", []):
                img_id = img_info.get("id", "")
                img_path = img_info.get("path", "")
                if img_id and img_path and Path(img_path).exists():
                    self.image_storage.register_image(
                        image_id=img_id,
                        file_path=img_path,
                        collection=self.collection,
                        doc_hash=file_hash,
                        page_num=img_info.get("page_num"),
                    )
                    images_registered += 1

            elapsed = (time.monotonic() - t0) * 1000
            stages["upsert"] = {
                "vector_count": len(vector_ids),
                "bm25_terms": len(term_stats),
                "images_registered": images_registered,
                "elapsed_ms": elapsed,
            }
            if trace:
                trace.record_stage("upsert", stages["upsert"], elapsed)
            logger.info(
                "Storage complete: %d vectors, %d BM25 terms, %d images",
                len(vector_ids),
                len(term_stats),
                images_registered,
            )

            # ── Mark success ────────────────────────────────────────
            self.integrity_checker.mark_success(
                file_hash,
                file_path,
                self.collection,
            )

            return PipelineResult(
                success=True,
                file_path=file_path,
                doc_id=document.id,
                chunk_count=len(chunks),
                image_count=images_registered,
                vector_ids=vector_ids,
                old_version_cleaned=old_version_cleaned,
                old_chunks_deleted=old_chunks_deleted,
                stages=stages,
            )

        except Exception as exc:
            logger.error(
                "Pipeline failed for %s: %s",
                file_path,
                exc,
                exc_info=True,
            )
            # Mark failure if we have a file hash
            integrity_data = stages.get("integrity") or {}
            if "file_hash" in integrity_data:
                try:
                    self.integrity_checker.mark_failed(
                        integrity_data["file_hash"],
                        file_path,
                        str(exc),
                    )
                except Exception:
                    logger.warning("Failed to record failure in integrity DB")

            return PipelineResult(
                success=False,
                file_path=file_path,
                error=str(exc),
                stages=stages,
            )

    def close(self) -> None:
        """Release resources held by pipeline components."""
        self.integrity_checker.close()
        self.image_storage.close()
        logger.info("Pipeline resources released")


# ── Convenience function ────────────────────────────────────────────


def run_pipeline(
    file_path: str | Path,
    collection: str = "default",
    force: bool = False,
    config_path: str | Path | None = None,
) -> PipelineResult:
    """Load settings, create pipeline, run, and close.

    Args:
        file_path: Document to process.
        collection: Target collection name.
        force: Reprocess even if already ingested.
        config_path: Path to settings.yaml.

    Returns:
        PipelineResult with ingestion outcome.
    """
    settings = load_settings(config_path)
    pipeline = IngestionPipeline(settings, collection=collection, force=force)
    try:
        trace = TraceContext(trace_type="ingestion")
        result = pipeline.run(file_path, trace=trace)
        trace.finish()
        return result
    finally:
        pipeline.close()
