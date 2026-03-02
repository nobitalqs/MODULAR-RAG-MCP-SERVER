"""Metadata enrichment transform: rule-based + optional LLM enhancement.

Enriches each chunk's metadata with:
    - **title**: extracted from heading, first line, or first sentence
    - **summary**: first few sentences of content
    - **tags**: capitalized words, code identifiers, markdown emphasis

Processing pipeline per chunk:
    1. Rule-based enrichment (always runs)
    2. (Optional) LLM enrichment via prompt template
    3. On LLM failure → graceful fallback to rule-based result

Design Principles:
    - Graceful Degradation: LLM errors never block ingestion
    - Atomic Processing: each chunk independent
    - Observable: ``enriched_by`` in metadata + TraceContext stage
    - Immutable: text is never modified, only metadata added
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.core.settings import Settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class MetadataEnricher(BaseTransform):
    """Enriches chunk metadata with title, summary, and tags.

    Args:
        settings: Application settings (reads ``ingestion.metadata_enricher``).
        llm: Pre-built LLM instance (for testing / explicit injection).
        prompt_path: Override path for the prompt template file.
    """

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_path: str | None = None,
    ) -> None:
        self.settings = settings
        self._llm = llm
        self._prompt_template: str | None = None
        self._prompt_path = prompt_path or str(
            resolve_path("config/prompts/metadata_enrichment.txt")
        )

        # Read use_llm from settings
        enricher_cfg = self._read_config(settings)
        self.use_llm: bool = enricher_cfg.get("use_llm", False)

    # ── public interface ──────────────────────────────────────────────

    @property
    def llm(self) -> BaseLLM | None:
        """Lazy-load LLM via factory on first access."""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialised for metadata enrichment")
            except Exception:
                logger.warning("Failed to initialise LLM; falling back to rule-based", exc_info=True)
                self.use_llm = False
        return self._llm

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Enrich metadata for every chunk.  Text is never modified."""
        if not chunks:
            return []

        enriched: list[Chunk] = []
        success_count = 0
        llm_enhanced_count = 0

        for chunk in chunks:
            result_chunk, enriched_by = self._enrich_single(chunk, trace)
            enriched.append(result_chunk)
            if enriched_by != "error":
                success_count += 1
            if enriched_by == "llm":
                llm_enhanced_count += 1

        if trace is not None:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "use_llm": self.use_llm,
            })

        return enriched

    # ── single-chunk orchestration ────────────────────────────────────

    def _enrich_single(
        self,
        chunk: Chunk,
        trace: TraceContext | None,
    ) -> tuple[Chunk, str]:
        """Enrich one chunk.  Returns ``(new_chunk, enriched_by)``."""
        try:
            rule_meta = self._rule_based_enrich(chunk.text)

            if self.use_llm and self.llm is not None:
                llm_meta = self._llm_enrich(chunk.text, trace)
                if llm_meta is not None:
                    return self._new_chunk(chunk, llm_meta, "llm"), "llm"
                # LLM failed → fallback
                rule_meta["enrich_fallback_reason"] = "llm_failed"

            return self._new_chunk(chunk, rule_meta, "rule"), "rule"

        except Exception as exc:
            logger.error("Failed to enrich chunk %s: %s", chunk.id, exc)
            error_meta = self._error_metadata(chunk, exc)
            return self._new_chunk(chunk, error_meta, "error", safe_text=True), "error"

    # ── new-chunk factory ─────────────────────────────────────────────

    @staticmethod
    def _new_chunk(
        original: Chunk,
        extra_meta: dict[str, Any],
        enriched_by: str,
        *,
        safe_text: bool = False,
    ) -> Chunk:
        """Create a new Chunk with merged metadata."""
        return Chunk(
            id=original.id,
            text=original.text if not safe_text else (original.text or ""),
            metadata={**(original.metadata or {}), **extra_meta, "enriched_by": enriched_by},
            start_offset=original.start_offset,
            end_offset=original.end_offset,
            source_ref=original.source_ref,
        )

    @staticmethod
    def _error_metadata(chunk: Chunk, exc: Exception) -> dict[str, Any]:
        """Minimal metadata for a chunk that failed enrichment."""
        preview = ""
        if chunk.text:
            preview = (chunk.text[:100] + "...") if len(chunk.text) > 100 else chunk.text
        return {
            "title": "Untitled",
            "summary": preview,
            "tags": [],
            "enrich_error": str(exc),
        }

    # ── rule-based enrichment ─────────────────────────────────────────

    def _rule_based_enrich(self, text: str) -> dict[str, Any]:
        """Extract metadata via heuristics.

        Raises:
            TypeError: If *text* is ``None``.
        """
        if text is None:
            raise TypeError("Chunk text cannot be None")

        return {
            "title": self._extract_title(text),
            "summary": self._extract_summary(text),
            "tags": self._extract_tags(text),
        }

    # ── title extraction ──────────────────────────────────────────────

    @staticmethod
    def _extract_title(text: str) -> str:
        """Extract title with priority: heading → short first line → first sentence."""
        if not text:
            return "Untitled"

        # 1. Markdown heading
        heading = re.match(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
        if heading:
            return heading.group(1).strip()

        # 2. Short first line that doesn't end with sentence punctuation
        first_line = text.split("\n")[0].strip()
        if first_line and len(first_line) <= 100 and not first_line.endswith((".", ",", ";")):
            return first_line

        # 3. First sentence (strip trailing punctuation)
        sentences = re.split(r"[.!?]\s+", text)
        if sentences and sentences[0]:
            title = re.sub(r"[.!?]+$", "", sentences[0].strip())
            return title[:147] + "..." if len(title) > 150 else title

        # 4. Fallback
        return text[:100].strip() + ("..." if len(text) > 100 else "")

    # ── summary extraction ────────────────────────────────────────────

    @staticmethod
    def _extract_summary(text: str, max_sentences: int = 3) -> str:
        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)
        summary = " ".join(sentences[:max_sentences]).strip()
        return summary[:497] + "..." if len(summary) > 500 else summary

    # ── tag extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_tags(text: str, max_tags: int = 10) -> list[str]:
        if not text:
            return []

        tags: set[str] = set()

        # Capitalized words / multi-word proper nouns
        tags.update(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)[:5])

        # camelCase / snake_case identifiers
        tags.update(re.findall(r"\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b", text)[:5])

        # Markdown bold/italic emphasis
        for groups in re.findall(r"\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_", text)[:5]:
            for g in groups:
                if g:
                    tags.add(g.strip())

        return sorted(tags)[:max_tags]

    # ── LLM enrichment ────────────────────────────────────────────────

    def _llm_enrich(
        self,
        text: str,
        trace: TraceContext | None = None,
    ) -> dict[str, Any] | None:
        """Attempt LLM-based enrichment.  Returns parsed metadata or ``None``."""
        if self._llm is None:
            return None

        try:
            prompt_template = self._load_prompt()
            formatted = prompt_template.replace("{chunk_text}", text[:2000])

            messages = [Message(role="user", content=formatted)]
            response = self._llm.chat(messages)

            # Normalise to string
            response_text: str = (
                response.content if hasattr(response, "content") else str(response)
            )

            if not response_text or not response_text.strip():
                logger.warning("LLM returned empty response for metadata enrichment")
                if trace:
                    trace.record_stage("llm_enrich", {"success": False, "error": "empty_response"})
                return None

            metadata = self._parse_llm_response(response_text)

            if trace:
                trace.record_stage("llm_enrich", {
                    "success": True,
                    "response_length": len(response_text),
                })

            return metadata

        except Exception as exc:
            logger.warning("LLM enrichment failed: %s", exc)
            if trace:
                trace.record_stage("llm_enrich", {"success": False, "error": str(exc)})
            return None

    # ── prompt loading ────────────────────────────────────────────────

    def _load_prompt(self) -> str:
        """Load and cache prompt template.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
        """
        if self._prompt_template is not None:
            return self._prompt_template

        path = Path(self._prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self._prompt_path}")

        self._prompt_template = path.read_text(encoding="utf-8")
        return self._prompt_template

    # ── LLM response parsing ─────────────────────────────────────────

    @staticmethod
    def _parse_llm_response(response: str) -> dict[str, Any]:
        """Parse ``Title: / Summary: / Tags:`` formatted LLM output."""
        metadata: dict[str, Any] = {"title": "", "summary": "", "tags": []}

        title_m = re.search(r"Title:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if title_m:
            metadata["title"] = title_m.group(1).strip()

        summary_m = re.search(
            r"Summary:\s*(.+?)(?:\nTags:|$)", response, re.IGNORECASE | re.DOTALL
        )
        if summary_m:
            metadata["summary"] = summary_m.group(1).strip()

        tags_m = re.search(r"Tags:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if tags_m:
            metadata["tags"] = [t.strip() for t in tags_m.group(1).split(",") if t.strip()]

        # Fallbacks
        if not metadata["title"]:
            metadata["title"] = "Untitled"
        if not metadata["summary"]:
            metadata["summary"] = response[:500]

        return metadata

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _read_config(settings: Settings) -> dict[str, Any]:
        """Safely read ``ingestion.metadata_enricher`` from settings."""
        if not hasattr(settings, "ingestion") or settings.ingestion is None:
            return {}
        cfg = settings.ingestion
        if hasattr(cfg, "metadata_enricher") and cfg.metadata_enricher:
            val = cfg.metadata_enricher
            return val if isinstance(val, dict) else {}
        if isinstance(cfg, dict):
            return cfg.get("metadata_enricher", {})
        return {}
