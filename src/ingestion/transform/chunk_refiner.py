"""Chunk refinement transform: rule-based cleaning + optional LLM enhancement.

Processing pipeline per chunk:
    1. Rule-based refine: remove noise (whitespace, headers/footers, HTML)
    2. (Optional) LLM refine: intelligent content improvement via prompt template
    3. On LLM failure: gracefully fall back to rule-based result

Design Principles:
    - Graceful Degradation: LLM errors don't block ingestion
    - Atomic Processing: each chunk processed independently
    - Observable: records ``refined_by`` in metadata + TraceContext stage
    - Immutable: returns new Chunk objects, never mutates input
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

# ── Compiled regex patterns ──────────────────────────────────────────

# Code block fences (```...```)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")

# Page headers/footers: lines containing separator chars + keywords
_PAGE_HEADER_FOOTER_RE = re.compile(
    r"^[─═\-\s]*(?:Page\s+\d+|Footer|©|Confidential)[^\n]*[─═\-\s]*$",
    re.IGNORECASE | re.MULTILINE,
)

# Separator-only lines (3+ box-drawing or rule chars)
_SEPARATOR_LINE_RE = re.compile(r"^[─═]{3,}\s*$", re.MULTILINE)

# HTML comments
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# HTML tags (preserve content, strip tags)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Runs of 2+ spaces
_MULTI_SPACE_RE = re.compile(r" {2,}")

# Runs of 3+ newlines
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


class ChunkRefiner(BaseTransform):
    """Refines chunks through rule-based cleaning and optional LLM enhancement.

    Args:
        settings: Application settings (reads ``ingestion.chunk_refiner``).
        llm: Pre-built LLM instance (for testing or explicit injection).
        prompt_path: Path to the prompt template file.
    """

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_path: str | None = None,
    ) -> None:
        self.settings = settings
        self._llm: BaseLLM | None = llm
        self._prompt_template: str | None = None
        self._prompt_path: str = prompt_path or str(
            resolve_path("config/prompts/chunk_refinement.txt")
        )

        # Read use_llm from settings
        enricher_cfg = self._read_config(settings)
        self.use_llm: bool = enricher_cfg.get("use_llm", False)

    # ── lazy LLM ─────────────────────────────────────────────────────

    @property
    def llm(self) -> BaseLLM | None:
        """Lazy-load LLM via factory on first access."""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create_llm(self.settings)
                logger.info("LLM initialised for chunk refinement")
            except Exception:
                logger.warning(
                    "Failed to initialise LLM; falling back to rule-based",
                    exc_info=True,
                )
                self.use_llm = False
        return self._llm

    # ── public interface ──────────────────────────────────────────────

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Transform chunks through the refinement pipeline.

        Returns a new list of the same length.  Failures on individual
        chunks are caught so that one bad chunk never blocks the rest.
        """
        if not chunks:
            return []

        refined: list[Chunk] = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0

        for chunk in chunks:
            try:
                doc_type = chunk.metadata.get("doc_type", "pdf")
                rule_text = self._rule_based_refine(chunk.text, doc_type=doc_type)
                rule_text = self._inject_source_context(rule_text, chunk.metadata)

                if self.use_llm and self.llm:
                    llm_text = self._llm_refine(rule_text, trace)
                    if llm_text:
                        refined.append(self._new_chunk(chunk, llm_text, "llm"))
                        llm_enhanced_count += 1
                    else:
                        refined.append(self._new_chunk(chunk, rule_text, "rule"))
                        fallback_count += 1
                else:
                    refined.append(self._new_chunk(chunk, rule_text, "rule"))

                success_count += 1

            except Exception:
                logger.exception(
                    "Failed to refine chunk %s, preserving original", chunk.id
                )
                refined.append(chunk)

        if trace is not None:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
            })

        return refined

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _new_chunk(original: Chunk, text: str, refined_by: str) -> Chunk:
        """Create a new Chunk with updated text and ``refined_by`` metadata."""
        return Chunk(
            id=original.id,
            text=text,
            metadata={**(original.metadata or {}), "refined_by": refined_by},
            start_offset=original.start_offset,
            end_offset=original.end_offset,
            source_ref=original.source_ref,
        )

    # ── source context injection ─────────────────────────────────────

    @staticmethod
    def _inject_source_context(text: str, metadata: dict[str, Any]) -> str:
        """Prepend source filename to chunk text for retrieval discoverability.

        Ensures filename keywords (e.g. "DimuonAnalysis") appear in both
        BM25 and dense search indices, not just in unsearchable metadata.

        Skips injection if:
        - No filename can be determined
        - Filename already appears in the text (avoids duplication)
        """
        filename = metadata.get("filename") or Path(
            metadata.get("source_path", "")
        ).name
        if not filename:
            return text
        if filename in text:
            return text
        return f"[Source: {filename}]\n{text}"

    # ── rule-based cleaning ───────────────────────────────────────────

    def _rule_based_refine(self, text: str, doc_type: str = "pdf") -> str:
        """Apply deterministic regex-based cleaning.

        Args:
            text: Raw chunk text.
            doc_type: Document type (e.g. "pdf", "markdown", "source_code").
                PDF-only rules (header/footer removal, separator removal) are
                skipped for non-PDF types to preserve Markdown/code formatting.

        Order of operations matters:
            1. Early return for None / empty
            2. Extract code blocks (protect internal formatting)
            3. (PDF only) Remove page header / footer lines
            4. (PDF only) Remove separator-only lines
            5. Remove HTML comments
            6. Remove HTML tags (preserve inner content)
            7. Collapse multi-space runs
            8. Collapse 3+ consecutive newlines → 2
            9. Strip trailing whitespace per line
           10. Restore code blocks
        """
        if not text:
            return text  # None → None, "" → ""

        if not text.strip():
            return ""

        # 1. Extract code blocks
        code_blocks: list[str] = []

        def _extract(match: re.Match[str]) -> str:
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        text = _CODE_BLOCK_RE.sub(_extract, text)

        # 2-3. PDF-only: Remove page header/footer lines and separator lines
        if doc_type == "pdf":
            text = _PAGE_HEADER_FOOTER_RE.sub("", text)
            text = _SEPARATOR_LINE_RE.sub("", text)

        # 4. Remove HTML comments
        text = _HTML_COMMENT_RE.sub("", text)

        # 5. Remove HTML tags (preserve content)
        text = _HTML_TAG_RE.sub("", text)

        # 6. Collapse 2+ spaces → single space
        text = _MULTI_SPACE_RE.sub(" ", text)

        # 7. Collapse 3+ newlines → 2
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)

        # 8. Strip trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        # 9. Restore code blocks
        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)

        return text.strip()

    # ── LLM-based enhancement ─────────────────────────────────────────

    def _llm_refine(
        self,
        text: str,
        trace: Any = None,
    ) -> str | None:
        """Attempt LLM-based refinement.

        Returns the refined text on success, or ``None`` on any failure
        (empty response, API error, missing prompt, missing placeholder).
        """
        if self._llm is None:
            return None

        try:
            prompt_template = self._load_prompt()
            if prompt_template is None:
                return None

            if "{text}" not in prompt_template:
                logger.error("Prompt template missing {text} placeholder")
                return None

            prompt = prompt_template.replace("{text}", text)
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace)

            content = response.content if not isinstance(response, str) else response
            if content and content.strip():
                return content.strip()

            logger.warning("LLM returned empty result")
            return None
        except Exception:
            logger.warning(
                "LLM refinement failed, will fall back to rule-based",
                exc_info=True,
            )
            return None

    def _load_prompt(self) -> str | None:
        """Load and cache the prompt template from disk.

        Returns ``None`` if the file is missing or unreadable.
        """
        if self._prompt_template is not None:
            return self._prompt_template

        path = Path(self._prompt_path)
        if not path.exists():
            logger.warning("Prompt file not found: %s", self._prompt_path)
            return None

        try:
            self._prompt_template = path.read_text(encoding="utf-8")
            return self._prompt_template
        except Exception:
            logger.error(
                "Failed to load prompt template from %s",
                self._prompt_path,
                exc_info=True,
            )
            return None

    # ── config reader ─────────────────────────────────────────────────

    @staticmethod
    def _read_config(settings: Settings) -> dict[str, Any]:
        """Safely read ``ingestion.chunk_refiner`` from settings."""
        if not hasattr(settings, "ingestion") or settings.ingestion is None:
            return {}
        cfg = settings.ingestion
        if hasattr(cfg, "chunk_refiner") and cfg.chunk_refiner:
            val = cfg.chunk_refiner
            return val if isinstance(val, dict) else {}
        if isinstance(cfg, dict):
            return cfg.get("chunk_refiner", {})
        return {}
