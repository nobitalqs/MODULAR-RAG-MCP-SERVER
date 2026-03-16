"""Retrieval text generation for code chunks.

Generates natural language summaries for source_code chunks to bridge
the semantic gap between code and natural language queries in dense
retrieval. Non-code chunks pass through unchanged.

Processing per chunk:
    1. Check doc_type against target_doc_types
    2. If match: LLM summary → metadata["retrieval_text"]
    3. On LLM failure: rule-based fallback (brief + signatures + tags)
    4. If no match: pass through unchanged

Design Principles:
    - Graceful Degradation: LLM errors fall back to rule-based summary
    - Atomic Processing: each chunk processed independently
    - Observable: records ``retrieval_text_by`` in metadata + TraceContext
    - Immutable: returns new Chunk objects, never mutates input
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.core.settings import Settings, resolve_path
from src.core.trace.trace_context import TraceContext
from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_llm import BaseLLM, Message

logger = logging.getLogger(__name__)

# Regex for extracting function/class signatures
_SIGNATURE_RE = re.compile(r"^(?:def|class|async\s+def)\s+(\w+)", re.MULTILINE)


class RetrievalTextGenerator(BaseTransform):
    """Generate natural language retrieval text for code chunks.

    Args:
        settings: Application settings (reads ``ingestion.retrieval_text_generator``).
        llm: Pre-built LLM instance (shared with other transforms).
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
            resolve_path("config/prompts/retrieval_text_generation.txt")
        )

        cfg = self._read_config(settings)
        self.enabled: bool = cfg.get("enabled", False)
        self.target_doc_types: list[str] = cfg.get("target_doc_types", ["source_code"])
        self.max_chunk_length: int = cfg.get("max_chunk_length", 2000)

    # ── public interface ──────────────────────────────────────────

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Generate retrieval text for target doc_type chunks.

        Non-target chunks pass through unchanged. LLM failures
        fall back to rule-based summary.
        """
        if not chunks or not self.enabled:
            return chunks

        results: list[Chunk] = []
        llm_count = 0
        rule_count = 0
        skip_count = 0

        for chunk in chunks:
            doc_type = chunk.metadata.get("doc_type", "")
            if doc_type not in self.target_doc_types:
                results.append(chunk)
                skip_count += 1
                continue

            retrieval_text = self._generate_llm_summary(chunk.text, trace)
            if retrieval_text:
                results.append(self._new_chunk(chunk, retrieval_text, "llm"))
                llm_count += 1
            else:
                fallback = self._rule_based_summary(chunk)
                results.append(self._new_chunk(chunk, fallback, "rule"))
                rule_count += 1

        if trace is not None:
            stage_data = {
                "total_chunks": len(chunks),
                "llm_generated": llm_count,
                "rule_fallback": rule_count,
                "skipped": skip_count,
                "enabled": self.enabled,
            }
            trace.record_stage("retrieval_text_generator", stage_data)

        return results

    # ── LLM summary ───────────────────────────────────────────────

    def _generate_llm_summary(
        self,
        text: str,
        trace: TraceContext | None = None,
    ) -> str | None:
        """Generate summary via LLM. Returns None on failure."""
        if self._llm is None:
            return None

        try:
            prompt_template = self._load_prompt()
            if prompt_template is None:
                return None

            truncated = text[: self.max_chunk_length]
            prompt = prompt_template.replace("{text}", truncated)
            messages = [Message(role="user", content=prompt)]
            response = self._llm.chat(messages, trace=trace)

            content = response.content if hasattr(response, "content") else str(response)
            if content and content.strip():
                return content.strip()

            logger.warning("LLM returned empty retrieval text")
            return None

        except Exception:
            logger.warning(
                "LLM retrieval text generation failed, using rule-based fallback",
                exc_info=True,
            )
            return None

    # ── rule-based fallback ───────────────────────────────────────

    def _rule_based_summary(self, chunk: Chunk) -> str:
        """Build a natural language summary from metadata and signatures."""
        parts: list[str] = []

        brief = chunk.metadata.get("brief", "")
        if brief:
            parts.append(brief)

        title = chunk.metadata.get("title", "")
        if title and title != brief:
            parts.append(title)

        # Extract function/class signatures
        signatures = _SIGNATURE_RE.findall(chunk.text)
        if signatures:
            parts.append(f"Defines: {', '.join(signatures)}")

        tags = chunk.metadata.get("tags", [])
        if tags:
            tag_str = ", ".join(tags[:5]) if isinstance(tags, list) else str(tags)
            parts.append(f"Topics: {tag_str}")

        return ". ".join(parts) if parts else chunk.text[:200]

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _new_chunk(original: Chunk, retrieval_text: str, generated_by: str) -> Chunk:
        """Create new Chunk with retrieval_text in metadata."""
        return Chunk(
            id=original.id,
            text=original.text,
            metadata={
                **(original.metadata or {}),
                "retrieval_text": retrieval_text,
                "retrieval_text_by": generated_by,
            },
            start_offset=original.start_offset,
            end_offset=original.end_offset,
            source_ref=original.source_ref,
        )

    def _load_prompt(self) -> str | None:
        """Load and cache prompt template from disk."""
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
            logger.error("Failed to load prompt: %s", self._prompt_path, exc_info=True)
            return None

    @staticmethod
    def _read_config(settings: Settings) -> dict[str, Any]:
        """Safely read ``ingestion.retrieval_text_generator`` from settings."""
        if not hasattr(settings, "ingestion") or settings.ingestion is None:
            return {}
        cfg = settings.ingestion
        if hasattr(cfg, "retrieval_text_generator") and cfg.retrieval_text_generator:
            val = cfg.retrieval_text_generator
            return val if isinstance(val, dict) else {}
        if isinstance(cfg, dict):
            return cfg.get("retrieval_text_generator", {})
        return {}
