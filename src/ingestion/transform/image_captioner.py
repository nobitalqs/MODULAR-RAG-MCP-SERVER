"""Image captioner transform: Vision LLM caption generation for image-rich chunks.

Scans chunks for ``[IMAGE: id]`` placeholders, resolves image paths from
chunk metadata, generates captions via a Vision LLM, and injects the
descriptions back into both the text and metadata.

Processing pipeline:
    1. Collect unique image IDs referenced across all chunks
    2. Caption each unique image (with caching to avoid duplicate API calls)
    3. Inject captions into text and metadata per chunk
    4. Mark ``has_unprocessed_images`` when disabled or on failure

Design Principles:
    - Graceful Degradation: disabled / failed Vision LLM → never blocks ingestion
    - Cache: same image_id captioned only once across all chunks
    - Observable: ``image_captions`` in metadata + TraceContext stage
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
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput

logger = logging.getLogger(__name__)

# Regex: [IMAGE: some_id]
_IMAGE_REF_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


class ImageCaptioner(BaseTransform):
    """Generate captions for ``[IMAGE: id]`` placeholders using Vision LLM.

    Args:
        settings: Application settings (reads ``vision_llm.enabled``).
        vision_llm: Pre-built Vision LLM instance (for testing / injection).
        prompt_path: Override path for the captioning prompt template.
    """

    def __init__(
        self,
        settings: Settings,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | None = None,
    ) -> None:
        self._settings = settings
        self._vision_llm = vision_llm
        self._prompt_path = prompt_path or str(
            resolve_path("config/prompts/image_captioning.txt")
        )
        self._prompt: str | None = None

        # Resolve enabled flag
        vlm_cfg = getattr(settings, "vision_llm", None)
        self._enabled: bool = (
            vlm_cfg is not None and getattr(vlm_cfg, "enabled", False)
        )

        # Caption cache: image_id → caption string
        self._caption_cache: dict[str, str] = {}

    # ── public interface ──────────────────────────────────────────────

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Process chunks, captioning any ``[IMAGE: id]`` references."""
        if not chunks:
            return []

        result: list[Chunk] = []
        captions_generated = 0

        # Clear cache per invocation (fresh document batch)
        self._caption_cache.clear()

        for chunk in chunks:
            image_ids = _IMAGE_REF_RE.findall(chunk.text)
            image_ids = [i.strip() for i in image_ids]

            if not image_ids:
                # No image references → pass through unchanged
                result.append(chunk)
                continue

            if not self._enabled or self._vision_llm is None:
                result.append(self._mark_unprocessed(chunk))
                continue

            new_chunk, n_captions = self._caption_chunk(chunk, image_ids, trace)
            result.append(new_chunk)
            captions_generated += n_captions

        if trace is not None:
            trace.record_stage("image_captioner", {
                "total_chunks": len(chunks),
                "enabled": self._enabled,
                "captions_generated": captions_generated,
            })

        return result

    # ── internal helpers ──────────────────────────────────────────────

    def _caption_chunk(
        self,
        chunk: Chunk,
        image_ids: list[str],
        trace: TraceContext | None,
    ) -> tuple[Chunk, int]:
        """Caption images in a single chunk.  Returns ``(new_chunk, n_captions)``."""
        image_lookup = self._build_image_lookup(chunk)
        new_text = chunk.text
        captions_list: list[dict[str, str]] = []

        for img_id in image_ids:
            caption = self._get_or_generate_caption(img_id, image_lookup, trace)
            if caption is None:
                continue

            captions_list.append({"id": img_id, "caption": caption})
            placeholder = f"[IMAGE: {img_id}]"
            replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
            new_text = new_text.replace(placeholder, replacement)

        if not captions_list:
            # All images failed → mark unprocessed
            return self._mark_unprocessed(chunk), 0

        meta = {
            **chunk.metadata,
            "image_captions": captions_list,
        }
        new_chunk = Chunk(
            id=chunk.id,
            text=new_text,
            metadata=meta,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )
        return new_chunk, len(captions_list)

    def _get_or_generate_caption(
        self,
        img_id: str,
        image_lookup: dict[str, str],
        trace: TraceContext | None,
    ) -> str | None:
        """Return cached caption or generate via Vision LLM."""
        if img_id in self._caption_cache:
            return self._caption_cache[img_id]

        img_path = image_lookup.get(img_id)
        if not img_path:
            logger.warning("No image path for id=%s, skipping", img_id)
            return None

        try:
            prompt = self._load_prompt()
            image_input = ImageInput(path=img_path)
            response = self._vision_llm.chat_with_image(
                text=prompt, image=image_input, trace=trace,
            )
            caption = response.content
            self._caption_cache[img_id] = caption
            return caption
        except Exception:
            logger.warning("Failed to caption image %s", img_id, exc_info=True)
            return None

    @staticmethod
    def _build_image_lookup(chunk: Chunk) -> dict[str, str]:
        """Build ``{image_id: path}`` from chunk metadata ``images`` list."""
        lookup: dict[str, str] = {}
        for img_meta in chunk.metadata.get("images", []):
            iid = img_meta.get("id")
            ipath = img_meta.get("path")
            if iid and ipath:
                lookup[iid] = ipath
        return lookup

    @staticmethod
    def _mark_unprocessed(chunk: Chunk) -> Chunk:
        """Return a new Chunk with ``has_unprocessed_images`` flag."""
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata={**chunk.metadata, "has_unprocessed_images": True},
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    def _load_prompt(self) -> str:
        """Load and cache the captioning prompt."""
        if self._prompt is not None:
            return self._prompt

        path = Path(self._prompt_path)
        if path.exists():
            self._prompt = path.read_text(encoding="utf-8").strip()
        else:
            self._prompt = "Describe this image in detail for indexing purposes."
            logger.warning("Prompt file not found: %s, using default", self._prompt_path)

        return self._prompt
