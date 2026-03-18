"""Multimodal content assembler for MCP responses.

Reads image files referenced by retrieval results and encodes them
as Base64 MCP ImageContent blocks for multimodal responses.

Pipeline integration:
    Ingestion: PdfLoader extracts images → ImageStorage saves to disk
    Query:     ResponseBuilder → MultimodalAssembler → Base64 ImageContent
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from mcp import types

logger = logging.getLogger(__name__)

# Supported MIME types by file extension
_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
}


class MultimodalAssembler:
    """Assembles image content blocks from retrieval results.

    Looks up image_refs in each result's metadata, reads the image
    files from disk, and encodes them as Base64 ImageContent.

    Args:
        image_storage: Optional ImageStorage instance for path lookup.
            If None, falls back to reading paths directly from metadata.
        max_images: Maximum images to include per response (default: 5).
        max_image_size_bytes: Skip images larger than this (default: 10MB).
    """

    def __init__(
        self,
        image_storage: Any | None = None,
        max_images: int = 5,
        max_image_size_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self._image_storage = image_storage
        self._max_images = max_images
        self._max_image_size_bytes = max_image_size_bytes

    def assemble(
        self,
        results: list[Any],
        collection: str | None = None,
    ) -> list[types.ImageContent]:
        """Extract and encode images from retrieval results.

        Args:
            results: List of RetrievalResult with metadata.image_refs.
            collection: Optional collection for ImageStorage lookup.

        Returns:
            List of MCP ImageContent blocks (Base64 encoded).
        """
        image_contents: list[types.ImageContent] = []
        seen_ids: set[str] = set()

        for result in results:
            metadata = getattr(result, "metadata", {}) or {}
            image_refs = metadata.get("image_refs", [])
            images_meta = metadata.get("images", [])

            # Collect image paths from image_refs or images metadata
            image_paths = self._resolve_image_paths(
                image_refs,
                images_meta,
                collection,
            )

            for image_id, image_path in image_paths:
                if image_id in seen_ids:
                    continue
                if len(image_contents) >= self._max_images:
                    break

                content = self._encode_image(image_id, image_path)
                if content is not None:
                    image_contents.append(content)
                    seen_ids.add(image_id)

            if len(image_contents) >= self._max_images:
                break

        return image_contents

    def _resolve_image_paths(
        self,
        image_refs: list[str],
        images_meta: list[dict[str, Any]],
        collection: str | None,
    ) -> list[tuple[str, str]]:
        """Resolve image IDs to file paths.

        Tries three sources in order:
        1. ImageStorage.get_image_path() if available
        2. images metadata list (has 'id' and 'path' fields)
        3. Skip if unresolvable

        Returns:
            List of (image_id, file_path) tuples.
        """
        resolved: list[tuple[str, str]] = []

        # From image_refs (list of image_id strings)
        for image_id in image_refs:
            path = self._lookup_path(image_id)
            if path:
                resolved.append((image_id, path))

        # From images metadata (list of dicts with 'id' and 'path')
        for img_meta in images_meta:
            image_id = img_meta.get("id", "")
            path = img_meta.get("path", "")
            if image_id and path and image_id not in {r[0] for r in resolved}:
                resolved.append((image_id, path))

        return resolved

    def _lookup_path(self, image_id: str) -> str | None:
        """Look up image path via ImageStorage if available."""
        if self._image_storage is None:
            return None
        try:
            return self._image_storage.get_image_path(image_id)
        except Exception as exc:
            logger.debug("ImageStorage lookup failed for %s: %s", image_id, exc)
            return None

    def _encode_image(
        self,
        image_id: str,
        image_path: str,
    ) -> types.ImageContent | None:
        """Read and Base64-encode an image file.

        Returns None if the file doesn't exist, is too large, or
        has an unsupported format.
        """
        path = Path(image_path)

        if not path.exists():
            logger.debug("Image file not found: %s", image_path)
            return None

        # Check file size
        try:
            size = path.stat().st_size
            if size > self._max_image_size_bytes:
                logger.warning(
                    "Image %s too large (%d bytes), skipping",
                    image_id,
                    size,
                )
                return None
        except OSError as exc:
            logger.debug("Cannot stat image %s: %s", image_path, exc)
            return None

        # Determine MIME type
        mime_type = _MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:
            logger.debug("Unsupported image format: %s", path.suffix)
            return None

        # Read and encode
        try:
            image_bytes = path.read_bytes()
            b64_data = base64.b64encode(image_bytes).decode("ascii")

            return types.ImageContent(
                type="image",
                data=b64_data,
                mimeType=mime_type,
            )
        except Exception as exc:
            logger.warning("Failed to encode image %s: %s", image_id, exc)
            return None
