"""Image storage with SQLite indexing for multimodal RAG.

Stores image files on the filesystem organised by collection, and
maintains a SQLite index for efficient lookup and querying.

Directory layout::

    data/images/{collection}/{image_id}.{ext}

Database schema::

    image_index (
        image_id   TEXT PRIMARY KEY,
        file_path  TEXT NOT NULL,
        collection TEXT,
        doc_hash   TEXT,
        page_num   INTEGER,
        created_at TEXT NOT NULL
    )

Design Principles:
    - Persistent: Images on disk, metadata in SQLite
    - Concurrent: WAL mode for concurrent reads/writes
    - Idempotent: INSERT OR REPLACE for safe re-saves
    - Organised: Namespace isolation by collection
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ImageStorage:
    """SQLite-backed image storage manager.

    Args:
        db_path: Path to SQLite database file.
        images_root: Root directory for image file storage.
    """

    def __init__(
        self,
        db_path: str = "data/db/image_index.db",
        images_root: str = "data/images",
    ) -> None:
        self.db_path = db_path
        self.images_root = Path(images_root)
        self._ensure_database()

    def close(self) -> None:
        """No-op — connections are per-operation. Kept for API symmetry."""

    def __del__(self) -> None:
        self.close()

    # ── save / register ──────────────────────────────────────────────

    def save_image(
        self,
        image_id: str,
        image_data: bytes | Path | str,
        collection: str | None = None,
        doc_hash: str | None = None,
        page_num: int | None = None,
        extension: str = "png",
    ) -> str:
        """Save image to filesystem and register in database.

        Idempotent — re-saving the same *image_id* updates both the
        file and the metadata row.

        Args:
            image_id: Unique identifier for the image.
            image_data: Raw bytes **or** path to an existing source file.
            collection: Optional namespace for organisation.
            doc_hash: Optional document hash for traceability.
            page_num: Optional page number from source document.
            extension: File extension without dot (default ``png``).

        Returns:
            Absolute path where the image was saved.

        Raises:
            ValueError: If *image_id* is blank.
            IOError: If the file cannot be written.
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")

        collection_dir = self.images_root / (collection or "default")
        collection_dir.mkdir(parents=True, exist_ok=True)

        image_path = collection_dir / f"{image_id}.{extension}"

        try:
            if isinstance(image_data, bytes):
                image_path.write_bytes(image_data)
            elif isinstance(image_data, (Path, str)):
                source = Path(image_data)
                if not source.exists():
                    raise FileNotFoundError(f"Source image not found: {source}")
                shutil.copy2(source, image_path)
            else:
                raise ValueError(
                    f"Unsupported image_data type: {type(image_data)}"
                )
        except Exception as e:
            raise IOError(f"Failed to save image {image_id}: {e}") from e

        stored_path = str(image_path.resolve())
        self._register(image_id, stored_path, collection, doc_hash, page_num)
        return stored_path

    def register_image(
        self,
        image_id: str,
        file_path: Path | str,
        collection: str | None = None,
        doc_hash: str | None = None,
        page_num: int | None = None,
    ) -> str:
        """Register an **existing** image file in the database index.

        Unlike :meth:`save_image`, this does *not* copy the file.

        Returns:
            Absolute path to the registered image.

        Raises:
            ValueError: If *image_id* is blank.
            FileNotFoundError: If the image file does not exist.
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        stored_path = str(path.resolve())
        self._register(image_id, stored_path, collection, doc_hash, page_num)
        return stored_path

    # ── query ────────────────────────────────────────────────────────

    def get_image_path(self, image_id: str) -> str | None:
        """Return absolute path for *image_id*, or ``None``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,),
            ).fetchone()
            return row[0] if row else None

    def image_exists(self, image_id: str) -> bool:
        """Return ``True`` if *image_id* is registered."""
        return self.get_image_path(image_id) is not None

    def list_images(
        self,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        """List images with optional filtering.

        Returns:
            List of metadata dicts (``image_id``, ``file_path``,
            ``collection``, ``doc_hash``, ``page_num``, ``created_at``).
        """
        query = "SELECT * FROM image_index WHERE 1=1"
        params: list[Any] = []

        if collection is not None:
            query += " AND collection = ?"
            params.append(collection)
        if doc_hash is not None:
            query += " AND doc_hash = ?"
            params.append(doc_hash)

        query += " ORDER BY created_at ASC"

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ── delete ───────────────────────────────────────────────────────

    def delete_image(self, image_id: str, remove_file: bool = True) -> bool:
        """Delete image from database and optionally from filesystem.

        Returns:
            ``True`` if the image was deleted, ``False`` if not found.
        """
        file_path = self.get_image_path(image_id)
        if file_path is None:
            return False

        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM image_index WHERE image_id = ?", (image_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0

        if remove_file and deleted:
            Path(file_path).unlink(missing_ok=True)

        return deleted

    # ── stats ────────────────────────────────────────────────────────

    def get_collection_stats(self, collection: str) -> dict[str, Any]:
        """Return ``{"total_images": int, "total_size_bytes": int}``."""
        images = self.list_images(collection=collection)

        total_size = 0
        for img in images:
            try:
                p = Path(img["file_path"])
                if p.exists():
                    total_size += p.stat().st_size
            except Exception:
                pass

        return {"total_images": len(images), "total_size_bytes": total_size}

    # ── internals ────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Return a fresh connection (per-operation for concurrency)."""
        return sqlite3.connect(self.db_path)

    def _ensure_database(self) -> None:
        """Create database file, schema, and indexes if needed."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.images_root.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id   TEXT PRIMARY KEY,
                    file_path  TEXT NOT NULL,
                    collection TEXT,
                    doc_hash   TEXT,
                    page_num   INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection "
                "ON image_index(collection)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_hash "
                "ON image_index(doc_hash)"
            )
            conn.commit()
        finally:
            conn.close()

    def _register(
        self,
        image_id: str,
        stored_path: str,
        collection: str | None,
        doc_hash: str | None,
        page_num: int | None,
    ) -> None:
        """Insert or replace a row in the image_index table."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO image_index "
                "(image_id, file_path, collection, doc_hash, page_num, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (image_id, stored_path, collection, doc_hash, page_num, now),
            )
            conn.commit()
