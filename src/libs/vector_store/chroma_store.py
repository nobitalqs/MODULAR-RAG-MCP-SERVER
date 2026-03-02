"""ChromaDB VectorStore implementation.

Provides persistent vector storage using ChromaDB with cosine similarity.
Supports upsert, query, delete, clear, and get_by_ids operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)
from src.libs.vector_store.base_vector_store import BaseVectorStore

# Try importing chromadb, with graceful fallback
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaStore(BaseVectorStore):
    """ChromaDB implementation of VectorStore.

    Uses PersistentClient for local storage with cosine similarity.
    Automatically sanitizes metadata to ChromaDB constraints
    (str/int/float/bool only).

    Args:
        persist_directory: Directory for ChromaDB persistence.
            Defaults to ``./data/db/chroma``.
        collection_name: Name of the collection.
            Defaults to ``knowledge_hub``.
        **kwargs: Additional provider-specific parameters (ignored).

    Raises:
        ImportError: If chromadb is not installed.
    """

    def __init__(
        self,
        persist_directory: str = "./data/db/chroma",
        collection_name: str = "knowledge_hub",
        **kwargs: Any,
    ) -> None:
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. "
                "Install with: pip install chromadb"
            )

        # Resolve to absolute path
        persist_dir = resolve_path(persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self._collection_name = collection_name

        # Get or create collection with cosine similarity
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        records: list[dict[str, Any]],
        trace: Any = None,
        **kwargs: Any,
    ) -> None:
        """Insert or update records in ChromaDB.

        Each record must contain:
        - ``id`` (str): Unique identifier.
        - ``vector`` (list[float]): Embedding vector.
        - ``text`` (str, optional): Document text.
        - ``metadata`` (dict, optional): Metadata fields.

        Args:
            records: Records to upsert.
            trace: Optional TraceContext for observability (unused).
            **kwargs: Additional parameters (unused).

        Raises:
            ValueError: If records are empty or malformed.
        """
        self.validate_records(records)

        ids = [record["id"] for record in records]
        embeddings = [record["vector"] for record in records]
        documents = [record.get("text", "") for record in records]

        # Sanitize metadata - ChromaDB requires non-empty dicts
        metadatas = []
        for record in records:
            metadata = self._sanitize_metadata(record.get("metadata", {}))
            # If metadata is empty, add a dummy field
            if not metadata:
                metadata = {"_empty": True}
            metadatas.append(metadata)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Query ChromaDB for similar vectors.

        Args:
            vector: Query embedding vector.
            top_k: Maximum number of results.
            filters: Optional metadata filters (ChromaDB where clause).
            trace: Optional TraceContext for observability (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            List of results sorted by similarity (descending), each with:
            - ``id`` (str): Document ID.
            - ``score`` (float): Similarity score (1 = identical, 0 = orthogonal).
            - ``text`` (str): Document text.
            - ``metadata`` (dict): Metadata fields.

        Raises:
            ValueError: If vector or top_k is invalid.
        """
        self.validate_query_vector(vector, top_k)

        # Build where clause from filters
        where = self._build_where_clause(filters) if filters else None

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"],
        )

        # Transform results
        output = []
        if results["ids"] and results["ids"][0]:
            for idx, doc_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score
                # ChromaDB cosine distance: 0=identical, 2=opposite
                # Similarity: 1=identical, 0=orthogonal, -1=opposite
                distance = results["distances"][0][idx]
                score = 1.0 - (distance / 2.0)

                output.append({
                    "id": doc_id,
                    "score": score,
                    "text": results["documents"][0][idx] if results["documents"] else "",
                    "metadata": results["metadatas"][0][idx] if results["metadatas"] else {},
                })

        return output

    def delete(
        self,
        ids: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> None:
        """Delete records by IDs.

        Args:
            ids: List of document IDs to delete.
            trace: Optional TraceContext for observability (unused).
            **kwargs: Additional parameters (unused).

        Raises:
            ValueError: If ids is empty.
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")

        self._collection.delete(ids=ids)

    def clear(
        self,
        collection_name: str | None = None,
        trace: Any = None,
        **kwargs: Any,
    ) -> None:
        """Clear all records from the collection.

        Deletes and recreates the collection to ensure a clean state.

        Args:
            collection_name: Ignored (uses instance collection).
            trace: Optional TraceContext for observability (unused).
            **kwargs: Additional parameters (unused).
        """
        # Delete and recreate collection
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_by_ids(
        self,
        ids: list[str],
        trace: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Retrieve records by IDs.

        Args:
            ids: List of document IDs to retrieve.
            trace: Optional TraceContext for observability (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            List of records in the same order as ``ids``, each with:
            - ``id`` (str): Document ID.
            - ``text`` (str): Document text.
            - ``metadata`` (dict): Metadata fields.

        Raises:
            ValueError: If ids is empty.
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")

        # Get records from ChromaDB
        result = self._collection.get(
            ids=ids,
            include=["metadatas", "documents"],
        )

        # Build a mapping for fast lookup
        id_to_record = {}
        if result["ids"]:
            for idx, doc_id in enumerate(result["ids"]):
                id_to_record[doc_id] = {
                    "id": doc_id,
                    "text": result["documents"][idx] if result["documents"] else "",
                    "metadata": result["metadatas"][idx] if result["metadatas"] else {},
                }

        # Return in the same order as input ids
        output = []
        for doc_id in ids:
            if doc_id in id_to_record:
                output.append(id_to_record[doc_id])

        return output

    # ------------------------------------------------------------------
    # Dashboard / DocumentManager convenience methods
    # (not part of BaseVectorStore ABC)
    # ------------------------------------------------------------------

    @property
    def collection(self) -> Any:
        """Expose the underlying ChromaDB collection object."""
        return self._collection

    def list_collections(self) -> list[str]:
        """Return names of all collections managed by this client."""
        return [col.name for col in self._client.list_collections()]

    def count(self, collection_name: str | None = None) -> int:
        """Return the number of records in a collection.

        Args:
            collection_name: Target collection. If *None*, uses the
                instance's default collection.

        Returns:
            Record count.
        """
        if collection_name is None or collection_name == self._collection_name:
            return self._collection.count()
        col = self._client.get_collection(name=collection_name)
        return col.count()

    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        trace: Any = None,
    ) -> int:
        """Delete records matching a metadata filter.

        Args:
            filter_dict: Metadata key/value pairs to match
                (e.g. ``{"doc_hash": "abc123"}``).
            trace: Optional TraceContext for observability.

        Returns:
            Number of records deleted.

        Raises:
            ValueError: If *filter_dict* is empty.
            RuntimeError: If the operation fails.
        """
        if not filter_dict:
            raise ValueError("filter_dict cannot be empty")

        try:
            where = self._build_where_clause(filter_dict)
            results = self._collection.get(where=where, include=[])
            matching_ids = results.get("ids", [])

            if not matching_ids:
                logger.debug(
                    "delete_by_metadata: no records matched %s", filter_dict
                )
                return 0

            self._collection.delete(ids=matching_ids)
            logger.info(
                "delete_by_metadata: deleted %d records matching %s",
                len(matching_ids),
                filter_dict,
            )
            return len(matching_ids)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete by metadata {filter_dict}: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metadata to ChromaDB constraints.

        ChromaDB only accepts str/int/float/bool values.
        - Converts lists to comma-separated strings.
        - Skips None values.
        - Skips unsupported types.

        Args:
            metadata: Raw metadata dict.

        Returns:
            Sanitized metadata dict.
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists to comma-separated strings
                sanitized[key] = ",".join(str(item) for item in value)
            # Skip unsupported types (dict, etc.)
        return sanitized

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """Build ChromaDB where clause from filters.

        Passes through the filters dict as-is, assuming it follows
        ChromaDB's where clause format.

        Args:
            filters: Metadata filters.

        Returns:
            Where clause for ChromaDB query, or None if filters is empty.
        """
        if not filters:
            return None
        # ChromaDB supports operator dicts like {"$eq": value}
        # For now, pass through as-is
        return filters
