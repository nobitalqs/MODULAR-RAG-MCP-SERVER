"""BM25 Indexer for building and querying inverted indexes.

Receives term statistics from SparseEncoder, computes IDF scores,
builds inverted index structures, and persists to disk.

Index Structure::

    {
        "metadata": {"num_docs", "avg_doc_length", "total_terms", "collection"},
        "index": {
            "term": {
                "idf": float,
                "df": int,
                "postings": [{"chunk_id", "tf", "doc_length"}, ...]
            }
        }
    }

BM25 IDF Formula:
    IDF(term) = log((N - df + 0.5) / (df + 0.5))

Design Principles:
    - Idempotent: Rebuild produces same results for same input
    - Observable: Accepts TraceContext for future integration
    - Persistent: Indexes saved to data/db/bm25/ via atomic writes
    - Deterministic: Same corpus produces same IDF scores
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class BM25Indexer:
    """Build and query BM25 inverted indexes.

    Args:
        index_dir: Directory for persisted index files (default ``data/db/bm25``).
        k1: Term frequency saturation parameter (default 1.5).
        b: Length normalization parameter (default 0.75).

    Raises:
        ValueError: If *k1* <= 0 or *b* not in [0, 1].
    """

    def __init__(
        self,
        index_dir: str = "data/db/bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b}")

        self.index_dir = Path(index_dir)
        self.k1 = k1
        self.b = b

        self._index: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, Any] = {}

    # ── build / rebuild ──────────────────────────────────────────────

    def build(
        self,
        term_stats: list[dict[str, Any]],
        collection: str = "default",
        trace: Any | None = None,
    ) -> None:
        """Build BM25 index from SparseEncoder output.

        Args:
            term_stats: List of per-chunk dicts with ``chunk_id``,
                ``term_frequencies``, and ``doc_length``.
            collection: Collection name for organizing indexes.
            trace: Optional TraceContext for observability.

        Raises:
            ValueError: If *term_stats* is empty or has invalid structure.
        """
        if not term_stats:
            raise ValueError("Cannot build index from empty term_stats")

        self._validate_term_stats(term_stats)

        # Corpus-level statistics
        num_docs = len(term_stats)
        total_length = sum(stat["doc_length"] for stat in term_stats)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0

        # Document frequency per term
        doc_freq: dict[str, int] = {}
        for stat in term_stats:
            for term in stat["term_frequencies"]:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        # Build inverted index
        index: dict[str, dict[str, Any]] = {}
        for term, df in doc_freq.items():
            idf = self._calculate_idf(num_docs, df)

            postings = [
                {
                    "chunk_id": stat["chunk_id"],
                    "tf": stat["term_frequencies"][term],
                    "doc_length": stat["doc_length"],
                }
                for stat in term_stats
                if stat["term_frequencies"].get(term, 0) > 0
            ]

            index[term] = {"idf": idf, "df": df, "postings": postings}

        self._metadata = {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "total_terms": len(index),
            "collection": collection,
        }
        self._index = index

        self._save(collection)

    def rebuild(
        self,
        term_stats: list[dict[str, Any]],
        collection: str = "default",
        trace: Any | None = None,
    ) -> None:
        """Rebuild index from scratch (clear-intent alias for :meth:`build`)."""
        self.build(term_stats, collection, trace)

    # ── load ─────────────────────────────────────────────────────────

    def load(
        self,
        collection: str = "default",
        trace: Any | None = None,
    ) -> bool:
        """Load index from disk.

        Returns:
            ``True`` if loaded successfully, ``False`` if file not found.

        Raises:
            ValueError: If index file is corrupted or has invalid structure.
        """
        index_path = self._get_index_path(collection)

        if not index_path.exists():
            return False

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted index file at {index_path}: {e}") from e

        if "metadata" not in data or "index" not in data:
            raise ValueError(
                f"Invalid index file structure: missing metadata or index"
            )

        self._metadata = data["metadata"]
        self._index = data["index"]
        return True

    # ── query ────────────────────────────────────────────────────────

    def query(
        self,
        query_terms: list[str],
        top_k: int = 10,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Query the index using BM25 scoring.

        Args:
            query_terms: Terms to search for.
            top_k: Maximum results to return.
            trace: Optional TraceContext for observability.

        Returns:
            Results sorted by BM25 score descending:
            ``[{"chunk_id": str, "score": float}, ...]``

        Raises:
            ValueError: If index not loaded or *query_terms* empty.
        """
        if not self._index:
            raise ValueError("Index not loaded. Call load() or build() first.")

        if not query_terms:
            raise ValueError("query_terms cannot be empty")

        scores: dict[str, float] = {}

        for term in query_terms:
            if term not in self._index:
                continue

            term_data = self._index[term]
            idf = term_data["idf"]

            for posting in term_data["postings"]:
                chunk_id = posting["chunk_id"]
                term_score = self._calculate_bm25_score(
                    tf=posting["tf"],
                    doc_length=posting["doc_length"],
                    avg_doc_length=self._metadata["avg_doc_length"],
                    idf=idf,
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        sorted_results = sorted(
            [{"chunk_id": cid, "score": score} for cid, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )
        return sorted_results[:top_k]

    # ── document removal ─────────────────────────────────────────────

    def remove_document(
        self,
        doc_id: str,
        collection: str = "default",
    ) -> bool:
        """Remove all postings whose ``chunk_id`` starts with *doc_id*.

        Recalculates IDF and metadata after removal and re-saves.

        Returns:
            ``True`` if any postings were removed.
        """
        if not self._index:
            if not self.load(collection):
                return False

        removed_any = False
        terms_to_delete: list[str] = []

        for term, term_data in self._index.items():
            original_len = len(term_data["postings"])
            term_data["postings"] = [
                p
                for p in term_data["postings"]
                if not p["chunk_id"].startswith(doc_id)
            ]
            if len(term_data["postings"]) < original_len:
                removed_any = True

            if not term_data["postings"]:
                terms_to_delete.append(term)
            else:
                term_data["df"] = len(term_data["postings"])

        for term in terms_to_delete:
            del self._index[term]

        if removed_any:
            # Recalculate global metadata
            all_chunk_ids: set[str] = set()
            total_length = 0
            for td in self._index.values():
                for p in td["postings"]:
                    all_chunk_ids.add(p["chunk_id"])
                    total_length += p["doc_length"]

            num_docs = len(all_chunk_ids)
            avg_doc_length = total_length / num_docs if num_docs else 0.0

            for td in self._index.values():
                td["idf"] = self._calculate_idf(num_docs, td["df"])

            self._metadata = {
                "num_docs": num_docs,
                "avg_doc_length": avg_doc_length,
                "total_terms": len(self._index),
                "collection": collection,
            }
            self._save(collection)

        return removed_any

    # ── BM25 math ────────────────────────────────────────────────────

    def _calculate_idf(self, num_docs: int, df: int) -> float:
        """IDF(term) = log((N - df + 0.5) / (df + 0.5))"""
        return math.log((num_docs - df + 0.5) / (df + 0.5))

    def _calculate_bm25_score(
        self,
        tf: int,
        doc_length: int,
        avg_doc_length: float,
        idf: float,
    ) -> float:
        """BM25 score = IDF * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))"""
        if avg_doc_length == 0:
            avg_doc_length = 1.0

        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (
            1 - self.b + self.b * (doc_length / avg_doc_length)
        )
        return idf * (numerator / denominator)

    # ── validation ───────────────────────────────────────────────────

    @staticmethod
    def _validate_term_stats(term_stats: list[dict[str, Any]]) -> None:
        """Validate term_stats structure."""
        for i, stat in enumerate(term_stats):
            if not isinstance(stat, dict):
                raise ValueError(
                    f"term_stats[{i}] must be a dict, got {type(stat)}"
                )

            for field in ("chunk_id", "term_frequencies", "doc_length"):
                if field not in stat:
                    raise ValueError(
                        f"term_stats[{i}] missing required field: {field}"
                    )

            if not isinstance(stat["term_frequencies"], dict):
                raise ValueError(
                    f"term_stats[{i}]['term_frequencies'] must be dict, "
                    f"got {type(stat['term_frequencies'])}"
                )

            if not isinstance(stat["doc_length"], int) or stat["doc_length"] < 0:
                raise ValueError(
                    f"term_stats[{i}]['doc_length'] must be non-negative int, "
                    f"got {stat['doc_length']}"
                )

    # ── persistence ──────────────────────────────────────────────────

    def _get_index_path(self, collection: str) -> Path:
        """Return file path for the collection's index."""
        return self.index_dir / f"{collection}_bm25.json"

    def _save(self, collection: str) -> None:
        """Atomically save index to disk (write-to-temp then rename)."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self._get_index_path(collection)
        temp_path = index_path.with_suffix(".tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"metadata": self._metadata, "index": self._index},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            temp_path.replace(index_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
