"""Integration test: Multi-Representation Indexing.

Verifies that code chunks are embedded using their retrieval_text (LLM summary),
not the raw code. Uses a deterministic word-overlap embedding to confirm
the semantic gap is bridged.
"""

from __future__ import annotations

import math

from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.libs.embedding.base_embedding import BaseEmbedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class WordOverlapEmbedding(BaseEmbedding):
    """Deterministic embedding using word-frequency vectors.

    Maps each unique word to a dimension, counts occurrences.
    This makes cosine similarity proportional to word overlap,
    which lets us verify that the summary (natural language)
    is closer to the query than the raw code.
    """

    VOCAB = [
        "function",
        "calculates",
        "invariant",
        "mass",
        "four",
        "vectors",
        "energy",
        "momentum",
        "physics",
        "analysis",
        "particle",
        "dimuon",
        "def",
        "return",
        "self",
        "import",
        "class",
        "p1",
        "p2",
        "sqrt",
    ]

    def embed(self, texts: list[str], trace=None, **kwargs) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            vec = [float(lower.count(w)) for w in self.VOCAB]
            vectors.append(vec)
        return vectors


class TestMultiRepresentationIntegration:
    """Verify that summary embeddings are closer to NL queries than code embeddings."""

    def test_summary_embedding_closer_to_query(self):
        """Core assertion: cos(query, summary) > cos(query, code)."""
        embedding = WordOverlapEmbedding()
        encoder = DenseEncoder(embedding, batch_size=10)

        code_text = (
            "def compute_mass(p1, p2):\n"
            "    e = p1.E + p2.E\n"
            "    px = p1.px + p2.px\n"
            "    return math.sqrt(e**2 - px**2 - py**2 - pz**2)"
        )
        summary_text = (
            "This function calculates the invariant mass of a dimuon system "
            "from two particle four-vectors. It sums the energy and momentum "
            "components and applies the relativistic invariant mass formula."
        )
        query = "How to calculate invariant mass from particle four vectors in physics analysis?"

        # Embed all three
        code_chunk = Chunk(
            id="code",
            text=code_text,
            metadata={"source_path": "test.py"},
        )
        summary_chunk = Chunk(
            id="summary",
            text=summary_text,
            metadata={"source_path": "test.py"},
        )
        query_chunk = Chunk(
            id="query",
            text=query,
            metadata={"source_path": "query"},
        )

        vecs = encoder.encode([code_chunk, summary_chunk, query_chunk])
        code_vec, summary_vec, query_vec = vecs

        sim_query_code = cosine_similarity(query_vec, code_vec)
        sim_query_summary = cosine_similarity(query_vec, summary_vec)

        # Summary should be closer to the query than raw code
        assert sim_query_summary > sim_query_code, (
            f"Expected summary closer to query: "
            f"cos(query, summary)={sim_query_summary:.4f} > "
            f"cos(query, code)={sim_query_code:.4f}"
        )

    def test_retrieval_text_used_for_embedding(self):
        """Verify DenseEncoder reads retrieval_text, not chunk.text."""
        embedding = WordOverlapEmbedding()
        encoder = DenseEncoder(embedding, batch_size=10)

        summary = "This function calculates invariant mass from particle four-vectors."
        chunk_with_rt = Chunk(
            id="rt_chunk",
            text="def compute_mass(p1, p2): return sqrt(e**2 - px**2)",
            metadata={
                "source_path": "test.py",
                "retrieval_text": summary,
            },
        )
        chunk_just_summary = Chunk(
            id="summary_only",
            text=summary,
            metadata={"source_path": "test.py"},
        )

        vec_rt = encoder.encode([chunk_with_rt])[0]
        vec_summary = encoder.encode([chunk_just_summary])[0]

        # Should be identical — both embed the same summary text
        assert vec_rt == vec_summary
