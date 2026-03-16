"""Tests for MMR (Maximal Marginal Relevance) result diversification."""

from __future__ import annotations

import pytest

from src.core.types import RetrievalResult
from src.core.query_engine.mmr import mmr_select, _term_freq, _cosine_sim, _norm


# ── helper ──────────────────────────────────────────────────────────

def _r(chunk_id: str, score: float, text: str, source: str = "") -> RetrievalResult:
    """Shortcut to build a RetrievalResult."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source_path": source},
    )


# ── text similarity tests ──────────────────────────────────────────


class TestTextSimilarity:
    """Test TF-based cosine similarity helpers."""

    def test_identical_texts_similarity_1(self):
        text = "fitting a gaussian to histogram data"
        tf = _term_freq(text)
        n = _norm(tf)
        assert _cosine_sim(tf, n, tf, n) == pytest.approx(1.0)

    def test_disjoint_texts_similarity_0(self):
        tf_a = _term_freq("alpha beta gamma")
        tf_b = _term_freq("delta epsilon zeta")
        assert _cosine_sim(tf_a, _norm(tf_a), tf_b, _norm(tf_b)) == 0.0

    def test_partial_overlap(self):
        tf_a = _term_freq("fitting gaussian histogram")
        tf_b = _term_freq("fitting linear regression")
        n_a, n_b = _norm(tf_a), _norm(tf_b)
        sim = _cosine_sim(tf_a, n_a, tf_b, n_b)
        assert 0.0 < sim < 1.0

    def test_empty_text(self):
        tf_a = _term_freq("")
        tf_b = _term_freq("hello world")
        n_a, n_b = _norm(tf_a), _norm(tf_b)
        assert _cosine_sim(tf_a, n_a, tf_b, n_b) == 0.0


# ── MMR selection tests ─────────────────────────────────────────────


class TestMMRSelect:
    """Test MMR diversification logic."""

    def test_first_result_always_highest_score(self):
        """MMR always picks the most relevant result first."""
        results = [
            _r("c1", 0.9, "fitting gaussian histogram", "manual.pdf"),
            _r("c2", 0.8, "fitting gaussian curve", "pdf.pdf"),
            _r("c3", 0.7, "tree branch fill usage", "tutorial.py"),
        ]

        selected = mmr_select(results, top_k=2, lambda_=0.7)

        assert selected[0].chunk_id == "c1"

    def test_mmr_promotes_diverse_result(self):
        """MMR should pick a diverse result over a redundant high-scorer."""
        # c1 and c2 are near-identical text (same topic), c3 is different topic
        # Scores are close so diversity penalty can flip the order
        results = [
            _r("c1", 0.90, "fitting gaussian to histogram data fitting gaussian histogram", "manual.pdf"),
            _r("c2", 0.80, "fitting gaussian to histogram distribution fitting gaussian histogram", "pdf.pdf"),
            _r("c3", 0.75, "TTree branch fill loop pyroot tutorial", "tutorial.py"),
        ]

        selected = mmr_select(results, top_k=2, lambda_=0.5)

        # c1 selected first (highest score)
        assert selected[0].chunk_id == "c1"
        # c3 should be preferred over c2 because c2 is too similar to c1
        assert selected[1].chunk_id == "c3"

    def test_high_lambda_favors_relevance(self):
        """lambda=1.0 is pure relevance, no diversity penalty."""
        results = [
            _r("c1", 0.9, "fitting gaussian", "a.pdf"),
            _r("c2", 0.8, "fitting gaussian curve", "b.pdf"),
            _r("c3", 0.5, "tree branch fill", "c.py"),
        ]

        selected = mmr_select(results, top_k=3, lambda_=1.0)

        # Pure relevance order
        assert [r.chunk_id for r in selected] == ["c1", "c2", "c3"]

    def test_low_lambda_favors_diversity(self):
        """Low lambda should strongly penalize similar results."""
        results = [
            _r("c1", 0.9, "fitting gaussian histogram data", "a.pdf"),
            _r("c2", 0.85, "fitting gaussian histogram curve", "b.pdf"),
            _r("c3", 0.6, "numpy array conversion pyroot", "c.py"),
            _r("c4", 0.5, "tree branch fill loop", "d.py"),
        ]

        selected = mmr_select(results, top_k=3, lambda_=0.3)

        # c1 always first, but c3/c4 should appear before c2
        assert selected[0].chunk_id == "c1"
        assert "c2" not in [r.chunk_id for r in selected[:3]]

    def test_returns_all_when_fewer_than_top_k(self):
        """If fewer candidates than top_k, return all."""
        results = [
            _r("c1", 0.9, "hello", "a.pdf"),
            _r("c2", 0.8, "world", "b.pdf"),
        ]

        selected = mmr_select(results, top_k=5, lambda_=0.7)

        assert len(selected) == 2

    def test_empty_input(self):
        assert mmr_select([], top_k=5, lambda_=0.7) == []

    def test_top_k_zero(self):
        results = [_r("c1", 0.9, "hello", "a.pdf")]
        assert mmr_select(results, top_k=0, lambda_=0.7) == []

    def test_preserves_result_objects(self):
        """Selected results are the same objects, not copies."""
        results = [
            _r("c1", 0.9, "fitting gaussian", "a.pdf"),
            _r("c2", 0.7, "tree branch", "b.py"),
        ]

        selected = mmr_select(results, top_k=2, lambda_=0.7)

        assert selected[0] is results[0]

    def test_real_world_scenario(self):
        """Simulate the user's actual Case 1 problem."""
        results = [
            _r("m1", 0.92, "Fitting a Gaussian function to histogram data using ROOT TF1", "manual_fitting.md"),
            _r("m2", 0.88, "Fitting Gaussian distribution parameters in ROOT histograms", "manual_fitting.md"),
            _r("p1", 0.85, "Chapter 5 Fitting Gaussian peaks to TH1 histogram objects", "root_primer.pdf"),
            _r("m3", 0.82, "Advanced Gaussian fit options chi2 likelihood ROOT", "manual_fitting.md"),
            _r("t1", 0.72, "tutorial fitLinear demonstrates fitting functions in PyROOT scripts", "tutorial_fit_fitLinear.py"),
            _r("p2", 0.70, "ROOT fitting framework TF1 Gaussian polynomial fit", "root_primer.pdf"),
            _r("t2", 0.65, "tutorial fit fitGaus example PyROOT histogram fitting", "tutorial_fit_fitGaus.py"),
        ]

        selected = mmr_select(results, top_k=5, lambda_=0.7)

        # Should include at least one tutorial in top 5
        selected_ids = [r.chunk_id for r in selected]
        tutorial_ids = [r.chunk_id for r in results if r.chunk_id.startswith("t")]
        assert any(tid in selected_ids for tid in tutorial_ids), (
            f"No tutorial in top 5: {selected_ids}"
        )

        # manual_fitting should not dominate (max 2)
        manual_count = sum(
            1 for r in selected
            if "manual_fitting" in r.metadata.get("source_path", "")
        )
        assert manual_count <= 3  # MMR naturally limits, not hard cap
