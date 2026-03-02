"""Tests for FileIntegrity.lookup_by_path()."""

from __future__ import annotations

import pytest

from src.libs.loader.file_integrity import SQLiteIntegrityChecker


@pytest.fixture
def checker(tmp_path):
    """Create a fresh integrity checker with tmp DB."""
    db_path = str(tmp_path / "test_integrity.db")
    return SQLiteIntegrityChecker(db_path=db_path)


class TestLookupByPath:
    """Tests for lookup_by_path method."""

    def test_returns_none_for_unknown_path(self, checker):
        """Unknown path returns None."""
        result = checker.lookup_by_path("/nonexistent/file.pdf")
        assert result is None

    def test_returns_hash_for_known_path(self, checker):
        """Returns file_hash for previously ingested path."""
        checker.mark_success("abc123hash", "/data/report.pdf", collection="default")
        result = checker.lookup_by_path("/data/report.pdf")
        assert result == "abc123hash"

    def test_respects_collection_filter(self, checker):
        """With collection filter, only returns matching collection."""
        checker.mark_success("hash_a", "/data/report.pdf", collection="alpha")
        checker.mark_success("hash_b", "/data/report.pdf", collection="beta")

        assert checker.lookup_by_path("/data/report.pdf", collection="alpha") == "hash_a"
        assert checker.lookup_by_path("/data/report.pdf", collection="beta") == "hash_b"

    def test_returns_none_for_wrong_collection(self, checker):
        """Returns None when path exists but in different collection."""
        checker.mark_success("hash_a", "/data/report.pdf", collection="alpha")
        result = checker.lookup_by_path("/data/report.pdf", collection="gamma")
        assert result is None

    def test_ignores_failed_records(self, checker):
        """Only returns records with status='success'."""
        checker.mark_failed("fail_hash", "/data/broken.pdf", error_msg="parse error")
        result = checker.lookup_by_path("/data/broken.pdf")
        assert result is None

    def test_returns_latest_on_multiple_versions(self, checker):
        """When same path ingested multiple times, returns the latest."""
        checker.mark_success("old_hash", "/data/report.pdf", collection="default")
        # Simulate re-ingestion with new hash (mark_success uses INSERT OR REPLACE)
        checker.mark_success("new_hash", "/data/report.pdf", collection="default")
        result = checker.lookup_by_path("/data/report.pdf", collection="default")
        assert result == "new_hash"

    def test_no_collection_filter_returns_any(self, checker):
        """Without collection filter, returns most recent across all collections."""
        checker.mark_success("hash_x", "/data/report.pdf", collection="coll_a")
        result = checker.lookup_by_path("/data/report.pdf")
        assert result == "hash_x"
