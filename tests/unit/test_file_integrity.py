"""Unit tests for file integrity checker.

Tests SHA256 hashing, incremental ingestion tracking, and SQLite persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)


class TestSQLiteIntegrityChecker:
    """Tests for SQLiteIntegrityChecker implementation."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> str:
        """Create temporary database path."""
        return str(tmp_path / "test_integrity.db")

    @pytest.fixture
    def checker(self, temp_db: str) -> SQLiteIntegrityChecker:
        """Create checker instance with temp database."""
        return SQLiteIntegrityChecker(db_path=temp_db)

    @pytest.fixture
    def temp_file(self, tmp_path: Path) -> Path:
        """Create temporary file with known content."""
        file_path = tmp_path / "test_file.txt"
        file_path.write_text("Hello, World!")
        return file_path

    def test_compute_sha256_same_content_same_hash(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """Same file content should produce identical hash."""
        hash1 = checker.compute_sha256(str(temp_file))
        hash2 = checker.compute_sha256(str(temp_file))

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64-character hex string

    def test_compute_sha256_different_content_different_hash(self, checker: SQLiteIntegrityChecker, tmp_path: Path):
        """Different content should produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content 1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content 2")

        hash1 = checker.compute_sha256(str(file1))
        hash2 = checker.compute_sha256(str(file2))

        assert hash1 != hash2

    def test_compute_sha256_file_not_found(self, checker: SQLiteIntegrityChecker):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            checker.compute_sha256("/non/existent/file.txt")

    def test_compute_sha256_directory_raises_error(self, checker: SQLiteIntegrityChecker, tmp_path: Path):
        """Should raise IOError when path is a directory."""
        with pytest.raises(IOError, match="Path is not a file"):
            checker.compute_sha256(str(tmp_path))

    def test_compute_sha256_large_file(self, checker: SQLiteIntegrityChecker, tmp_path: Path):
        """Should handle large files using chunked reading."""
        # Create a file larger than 64KB
        large_file = tmp_path / "large.bin"
        large_file.write_bytes(b"X" * (100 * 1024))  # 100KB

        file_hash = checker.compute_sha256(str(large_file))

        assert len(file_hash) == 64
        # Verify hash is deterministic
        assert file_hash == checker.compute_sha256(str(large_file))

    def test_should_skip_new_file(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """New file should not be skipped."""
        file_hash = checker.compute_sha256(str(temp_file))

        assert checker.should_skip(file_hash) is False

    def test_should_skip_after_mark_success(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """File should be skipped after mark_success."""
        file_hash = checker.compute_sha256(str(temp_file))

        checker.mark_success(file_hash, str(temp_file))

        assert checker.should_skip(file_hash) is True

    def test_should_not_skip_after_mark_failed(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """Failed file should not be skipped (allows retry)."""
        file_hash = checker.compute_sha256(str(temp_file))

        checker.mark_failed(file_hash, str(temp_file), "Test error")

        assert checker.should_skip(file_hash) is False

    def test_mark_success_with_collection(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """Should store collection metadata."""
        file_hash = checker.compute_sha256(str(temp_file))

        checker.mark_success(file_hash, str(temp_file), collection="test_collection")

        assert checker.should_skip(file_hash) is True

    def test_mark_success_idempotent(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """Multiple mark_success calls should be idempotent."""
        file_hash = checker.compute_sha256(str(temp_file))

        checker.mark_success(file_hash, str(temp_file))
        checker.mark_success(file_hash, str(temp_file))

        assert checker.should_skip(file_hash) is True

    def test_mark_failed_records_error(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """Should record error message."""
        file_hash = checker.compute_sha256(str(temp_file))
        error_msg = "Parsing failed: invalid format"

        checker.mark_failed(file_hash, str(temp_file), error_msg)

        # Failed files should not be skipped
        assert checker.should_skip(file_hash) is False

    def test_mark_success_overwrites_failed(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """mark_success should overwrite previous failed status."""
        file_hash = checker.compute_sha256(str(temp_file))

        checker.mark_failed(file_hash, str(temp_file), "Initial failure")
        assert checker.should_skip(file_hash) is False

        checker.mark_success(file_hash, str(temp_file))
        assert checker.should_skip(file_hash) is True

    def test_database_auto_created(self, tmp_path: Path):
        """Database file and parent directories should be auto-created."""
        db_path = tmp_path / "nested" / "dir" / "integrity.db"

        checker = SQLiteIntegrityChecker(db_path=str(db_path))

        assert db_path.exists()
        assert db_path.parent.exists()

    def test_wal_mode_enabled(self, temp_db: str, checker: SQLiteIntegrityChecker):
        """Database should use WAL mode for concurrent access."""
        import sqlite3

        conn = sqlite3.connect(temp_db)
        try:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            conn.close()

    def test_concurrent_writes(self, temp_db: str, temp_file: Path):
        """Multiple checker instances should be able to write concurrently."""
        checker1 = SQLiteIntegrityChecker(db_path=temp_db)
        checker2 = SQLiteIntegrityChecker(db_path=temp_db)

        hash1 = checker1.compute_sha256(str(temp_file))

        # Both instances mark success
        checker1.mark_success(hash1, str(temp_file), collection="col1")
        checker2.mark_success(hash1, str(temp_file), collection="col2")

        # Both should see the updated status
        assert checker1.should_skip(hash1) is True
        assert checker2.should_skip(hash1) is True

    def test_remove_record_success(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """Should successfully remove an existing record."""
        file_hash = checker.compute_sha256(str(temp_file))
        checker.mark_success(file_hash, str(temp_file))

        result = checker.remove_record(file_hash)

        assert result is True
        assert checker.should_skip(file_hash) is False

    def test_remove_record_not_found(self, checker: SQLiteIntegrityChecker):
        """Should return False when removing non-existent record."""
        result = checker.remove_record("nonexistent_hash")

        assert result is False

    def test_list_processed_empty(self, checker: SQLiteIntegrityChecker):
        """Should return empty list when no files processed."""
        result = checker.list_processed()

        assert result == []

    def test_list_processed_success_only(self, checker: SQLiteIntegrityChecker, tmp_path: Path):
        """Should only list successfully processed files."""
        file1 = tmp_path / "success.txt"
        file1.write_text("Success")
        hash1 = checker.compute_sha256(str(file1))
        checker.mark_success(hash1, str(file1), collection="col1")

        file2 = tmp_path / "failed.txt"
        file2.write_text("Failed")
        hash2 = checker.compute_sha256(str(file2))
        checker.mark_failed(hash2, str(file2), "Error")

        result = checker.list_processed()

        assert len(result) == 1
        assert result[0]["file_hash"] == hash1
        assert result[0]["file_path"] == str(file1)
        assert result[0]["collection"] == "col1"
        assert "processed_at" in result[0]
        assert "updated_at" in result[0]

    def test_list_processed_filter_by_collection(self, checker: SQLiteIntegrityChecker, tmp_path: Path):
        """Should filter results by collection."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("File 1")
        hash1 = checker.compute_sha256(str(file1))
        checker.mark_success(hash1, str(file1), collection="col1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("File 2")
        hash2 = checker.compute_sha256(str(file2))
        checker.mark_success(hash2, str(file2), collection="col2")

        result = checker.list_processed(collection="col1")

        assert len(result) == 1
        assert result[0]["file_hash"] == hash1
        assert result[0]["collection"] == "col1"

    def test_list_processed_ordered_by_time(self, checker: SQLiteIntegrityChecker, tmp_path: Path):
        """Should return results ordered by processed_at ascending."""
        files = []
        for i in range(3):
            file = tmp_path / f"file{i}.txt"
            file.write_text(f"Content {i}")
            file_hash = checker.compute_sha256(str(file))
            checker.mark_success(file_hash, str(file))
            files.append((file_hash, str(file)))

        result = checker.list_processed()

        assert len(result) == 3
        # Should be in order of insertion (processed_at ascending)
        for i, (expected_hash, expected_path) in enumerate(files):
            assert result[i]["file_hash"] == expected_hash
            assert result[i]["file_path"] == expected_path

    def test_close_connection(self, checker: SQLiteIntegrityChecker, temp_file: Path):
        """Should be able to close and reopen connection."""
        file_hash = checker.compute_sha256(str(temp_file))
        checker.mark_success(file_hash, str(temp_file))

        checker.close()

        # Should still work after close (creates new connection)
        assert checker.should_skip(file_hash) is True

    def test_abstract_base_class_cannot_instantiate(self):
        """FileIntegrityChecker ABC should not be directly instantiable."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FileIntegrityChecker()
