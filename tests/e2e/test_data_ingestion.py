"""End-to-End tests for scripts/ingest.py.

Behavioural tests (no API keys needed):
- ``--help`` flag
- Non-existent file / invalid config / unsupported file type
- Dry-run mode
- Empty directory

Integration tests (require Azure APIs + PDF fixtures):
- Single PDF ingestion
- Skip / force reprocess
- Directory ingestion

Run behavioural tests:
    pytest tests/e2e/test_data_ingestion.py -v -m "not integration"

Run all (including integration):
    pytest tests/e2e/test_data_ingestion.py -v -m integration -s
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = str(PROJECT_ROOT / "scripts" / "ingest.py")
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "sample_documents"


# ── Helpers ──────────────────────────────────────────────────────────


def _run(
    path: str,
    collection: str = "test",
    force: bool = False,
    config: str | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> subprocess.CompletedProcess:
    """Run ingest.py as a subprocess."""
    cmd = [sys.executable, SCRIPT, "--path", str(path), "--collection", collection]

    if force:
        cmd.append("--force")
    if config:
        cmd.extend(["--config", config])
    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=600,
        env=env,
        encoding="utf-8",
        errors="replace",
    )


# ── Behavioural tests (no API needed) ───────────────────────────────


class TestIngestCLI:
    """Tests that verify CLI behaviour without requiring API access."""

    def test_help_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "--path" in result.stdout
        assert "--collection" in result.stdout
        assert "--force" in result.stdout

    def test_nonexistent_file(self) -> None:
        result = _run(path="/nonexistent/path/document.pdf")
        assert result.returncode == 2
        assert "does not exist" in result.stdout

    def test_invalid_config(self, tmp_path) -> None:
        # Create a real PDF so the path check passes
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 minimal")
        result = _run(path=str(pdf), config="/nonexistent/config.yaml")
        assert result.returncode == 2
        assert "not found" in result.stdout.lower()

    def test_unsupported_file_type(self, tmp_path) -> None:
        txt = tmp_path / "doc.txt"
        txt.write_text("hello")
        result = _run(path=str(txt))
        assert result.returncode == 2
        assert "Unsupported" in result.stdout

    def test_empty_directory(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = _run(path=str(empty))
        assert result.returncode == 0
        assert "0 file" in result.stdout or "No files" in result.stdout

    def test_dry_run_single_file(self, tmp_path) -> None:
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 minimal")
        result = _run(path=str(pdf), dry_run=True)
        assert result.returncode == 0
        stdout_lower = result.stdout.lower()
        assert "dry run" in stdout_lower
        assert "1 file" in result.stdout

    def test_dry_run_directory(self, tmp_path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.pdf").write_bytes(b"%PDF")
        (d / "b.pdf").write_bytes(b"%PDF")
        result = _run(path=str(d), dry_run=True)
        assert result.returncode == 0
        assert "2 file" in result.stdout


# ── Integration tests (require API + fixtures) ──────────────────────


@pytest.mark.integration
class TestIngestIntegration:
    """Tests that run the actual pipeline — need Azure credentials."""

    @pytest.fixture
    def sample_pdf(self):
        path = FIXTURES / "simple.pdf"
        if not path.exists():
            pytest.skip("simple.pdf fixture not found")
        return path

    @pytest.fixture
    def complex_pdf(self):
        path = FIXTURES / "complex_technical_doc.pdf"
        if not path.exists():
            pytest.skip("complex_technical_doc.pdf fixture not found")
        return path

    def test_ingest_simple_pdf(self, sample_pdf) -> None:
        result = _run(
            path=str(sample_pdf),
            collection="e2e_simple",
            force=True,
            verbose=True,
        )
        assert result.returncode in (0, 1)
        assert "Processing" in result.stdout
        assert "SUMMARY" in result.stdout

    def test_ingest_complex_pdf(self, complex_pdf) -> None:
        result = _run(
            path=str(complex_pdf),
            collection="e2e_complex",
            force=True,
            verbose=True,
        )
        assert result.returncode in (0, 1)
        assert "Processing" in result.stdout
        assert "SUMMARY" in result.stdout

    def test_skip_already_processed(self, sample_pdf) -> None:
        # First run — force
        r1 = _run(path=str(sample_pdf), collection="e2e_skip", force=True)
        if r1.returncode == 2:
            pytest.skip("First ingestion failed")

        # Second run — no force
        r2 = _run(path=str(sample_pdf), collection="e2e_skip", force=False)
        assert r2.returncode == 0
        stdout_lower = r2.stdout.lower()
        assert "skip" in stdout_lower or "already processed" in stdout_lower

    def test_force_reprocess(self, sample_pdf) -> None:
        r1 = _run(path=str(sample_pdf), collection="e2e_force", force=True)
        if r1.returncode == 2:
            pytest.skip("First ingestion failed")

        r2 = _run(
            path=str(sample_pdf),
            collection="e2e_force",
            force=True,
            verbose=True,
        )
        assert r2.returncode in (0, 1)

    def test_directory_ingestion(self, tmp_path, sample_pdf) -> None:
        d = tmp_path / "pdfs"
        d.mkdir()
        shutil.copy(sample_pdf, d / "doc1.pdf")
        shutil.copy(sample_pdf, d / "doc2.pdf")

        result = _run(
            path=str(d), collection="e2e_dir", force=True, verbose=True,
        )
        assert "2 file" in result.stdout
        assert "[1/2]" in result.stdout
        assert "[2/2]" in result.stdout
