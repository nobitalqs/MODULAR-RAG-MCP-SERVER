#!/usr/bin/env python
"""Ingestion script for the Modular RAG MCP Server.

Provides a command-line interface for ingesting documents into the
knowledge hub.  Supports single files and recursive directories.

Usage:
    # Single PDF
    python scripts/ingest.py --path documents/report.pdf --collection contracts

    # All PDFs in a directory
    python scripts/ingest.py --path documents/ --collection tech_docs

    # Force re-processing
    python scripts/ingest.py --path documents/report.pdf -c contracts --force

    # Dry run (list files only)
    python scripts/ingest.py --path documents/ --dry-run

Exit codes:
    0 - All files processed successfully
    1 - Partial failure (some files failed)
    2 - Complete failure (config error / all files failed)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path so ``src.*`` imports work.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.core.settings import load_settings  # noqa: E402
from src.core.trace.trace_context import TraceContext  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline, PipelineResult  # noqa: E402
from src.observability.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Modular RAG knowledge hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--path", "-p", required=True,
        help="Path to file or directory to ingest (recursive for dirs).",
    )
    parser.add_argument(
        "--collection", "-c", default="default",
        help="Collection name for organising documents (default: 'default').",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-processing even if file was previously ingested.",
    )
    parser.add_argument(
        "--config", default=str(_REPO_ROOT / "config" / "settings.yaml"),
        help="Path to configuration file (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be processed without actually processing.",
    )

    return parser.parse_args()


# ── File discovery ──────────────────────────────────────────────────


def discover_files(
    path: str,
    extensions: List[str] | None = None,
) -> List[Path]:
    """Discover files to process from *path*.

    Args:
        path: File or directory path.
        extensions: Accepted extensions (default ``['.pdf']``).

    Returns:
        Sorted list of file paths.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a single file has an unsupported extension.
    """
    if extensions is None:
        extensions = [
            ".pdf", ".md", ".markdown",
            ".c", ".cpp", ".cxx", ".cc", ".h", ".hxx", ".py",
        ]

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        if p.suffix.lower() in extensions:
            return [p]
        raise ValueError(
            f"Unsupported file type: {p.suffix}. Supported: {extensions}",
        )

    # Directory — recursively find matching files (case-insensitive).
    files: set[Path] = set()
    for ext in extensions:
        files.update(p.rglob(f"*{ext}"))
        files.update(p.rglob(f"*{ext.upper()}"))

    return sorted(files)


# ── Summary ─────────────────────────────────────────────────────────


def print_summary(
    results: List[PipelineResult],
    verbose: bool = False,
) -> None:
    """Print a human-readable processing summary."""
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    total_chunks = sum(r.chunk_count for r in results if r.success)
    total_images = sum(r.image_count for r in results if r.success)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed:     {failed}")
    print(f"\nTotal chunks generated: {total_chunks}")
    print(f"Total images processed: {total_images}")

    if verbose and failed > 0:
        print("\nFailed files:")
        for r in results:
            if not r.success:
                print(f"  [FAIL] {r.file_path}: {r.error}")

    if verbose and successful > 0:
        print("\nSuccessful files:")
        for r in results:
            if r.success:
                skipped = r.stages.get("integrity", {}).get("skipped", False)
                status = "skipped" if skipped else f"{r.chunk_count} chunks"
                print(f"  [OK] {status}: {r.file_path}")

    print("=" * 60)


# ── Main ────────────────────────────────────────────────────────────


def main() -> int:
    """Entry point.  Returns exit code (0/1/2)."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("[*] Modular RAG Ingestion Script")
    print("=" * 60)

    # ── Load config ─────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[FAIL] Configuration file not found: {config_path}")
        return 2

    try:
        settings = load_settings(str(config_path))
        print(f"[OK] Configuration loaded from: {config_path}")
    except Exception as exc:
        print(f"[FAIL] Failed to load configuration: {exc}")
        return 2

    # ── Discover files ──────────────────────────────────────────────
    try:
        files = discover_files(args.path)
        print(f"[INFO] Found {len(files)} file(s) to process")

        if not files:
            print("[WARN] No files found to process")
            return 0

        for f in files:
            print(f"   - {f}")
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}")
        return 2
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    # ── Dry-run ─────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n[INFO] Dry run complete — {len(files)} file(s) would be processed")
        return 0

    # ── Initialise pipeline ─────────────────────────────────────────
    print(f"\n[INFO] Initializing pipeline...")
    print(f"   Collection: {args.collection}")
    print(f"   Force:      {args.force}")

    try:
        pipeline = IngestionPipeline(
            settings=settings,
            collection=args.collection,
            force=args.force,
        )
    except Exception as exc:
        print(f"[FAIL] Failed to initialize pipeline: {exc}")
        logger.exception("Pipeline initialization failed")
        return 2

    # ── Process files ───────────────────────────────────────────────
    print("\n[INFO] Processing files...")
    results: List[PipelineResult] = []

    try:
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing: {file_path}")

            try:
                trace = TraceContext(trace_type="ingestion")
                trace.metadata["source_path"] = str(file_path)
                result = pipeline.run(str(file_path), trace=trace)
                trace.finish()
                results.append(result)

                if result.success:
                    skipped = result.stages.get("integrity", {}).get(
                        "skipped", False,
                    )
                    if skipped:
                        print("   [SKIP] Skipped (already processed)")
                    else:
                        print(
                            f"   [OK] Success: {result.chunk_count} chunks, "
                            f"{result.image_count} images",
                        )
                else:
                    print(f"   [FAIL] Failed: {result.error}")

            except Exception as exc:
                logger.exception("Unexpected error processing %s", file_path)
                results.append(
                    PipelineResult(
                        success=False,
                        file_path=str(file_path),
                        error=str(exc),
                    ),
                )
                print(f"   [FAIL] Error: {exc}")
    finally:
        pipeline.close()

    # ── Summary & exit ──────────────────────────────────────────────
    print_summary(results, args.verbose)

    successful = sum(1 for r in results if r.success)
    if successful == len(results):
        return 0
    elif successful > 0:
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
