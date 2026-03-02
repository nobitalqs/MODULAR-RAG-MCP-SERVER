"""
Shared pytest fixtures and configuration.

Adds project root to sys.path so that 'src' package is importable.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def config_dir() -> Path:
    """Return the config directory path."""
    return PROJECT_ROOT / "config"


@pytest.fixture
def sample_documents_dir() -> Path:
    """Return the sample documents directory path."""
    return PROJECT_ROOT / "tests" / "fixtures" / "sample_documents"
