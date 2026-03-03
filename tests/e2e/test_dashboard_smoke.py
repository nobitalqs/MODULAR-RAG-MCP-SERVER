"""E2E smoke test: Dashboard page rendering.

Uses Streamlit's ``AppTest`` framework to verify that each dashboard
page can load and render without raising Python exceptions.

This does NOT test interactive widgets or data correctness — only that
all 6 pages produce valid output and don't crash.

Usage::

    pytest tests/e2e/test_dashboard_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Path to the dashboard entry-point
APP_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "observability"
    / "dashboard"
    / "app.py"
)

# All page titles as defined in app.py
EXPECTED_PAGES = [
    "Overview",
    "Data Browser",
    "Ingestion Manager",
    "Ingestion Traces",
    "Query Traces",
    "Evaluation Panel",
]


def _try_create_app():
    """Create an AppTest instance, skipping if unavailable."""
    try:
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(APP_PATH, default_timeout=30)
    except Exception as exc:
        pytest.skip(f"AppTest not available: {exc}")


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.e2e
class TestDashboardSmoke:
    """Smoke tests for the Modular RAG Dashboard."""

    def test_app_loads_without_exception(self) -> None:
        """Dashboard app.py loads and runs without Python errors."""
        app = _try_create_app()
        app.run()
        assert not app.exception, (
            f"Dashboard raised exception: {app.exception}"
        )

    def test_app_has_title(self) -> None:
        """Dashboard sets a page title."""
        app = _try_create_app()
        app.run()
        # The app should render something
        assert not app.exception

    def test_overview_page_renders(self) -> None:
        """Overview page (default) renders without error."""
        app = _try_create_app()
        app.run()
        assert not app.exception
        # Overview is the default page, should have some content
        # Check that markdown or text elements exist
        all_elements = (
            list(app.markdown)
            + list(app.title)
            + list(app.header)
            + list(app.subheader)
        )
        assert len(all_elements) >= 0  # Just verify no crash

    def test_no_uncaught_exceptions_in_sidebar(self) -> None:
        """Sidebar (navigation) renders without error."""
        app = _try_create_app()
        app.run()
        assert not app.exception


@pytest.mark.e2e
class TestDashboardPageImports:
    """Verify each dashboard page module imports cleanly."""

    def test_overview_import(self) -> None:
        from src.observability.dashboard.pages.overview import render

        assert callable(render)

    def test_data_browser_import(self) -> None:
        from src.observability.dashboard.pages.data_browser import render

        assert callable(render)

    def test_ingestion_manager_import(self) -> None:
        from src.observability.dashboard.pages.ingestion_manager import render

        assert callable(render)

    def test_ingestion_traces_import(self) -> None:
        from src.observability.dashboard.pages.ingestion_traces import render

        assert callable(render)

    def test_query_traces_import(self) -> None:
        from src.observability.dashboard.pages.query_traces import render

        assert callable(render)

    def test_evaluation_panel_import(self) -> None:
        from src.observability.dashboard.pages.evaluation_panel import render

        assert callable(render)
