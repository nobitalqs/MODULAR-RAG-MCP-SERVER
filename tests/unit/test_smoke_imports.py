"""Smoke tests for package imports.

Verifies that all key packages can be imported successfully.
Serves as a basic sanity check for the project structure (DEV_SPEC 5.2).
"""

import pytest


@pytest.mark.unit
class TestSmokeImports:
    """Smoke tests to verify all key packages are importable."""

    # ── Top-level ──

    def test_import_src_package(self) -> None:
        import src
        assert src is not None

    # ── MCP Server Layer ──

    def test_import_mcp_server(self) -> None:
        from src import mcp_server
        assert mcp_server is not None

    def test_import_mcp_server_tools(self) -> None:
        from src.mcp_server import tools
        assert tools is not None

    # ── Core Layer ──

    def test_import_core(self) -> None:
        from src import core
        assert core is not None

    def test_import_core_query_engine(self) -> None:
        from src.core import query_engine
        assert query_engine is not None

    def test_import_core_response(self) -> None:
        from src.core import response
        assert response is not None

    def test_import_core_trace(self) -> None:
        from src.core import trace
        assert trace is not None

    # ── Ingestion Layer ──

    def test_import_ingestion(self) -> None:
        from src import ingestion
        assert ingestion is not None

    def test_import_ingestion_chunking(self) -> None:
        from src.ingestion import chunking
        assert chunking is not None

    def test_import_ingestion_transform(self) -> None:
        from src.ingestion import transform
        assert transform is not None

    def test_import_ingestion_embedding(self) -> None:
        from src.ingestion import embedding
        assert embedding is not None

    def test_import_ingestion_storage(self) -> None:
        from src.ingestion import storage
        assert storage is not None

    # ── Libs Layer ──

    def test_import_libs(self) -> None:
        from src import libs
        assert libs is not None

    def test_import_libs_llm(self) -> None:
        from src.libs import llm
        assert llm is not None

    def test_import_libs_embedding(self) -> None:
        from src.libs import embedding
        assert embedding is not None

    def test_import_libs_splitter(self) -> None:
        from src.libs import splitter
        assert splitter is not None

    def test_import_libs_vector_store(self) -> None:
        from src.libs import vector_store
        assert vector_store is not None

    def test_import_libs_reranker(self) -> None:
        from src.libs import reranker
        assert reranker is not None

    def test_import_libs_evaluator(self) -> None:
        from src.libs import evaluator
        assert evaluator is not None

    def test_import_libs_loader(self) -> None:
        from src.libs import loader
        assert loader is not None

    # ── Observability Layer ──

    def test_import_observability(self) -> None:
        from src import observability
        assert observability is not None

    def test_import_observability_dashboard(self) -> None:
        from src.observability import dashboard
        assert dashboard is not None

    def test_import_observability_dashboard_pages(self) -> None:
        from src.observability.dashboard import pages
        assert pages is not None

    def test_import_observability_dashboard_services(self) -> None:
        from src.observability.dashboard import services
        assert services is not None

    def test_import_observability_evaluation(self) -> None:
        from src.observability import evaluation
        assert evaluation is not None

    # ── Stub modules ──

    def test_import_core_settings(self) -> None:
        from src.core.settings import SettingsError
        assert SettingsError is not None

    def test_import_observability_logger(self) -> None:
        from src.observability.logger import get_logger
        assert callable(get_logger)
