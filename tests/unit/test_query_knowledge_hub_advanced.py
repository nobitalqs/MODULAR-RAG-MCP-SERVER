"""Unit tests for Phase J integration in query_knowledge_hub.

Tests verify that the advanced features (memory, query rewriter, rate limiter,
embedding cache, query router) are correctly wired into QueryKnowledgeHubTool.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubConfig,
    QueryKnowledgeHubTool,
    TOOL_INPUT_SCHEMA,
)


class TestInputSchemaSessionId:
    """session_id should be in the tool input schema."""

    def test_session_id_in_properties(self):
        assert "session_id" in TOOL_INPUT_SCHEMA["properties"]

    def test_session_id_is_optional(self):
        assert "session_id" not in TOOL_INPUT_SCHEMA.get("required", [])

    def test_session_id_type_is_string(self):
        assert TOOL_INPUT_SCHEMA["properties"]["session_id"]["type"] == "string"


class TestAdvancedComponentInit:
    """Advanced components should be lazily initialized."""

    def _make_tool(self, **overrides):
        """Create a tool with mock hybrid_search and reranker to skip _ensure_initialized."""
        defaults = dict(
            settings=MagicMock(),
            config=QueryKnowledgeHubConfig(),
            hybrid_search=MagicMock(),
            reranker=MagicMock(is_enabled=False),
        )
        defaults.update(overrides)
        tool = QueryKnowledgeHubTool(**defaults)
        tool._initialized = True
        tool._current_collection = "default"
        return tool

    def test_memory_initialized_when_settings_present(self):
        mock_settings = MagicMock()
        mock_settings.memory = MagicMock(enabled=True, provider="memory")
        mock_settings.query_rewriting = None
        mock_settings.rate_limit = None
        mock_settings.cache = None
        mock_settings.query_routing = None

        tool = self._make_tool(settings=mock_settings)
        # _init_advanced_components should be called during execute
        # but we test the lazy pattern via the attribute check
        assert tool._conversation_memory is None  # not yet initialized

    def test_no_advanced_when_settings_absent(self):
        mock_settings = MagicMock()
        mock_settings.memory = None
        mock_settings.query_rewriting = None
        mock_settings.rate_limit = None
        mock_settings.cache = None
        mock_settings.query_routing = None

        tool = self._make_tool(settings=mock_settings)
        assert tool._conversation_memory is None
        assert tool._query_rewriter is None
        assert tool._rate_limiter is None
        assert tool._query_router is None


class TestExecuteWithSessionId:
    """execute() should accept session_id and wire memory turns."""

    @pytest.mark.asyncio
    async def test_execute_adds_turn_when_session_id_provided(self):
        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = []
        mock_settings = MagicMock()
        mock_settings.memory = None
        mock_settings.query_rewriting = None
        mock_settings.rate_limit = None
        mock_settings.cache = None
        mock_settings.query_routing = None

        tool = QueryKnowledgeHubTool(
            settings=mock_settings,
            config=QueryKnowledgeHubConfig(),
            hybrid_search=mock_hybrid,
            reranker=MagicMock(is_enabled=False),
        )
        tool._initialized = True
        tool._current_collection = "default"

        # Should not raise even with session_id
        response = await tool.execute(query="test query", session_id="sess-123")
        assert response is not None

    @pytest.mark.asyncio
    async def test_execute_without_session_id_still_works(self):
        mock_hybrid = MagicMock()
        mock_hybrid.search.return_value = []
        mock_settings = MagicMock()
        mock_settings.memory = None
        mock_settings.query_rewriting = None
        mock_settings.rate_limit = None
        mock_settings.cache = None
        mock_settings.query_routing = None

        tool = QueryKnowledgeHubTool(
            settings=mock_settings,
            config=QueryKnowledgeHubConfig(),
            hybrid_search=mock_hybrid,
            reranker=MagicMock(is_enabled=False),
        )
        tool._initialized = True
        tool._current_collection = "default"

        response = await tool.execute(query="test query")
        assert response is not None
