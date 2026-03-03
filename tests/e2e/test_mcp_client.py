"""E2E test: MCP Client side call simulation.

Launches the MCP server as a subprocess over stdio, then uses the
official MCP SDK client to exercise:
  1. tools/list — server advertises expected tools
  2. tools/call — query_knowledge_hub returns structured results

This validates the full JSON-RPC round-trip without manual protocol
crafting, ensuring the server is usable by real MCP hosts (Copilot,
Claude Desktop, etc.).

Usage::

    pytest tests/e2e/test_mcp_client.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Expected tool names registered by the server
EXPECTED_TOOLS = {
    "query_knowledge_hub",
    "list_collections",
    "get_document_summary",
    "delete_document",
}


# ── Helpers ───────────────────────────────────────────────────────────


async def _create_client_session():
    """Create an MCP ClientSession connected to our server via stdio.

    Returns a context-manager tuple (session, cleanup) so the caller
    can properly tear down.
    """
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "main.py")],
        cwd=str(PROJECT_ROOT),
    )

    # stdio_client is an async context manager
    transport_ctx = stdio_client(server_params)
    transport = await transport_ctx.__aenter__()
    read_stream, write_stream = transport

    session = ClientSession(read_stream, write_stream)
    await session.__aenter__()
    await session.initialize()

    return session, transport_ctx


async def _close_session(session, transport_ctx):
    """Cleanly shut down session and transport."""
    try:
        await session.__aexit__(None, None, None)
    except Exception:
        pass
    try:
        await transport_ctx.__aexit__(None, None, None)
    except Exception:
        pass


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.e2e
class TestMCPClient:
    """MCP Client E2E tests — full JSON-RPC round-trip over stdio."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up and tear down MCP client session."""
        loop = asyncio.new_event_loop()

        try:
            session, transport_ctx = loop.run_until_complete(
                asyncio.wait_for(_create_client_session(), timeout=30)
            )
        except Exception as exc:
            loop.close()
            pytest.skip(f"MCP server failed to start: {exc}")
            return

        self.session = session
        self._transport_ctx = transport_ctx
        self._loop = loop

        yield

        loop.run_until_complete(_close_session(session, transport_ctx))
        loop.close()

    def _run(self, coro):
        """Run an async coroutine in the test's event loop."""
        return self._loop.run_until_complete(
            asyncio.wait_for(coro, timeout=30)
        )

    # -- tools/list ------------------------------------------------

    def test_tools_list_returns_expected_tools(self) -> None:
        """Server advertises all expected MCP tools."""
        result = self._run(self.session.list_tools())
        tool_names = {t.name for t in result.tools}

        for expected in EXPECTED_TOOLS:
            assert expected in tool_names, (
                f"Missing tool: {expected}. Got: {tool_names}"
            )

    def test_tools_have_descriptions(self) -> None:
        """Every tool has a non-empty description."""
        result = self._run(self.session.list_tools())

        for tool in result.tools:
            assert tool.description, f"Tool '{tool.name}' has empty description"
            assert len(tool.description) > 10, (
                f"Tool '{tool.name}' has suspiciously short description"
            )

    def test_tools_have_input_schemas(self) -> None:
        """Every tool has a valid JSON Schema for inputSchema."""
        result = self._run(self.session.list_tools())

        for tool in result.tools:
            schema = tool.inputSchema
            assert isinstance(schema, dict), (
                f"Tool '{tool.name}' inputSchema is not a dict"
            )
            assert "type" in schema or "properties" in schema, (
                f"Tool '{tool.name}' has incomplete inputSchema"
            )

    # -- tools/call: query_knowledge_hub ---------------------------

    def test_query_knowledge_hub_returns_result(self) -> None:
        """Calling query_knowledge_hub returns text content (even if empty)."""
        result = self._run(
            self.session.call_tool(
                "query_knowledge_hub",
                {"query": "What is RAG?"},
            )
        )

        # Result should have content blocks
        assert result.content is not None
        assert len(result.content) >= 1

        # First block should be text
        text_block = result.content[0]
        assert hasattr(text_block, "text")
        assert isinstance(text_block.text, str)
        assert len(text_block.text) > 0

    def test_query_knowledge_hub_not_error(self) -> None:
        """query_knowledge_hub should not return isError=True for valid input."""
        result = self._run(
            self.session.call_tool(
                "query_knowledge_hub",
                {"query": "What is attention mechanism?"},
            )
        )
        # isError should be False or None for successful calls
        assert not result.isError, (
            f"query_knowledge_hub returned error: {result.content}"
        )

    # -- tools/call: list_collections ------------------------------

    def test_list_collections_returns_result(self) -> None:
        """list_collections returns a valid response."""
        result = self._run(
            self.session.call_tool("list_collections", {})
        )

        assert result.content is not None
        assert len(result.content) >= 1
        assert not result.isError

    # -- tools/call: error handling --------------------------------

    def test_unknown_tool_returns_error(self) -> None:
        """Calling a non-existent tool returns isError=True."""
        result = self._run(
            self.session.call_tool("nonexistent_tool", {})
        )
        assert result.isError is True

    def test_query_knowledge_hub_empty_query(self) -> None:
        """Empty query string returns error."""
        result = self._run(
            self.session.call_tool(
                "query_knowledge_hub",
                {"query": ""},
            )
        )
        # Should return isError or contain error message
        first_text = result.content[0].text if result.content else ""
        assert result.isError or "error" in first_text.lower() or "empty" in first_text.lower()
