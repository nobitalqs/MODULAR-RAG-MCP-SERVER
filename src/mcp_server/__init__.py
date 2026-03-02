"""
MCP Server Layer - Interface layer for MCP protocol.

This package contains the MCP Server implementation that exposes
tools via JSON-RPC 2.0 over Stdio Transport.
"""

from src.mcp_server.protocol_handler import (
    JSONRPCErrorCodes,
    ProtocolHandler,
    ToolDefinition,
    create_mcp_server,
    get_protocol_handler,
)
from src.mcp_server.server import SERVER_NAME, SERVER_VERSION

__all__ = [
    "JSONRPCErrorCodes",
    "ProtocolHandler",
    "ToolDefinition",
    "create_mcp_server",
    "get_protocol_handler",
    "SERVER_NAME",
    "SERVER_VERSION",
]
