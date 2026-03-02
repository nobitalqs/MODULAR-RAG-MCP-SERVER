"""
Modular RAG MCP Server - Main Entry Point

This is the entry point for the MCP Server. It delegates to the
stdio transport server which handles MCP protocol communication.
"""

import sys

from src.mcp_server.server import run_stdio_server


def main() -> int:
    """Main entry point for the MCP Server.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())
