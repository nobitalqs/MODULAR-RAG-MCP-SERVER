"""
Modular RAG MCP Server - Main Entry Point

Usage:
    python main.py                          # stdio transport (default)
    python main.py --transport http         # streamable HTTP on :8000
    python main.py --transport http --port 9000
"""

import sys

from src.mcp_server.server import main

if __name__ == "__main__":
    sys.exit(main())
