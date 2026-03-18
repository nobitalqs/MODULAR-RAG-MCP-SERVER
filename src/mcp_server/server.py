"""MCP Server entry point using official MCP SDK.

Supports two transport modes:
- **stdio** (default): Client launches server as subprocess, communicates
  via stdin/stdout. Used by VS Code Copilot, Claude Desktop, etc.
- **streamable-http**: Server runs as HTTP service, clients connect via
  HTTP POST + SSE. Used for remote deployment, multi-user, Docker.

Usage:
    python main.py                          # stdio (default)
    python main.py --transport http         # streamable HTTP on :8000
    python main.py --transport http --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

from src.mcp_server.protocol_handler import create_mcp_server
from src.observability.logger import get_logger

if TYPE_CHECKING:
    pass


SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


def _redirect_all_loggers_to_stderr() -> None:
    """Redirect all root logger handlers to stderr.

    MCP stdio transport reserves stdout for JSON-RPC messages.
    Any logging to stdout corrupts the protocol stream.
    """
    import logging as _logging

    root = _logging.getLogger()
    stderr_handler = _logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    for handler in root.handlers[:]:
        if isinstance(handler, _logging.StreamHandler) and not isinstance(
            handler, _logging.FileHandler
        ):
            root.removeHandler(handler)
    root.addHandler(stderr_handler)


# ── Stdio Transport ─────────────────────────────────────────────


async def run_stdio_server_async() -> int:
    """Run MCP server over stdio asynchronously.

    Returns:
        Exit code.
    """
    import mcp.server.stdio

    _redirect_all_loggers_to_stderr()

    logger = get_logger(log_level="INFO")
    logger.info("Starting MCP server (stdio transport) with official SDK.")

    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    logger.info("MCP server shutting down.")
    return 0


def run_stdio_server() -> int:
    """Run MCP server over stdio (synchronous wrapper)."""
    return asyncio.run(run_stdio_server_async())


# ── Streamable HTTP Transport ────────────────────────────────────


def create_http_app(
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
) -> tuple:
    """Create Starlette ASGI app with StreamableHTTP transport.

    Args:
        host: Bind address.
        port: Bind port.
        path: URL path for MCP endpoint.

    Returns:
        Tuple of (app, host, port) for uvicorn.
    """
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    session_manager = StreamableHTTPSessionManager(app=server)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    app = Starlette(
        routes=[Mount(path, app=session_manager.handle_request)],
        lifespan=lifespan,
    )

    return app, host, port


def run_http_server(host: str = "127.0.0.1", port: int = 8000) -> int:
    """Run MCP server over Streamable HTTP.

    Args:
        host: Bind address.
        port: Bind port.

    Returns:
        Exit code.
    """
    import uvicorn

    logger = get_logger(log_level="INFO")
    logger.info(
        "Starting MCP server (streamable HTTP) on %s:%d/mcp",
        host,
        port,
    )

    app, _, _ = create_http_app(host=host, port=port)
    uvicorn.run(app, host=host, port=port)
    return 0


# ── CLI Entry Point ──────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Modular RAG MCP Server",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode: stdio (default) or http (streamable HTTP)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP bind port (default: 8000)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for MCP server."""
    args = parse_args(argv)

    if args.transport == "http":
        return run_http_server(host=args.host, port=args.port)
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())
