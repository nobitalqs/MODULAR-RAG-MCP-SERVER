"""Tests for server transport modes."""

from __future__ import annotations

from src.mcp_server.server import create_http_app, parse_args


class TestParseArgs:
    def test_default_is_stdio(self):
        args = parse_args([])
        assert args.transport == "stdio"

    def test_http_transport(self):
        args = parse_args(["--transport", "http"])
        assert args.transport == "http"
        assert args.host == "127.0.0.1"
        assert args.port == 8000

    def test_custom_host_port(self):
        args = parse_args(["--transport", "http", "--host", "0.0.0.0", "--port", "9000"])
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_stdio_ignores_host_port(self):
        args = parse_args(["--transport", "stdio", "--port", "9999"])
        assert args.transport == "stdio"
        assert args.port == 9999  # parsed but not used


class TestCreateHTTPApp:
    def test_returns_starlette_app(self):
        app, host, port = create_http_app()
        from starlette.applications import Starlette

        assert isinstance(app, Starlette)
        assert host == "127.0.0.1"
        assert port == 8000

    def test_custom_host_port(self):
        _, host, port = create_http_app(host="0.0.0.0", port=9000)
        assert host == "0.0.0.0"
        assert port == 9000

    def test_app_has_mcp_route(self):
        app, _, _ = create_http_app(path="/mcp")
        route_paths = [r.path for r in app.routes]
        assert "/mcp" in route_paths
