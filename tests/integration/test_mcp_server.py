"""Integration tests for MCP server stdio entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import pytest


def send_and_receive(
    proc: subprocess.Popen,
    requests: List[Dict[str, Any]],
    timeout: float = 5.0,
    expected_responses: int = 0,
) -> List[str]:
    """Send requests to proc stdin and collect stdout lines.

    Args:
        proc: Subprocess with stdin/stdout pipes.
        requests: List of JSON-RPC requests/notifications to send.
        timeout: Max time to wait for responses.
        expected_responses: Number of responses to wait for (0 = auto-count).

    Returns:
        List of lines read from stdout.
    """
    assert proc.stdin is not None
    assert proc.stdout is not None

    # Send all requests
    for req in requests:
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()

    # Read stdout with timeout
    lines = []
    start = time.time()
    response_count = 0

    # Count expected responses (requests with 'id' field, excluding notifications)
    if expected_responses == 0:
        expected_responses = sum(1 for req in requests if "id" in req)

    while time.time() - start < timeout:
        # Check if we got enough responses
        if expected_responses > 0 and response_count >= expected_responses:
            break

        line = proc.stdout.readline()
        if not line:
            # Give a bit more time for slow responses
            time.sleep(0.1)
            continue
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
            # Count JSON-RPC responses (have 'id' and 'result' or 'error')
            try:
                data = json.loads(stripped)
                if "id" in data and ("result" in data or "error" in data):
                    response_count += 1
            except json.JSONDecodeError:
                pass

    return lines


def find_response(lines: List[str], request_id: int) -> Optional[Dict[str, Any]]:
    """Find JSON-RPC response with given id in lines."""
    for line in lines:
        if not line.startswith('{"jsonrpc"'):
            continue
        try:
            data = json.loads(line)
            if data.get("id") == request_id:
                return data
        except json.JSONDecodeError:
            continue
    return None


@pytest.mark.integration
def test_mcp_server_initialize_stdio() -> None:
    """Ensure initialize works and stdout is clean JSON-RPC output."""

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "clientInfo": {"name": "pytest", "version": "0.0.0"},
            "capabilities": {},
        },
    }

    try:
        lines = send_and_receive(proc, [request], timeout=5.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    assert len(lines) > 0, "No stdout lines received."

    response = find_response(lines, 1)
    assert response is not None, f"No initialize response found in: {lines}"

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    assert "serverInfo" in response["result"]
    assert "capabilities" in response["result"]


@pytest.mark.integration
def test_mcp_server_tools_list_stdio() -> None:
    """Ensure tools/list works and returns registered tools including query_knowledge_hub."""

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    requests = [
        # Initialize request
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "clientInfo": {"name": "pytest", "version": "0.0.0"},
                "capabilities": {},
            },
        },
        # Initialized notification (required by MCP protocol)
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        },
        # Tools list request
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        },
    ]

    try:
        lines = send_and_receive(proc, requests, timeout=10.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    assert len(lines) > 0, "No stdout lines received."

    # Verify initialize response
    init_response = find_response(lines, 1)
    assert init_response is not None, f"No initialize response found in: {lines}"
    assert "result" in init_response

    # Verify tools/list response
    tools_response = find_response(lines, 2)
    assert tools_response is not None, f"No tools/list response found in: {lines}"

    assert tools_response["jsonrpc"] == "2.0"
    assert tools_response["id"] == 2
    assert "result" in tools_response
    assert "tools" in tools_response["result"]
    # E3: query_knowledge_hub should be registered
    assert isinstance(tools_response["result"]["tools"], list)
    assert len(tools_response["result"]["tools"]) >= 1

    # Verify query_knowledge_hub is present
    tool_names = [t["name"] for t in tools_response["result"]["tools"]]
    assert "query_knowledge_hub" in tool_names
