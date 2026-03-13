"""MCP Tool: ingest_document

Ingests a document file into the knowledge hub via the IngestionPipeline.
Supports PDF, Markdown, and source code files.

Usage via MCP:
    Tool name: ingest_document
    Input schema:
        - file_path (string, required): Path to the file to ingest
        - collection (string, optional): Target collection (default: "default")
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from mcp import types

logger = logging.getLogger(__name__)

# Tool metadata
TOOL_NAME = "ingest_document"
TOOL_DESCRIPTION = """Ingest a document file into the knowledge hub.

Supports PDF, Markdown (.md/.markdown), and source code (.py, .c, .cpp, etc.).
The document is parsed, chunked, embedded, and stored for later retrieval.

Parameters:
- file_path: Absolute or relative path to the file to ingest
- collection: Target collection name (default: "default")
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Path to the document file to ingest.",
        },
        "collection": {
            "type": "string",
            "description": "Target collection name for the document.",
            "default": "default",
        },
    },
    "required": ["file_path"],
}


def _run_pipeline(file_path: str, collection: str) -> Any:
    """Run the ingestion pipeline synchronously.

    Separated for easy mocking in tests.

    Args:
        file_path: Resolved file path.
        collection: Target collection.

    Returns:
        PipelineResult from the pipeline.
    """
    from src.ingestion.pipeline import run_pipeline

    return run_pipeline(file_path=file_path, collection=collection)


async def ingest_document_handler(
    file_path: str,
    collection: str = "default",
) -> types.CallToolResult:
    """Handler function for MCP tool registration.

    Args:
        file_path: Path to the file to ingest.
        collection: Target collection name.

    Returns:
        MCP CallToolResult with success info or error.
    """
    # Validate file exists
    path = Path(file_path).resolve()
    if not path.exists():
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Error: File not found — {path}",
                )
            ],
            isError=True,
        )

    logger.info(
        "Ingesting document: %s into collection '%s'",
        path,
        collection,
    )

    try:
        result = await asyncio.to_thread(
            _run_pipeline,
            file_path=str(path),
            collection=collection,
        )

        if not result.success:
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Ingestion failed: {result.error}",
                    )
                ],
                isError=True,
            )

        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=(
                        f"Document ingested successfully.\n"
                        f"- doc_id: {result.doc_id}\n"
                        f"- chunks: {result.chunk_count}\n"
                        f"- collection: {collection}"
                    ),
                )
            ],
            isError=False,
        )

    except Exception as exc:
        logger.exception("ingest_document handler error: %s", exc)
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Ingestion error: {exc}",
                )
            ],
            isError=True,
        )


def register_tool(protocol_handler: Any) -> None:
    """Register ingest_document tool with the protocol handler.

    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=ingest_document_handler,
    )
    logger.info("Registered MCP tool: %s", TOOL_NAME)
