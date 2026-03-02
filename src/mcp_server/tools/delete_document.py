"""MCP Tool: delete_document

Two-phase document deletion with user confirmation:
- Phase 1 (confirm_delete_data=false): Returns preview of associated data
- Phase 2 (confirm_delete_data=true): Executes cascading deletion

Usage via MCP:
    Tool name: delete_document
    Input schema:
        - source_path (string, required): Document file path
        - collection (string, optional): Collection name (default: "default")
        - confirm_delete_data (boolean, optional): Execute deletion (default: false)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp import types

if TYPE_CHECKING:
    from src.ingestion.document_manager import DocumentManager
    from src.mcp_server.protocol_handler import ProtocolHandler

logger = logging.getLogger(__name__)


TOOL_NAME = "delete_document"
TOOL_DESCRIPTION = """Delete a document and its associated data from the RAG knowledge base.

First call without confirm_delete_data (or with confirm_delete_data=false) to preview
what will be deleted (chunk count, image count). Then call again with
confirm_delete_data=true to execute the deletion.

This removes all associated data: vector embeddings, BM25 index entries,
extracted images, and the ingestion history record.
"""

TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "source_path": {
            "type": "string",
            "description": "Path to the document to delete.",
        },
        "collection": {
            "type": "string",
            "description": "Collection the document belongs to.",
            "default": "default",
        },
        "confirm_delete_data": {
            "type": "boolean",
            "description": (
                "Set to true to execute deletion. "
                "When false (default), returns a preview of what will be deleted."
            ),
            "default": False,
        },
    },
    "required": ["source_path"],
}


class DeleteDocumentTool:
    """MCP Tool for deleting documents with two-phase confirmation.

    Phase 1 (preview): Returns associated data statistics.
    Phase 2 (execute): Cascading deletion across all storage backends.
    """

    def __init__(
        self,
        document_manager: DocumentManager | None = None,
    ) -> None:
        self._document_manager = document_manager

    @property
    def document_manager(self) -> DocumentManager:
        """Get or create DocumentManager."""
        if self._document_manager is None:
            self._document_manager = self._create_document_manager()
        return self._document_manager

    @staticmethod
    def _create_document_manager() -> DocumentManager:
        """Create DocumentManager from default settings."""
        from src.core.settings import load_settings
        from src.ingestion.document_manager import DocumentManager
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.ingestion.storage.image_storage import ImageStorage
        from src.libs.loader.file_integrity import SQLiteIntegrityChecker
        from src.libs.vector_store import ChromaStore, VectorStoreFactory

        settings = load_settings()
        factory = VectorStoreFactory()
        factory.register_provider("chroma", ChromaStore)
        vector_store = factory.create_from_settings(settings.vector_store)

        return DocumentManager(
            chroma_store=vector_store,
            bm25_indexer=BM25Indexer(index_dir="data/db/bm25"),
            image_storage=ImageStorage(
                db_path="data/db/image_index.db",
                images_root="data/images",
            ),
            file_integrity=SQLiteIntegrityChecker(
                db_path="data/db/file_integrity.db",
            ),
        )

    def _resolve_source_hash(
        self,
        source_path: str,
        collection: str,
    ) -> str | None:
        """Resolve the source_hash for a document.

        Tries: live file hash -> integrity DB lookup by path.
        """
        try:
            return self.document_manager.integrity.compute_sha256(source_path)
        except Exception:
            return self.document_manager.integrity.lookup_by_path(
                source_path,
                collection,
            )

    async def execute(
        self,
        source_path: str,
        collection: str = "default",
        confirm_delete_data: bool = False,
    ) -> types.CallToolResult:
        """Execute the delete_document tool.

        Args:
            source_path: Document file path.
            collection: Collection name.
            confirm_delete_data: If True, execute deletion. If False, preview.

        Returns:
            CallToolResult with JSON response.
        """
        logger.info(
            "delete_document: path=%s, collection=%s, confirm=%s",
            source_path,
            collection,
            confirm_delete_data,
        )

        source_hash = await asyncio.to_thread(
            self._resolve_source_hash,
            source_path,
            collection,
        )

        if source_hash is None:
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "status": "not_found",
                                "message": (
                                    f"Document not found: '{source_path}' "
                                    f"(collection: {collection}). "
                                    "File does not exist and no ingestion record found."
                                ),
                            }
                        ),
                    )
                ],
                isError=True,
            )

        detail = await asyncio.to_thread(
            self.document_manager.get_document_detail,
            source_hash,
        )

        doc_name = Path(source_path).name

        if not confirm_delete_data:
            # Phase 1: Preview
            chunk_count = detail.chunk_count if detail else 0
            image_count = detail.image_count if detail else 0

            response = {
                "status": "confirmation_required",
                "document": doc_name,
                "source_path": source_path,
                "collection": collection,
                "associated_data": {
                    "chunks": chunk_count,
                    "images": image_count,
                },
                "message": (
                    f"Document '{doc_name}' has {chunk_count} chunks "
                    f"and {image_count} images in the RAG system. "
                    "Call again with confirm_delete_data=true to "
                    "delete all associated data."
                ),
                "instructions": ("To proceed, call delete_document with confirm_delete_data=true"),
            }

            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=json.dumps(response),
                    )
                ],
                isError=False,
            )

        # Phase 2: Execute deletion
        del_result = await asyncio.to_thread(
            self.document_manager.delete_document,
            source_path,
            collection,
            source_hash,
        )

        if del_result.success:
            response = {
                "status": "deleted",
                "document": doc_name,
                "result": {
                    "chunks_deleted": del_result.chunks_deleted,
                    "bm25_removed": del_result.bm25_removed,
                    "images_deleted": del_result.images_deleted,
                    "integrity_removed": del_result.integrity_removed,
                },
            }
        else:
            response = {
                "status": "partial_failure",
                "document": doc_name,
                "result": {
                    "chunks_deleted": del_result.chunks_deleted,
                    "bm25_removed": del_result.bm25_removed,
                    "images_deleted": del_result.images_deleted,
                    "integrity_removed": del_result.integrity_removed,
                },
                "errors": del_result.errors,
            }

        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps(response),
                )
            ],
            isError=False,
        )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """Register the delete_document tool with the protocol handler.

    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    tool = DeleteDocumentTool()

    async def handler(
        source_path: str,
        collection: str = "default",
        confirm_delete_data: bool = False,
    ) -> types.CallToolResult:
        return await tool.execute(
            source_path=source_path,
            collection=collection,
            confirm_delete_data=confirm_delete_data,
        )

    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )

    logger.info("Registered MCP tool: %s", TOOL_NAME)
