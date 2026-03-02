"""
MCP Server Tools.

This package contains the MCP tool definitions exposed to clients.
"""

from src.mcp_server.tools.query_knowledge_hub import (
    TOOL_NAME as QUERY_KNOWLEDGE_HUB_NAME,
    TOOL_DESCRIPTION as QUERY_KNOWLEDGE_HUB_DESCRIPTION,
    TOOL_INPUT_SCHEMA as QUERY_KNOWLEDGE_HUB_SCHEMA,
    QueryKnowledgeHubTool,
    query_knowledge_hub_handler,
    register_tool as register_query_knowledge_hub,
)

from src.mcp_server.tools.list_collections import (
    TOOL_NAME as LIST_COLLECTIONS_NAME,
    TOOL_DESCRIPTION as LIST_COLLECTIONS_DESCRIPTION,
    TOOL_INPUT_SCHEMA as LIST_COLLECTIONS_SCHEMA,
    ListCollectionsTool,
    CollectionInfo,
    register_tool as register_list_collections,
)

from src.mcp_server.tools.get_document_summary import (
    TOOL_NAME as GET_DOCUMENT_SUMMARY_NAME,
    TOOL_DESCRIPTION as GET_DOCUMENT_SUMMARY_DESCRIPTION,
    TOOL_INPUT_SCHEMA as GET_DOCUMENT_SUMMARY_SCHEMA,
    GetDocumentSummaryTool,
    DocumentSummary,
    DocumentNotFoundError,
    register_tool as register_get_document_summary,
)

__all__ = [
    "QUERY_KNOWLEDGE_HUB_NAME",
    "QUERY_KNOWLEDGE_HUB_DESCRIPTION",
    "QUERY_KNOWLEDGE_HUB_SCHEMA",
    "QueryKnowledgeHubTool",
    "query_knowledge_hub_handler",
    "register_query_knowledge_hub",
    "LIST_COLLECTIONS_NAME",
    "LIST_COLLECTIONS_DESCRIPTION",
    "LIST_COLLECTIONS_SCHEMA",
    "ListCollectionsTool",
    "CollectionInfo",
    "register_list_collections",
    "GET_DOCUMENT_SUMMARY_NAME",
    "GET_DOCUMENT_SUMMARY_DESCRIPTION",
    "GET_DOCUMENT_SUMMARY_SCHEMA",
    "GetDocumentSummaryTool",
    "DocumentSummary",
    "DocumentNotFoundError",
    "register_get_document_summary",
]
