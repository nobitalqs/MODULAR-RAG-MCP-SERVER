"""
Response Building - Citation generation and multimodal assembly.

Components:
- ResponseBuilder: Markdown response formatting
- CitationGenerator: Source citation generation
- MultimodalAssembler: Text + Image content assembly (E6)
"""

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.response_builder import MCPToolResponse, ResponseBuilder

__all__ = [
    "Citation",
    "CitationGenerator",
    "MCPToolResponse",
    "ResponseBuilder",
]
