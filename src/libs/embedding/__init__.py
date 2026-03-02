"""
Embedding - Text embedding abstraction.

Components:
- BaseEmbedding: Abstract base class
- EmbeddingFactory: Provider routing factory
- Providers: OpenAI / Azure / Ollama
"""

from src.libs.embedding.azure_embedding import AzureEmbedding
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.ollama_embedding import OllamaEmbedding
from src.libs.embedding.openai_embedding import OpenAIEmbedding

__all__: list[str] = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "OpenAIEmbedding",
    "AzureEmbedding",
    "OllamaEmbedding",
]
