"""
LLM - Large Language Model abstraction.

Components:
- BaseLLM: Abstract base class
- LLMFactory: Provider routing factory
- Providers: Azure / OpenAI / Ollama / DeepSeek
- BaseVisionLLM: Vision-capable LLM base
- AzureVisionLLM: Azure GPT-4o vision implementation
- OpenAIVisionLLM: OpenAI GPT-4o vision implementation
"""

from src.libs.llm.azure_llm import AzureLLM
from src.libs.llm.azure_vision_llm import AzureVisionLLM
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.ollama_llm import OllamaLLM
from src.libs.llm.openai_llm import OpenAILLM
from src.libs.llm.openai_vision_llm import OpenAIVisionLLM

__all__: list[str] = [
    "AzureLLM",
    "AzureVisionLLM",
    "BaseLLM",
    "BaseVisionLLM",
    "ChatResponse",
    "DeepSeekLLM",
    "ImageInput",
    "LLMFactory",
    "Message",
    "OllamaLLM",
    "OpenAILLM",
    "OpenAIVisionLLM",
]
