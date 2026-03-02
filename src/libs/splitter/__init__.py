"""
Splitter - Text splitting abstraction.

Components:
- BaseSplitter: Abstract base class
- SplitterFactory: Strategy routing factory
- RecursiveSplitter: LangChain RecursiveCharacterTextSplitter wrapper
"""

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

__all__: list[str] = [
    "BaseSplitter",
    "RecursiveSplitter",
    "SplitterFactory",
]
