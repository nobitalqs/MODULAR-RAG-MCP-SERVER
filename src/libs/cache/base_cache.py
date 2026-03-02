"""Abstract base class for cache providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseCache(ABC):
    """Pluggable cache interface.

    All cache providers must implement get/set/delete/exists.
    Values can be any picklable Python object.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None if missing or expired."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value. ttl overrides default_ttl if provided."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed, False otherwise."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
