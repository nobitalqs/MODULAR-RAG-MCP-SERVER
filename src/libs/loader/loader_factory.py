"""LoaderFactory — extension-based routing with registry pattern.

Maps file extensions to loader classes, creating the appropriate loader
for each file type.  Follows the same Registry + Factory pattern used
throughout the project (EmbeddingFactory, LLMFactory, etc.) but keys
on file extension instead of a provider name string.

Usage:
    factory = LoaderFactory()
    factory.register_provider(".pdf", PdfLoader)
    factory.register_provider(".md", MarkdownLoader)
    loader = factory.create_for_file("report.pdf", extract_images=True)
    doc = loader.load("report.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class LoaderFactory:
    """Extension-based loader registry and factory.

    Attributes:
        _registry: Maps lowercase dotted extension (e.g. ".pdf") to loader class.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseLoader]] = {}

    def register_provider(self, ext: str, cls: type[BaseLoader]) -> None:
        """Register a loader class for a file extension.

        Args:
            ext: Dotted extension, e.g. ".pdf", ".md".
            cls: BaseLoader subclass to instantiate for this extension.
        """
        key = ext.lower()
        self._registry[key] = cls
        logger.debug("Registered loader for %s: %s", key, cls.__name__)

    def create_for_file(self, file_path: str | Path, **kwargs) -> BaseLoader:
        """Create a loader instance for the given file.

        Looks up the file's extension in the registry and instantiates the
        corresponding loader class, forwarding all *kwargs* to its constructor.

        Args:
            file_path: Path to the file (need not exist yet).
            **kwargs: Passed through to the loader's ``__init__``.

        Returns:
            An instance of the registered BaseLoader subclass.

        Raises:
            ValueError: If the extension is not registered.
        """
        suffix = Path(file_path).suffix.lower()
        cls = self._registry.get(suffix)
        if cls is None:
            supported = ", ".join(sorted(self._registry))
            raise ValueError(
                f"No loader registered for extension '{suffix}'. "
                f"Supported: {supported}"
            )
        return cls(**kwargs)
