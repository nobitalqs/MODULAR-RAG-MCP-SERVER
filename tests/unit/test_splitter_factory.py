"""Tests for Splitter abstraction: BaseSplitter, SplitterFactory."""

from __future__ import annotations

from typing import Any

import pytest

from src.libs.splitter.base_splitter import BaseSplitter


# ---------------------------------------------------------------------------
# Fake provider for testing
# ---------------------------------------------------------------------------

class FakeSplitter(BaseSplitter):
    """Minimal Splitter stub that splits by whitespace."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs
        self.call_count = 0

    def split_text(
        self,
        text: str,
        trace: Any = None,
        **kwargs: Any,
    ) -> list[str]:
        self.validate_text(text)
        self.call_count += 1
        chunks = [w for w in text.split() if w]
        self.validate_chunks(chunks)
        return chunks


# ===========================================================================
# BaseSplitter.validate_text
# ===========================================================================

class TestBaseSplitterValidateText:
    """Tests for BaseSplitter.validate_text."""

    def setup_method(self) -> None:
        self.sp = FakeSplitter()

    def test_valid_text(self) -> None:
        self.sp.validate_text("hello world")  # should not raise

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            self.sp.validate_text("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            self.sp.validate_text("   ")

    def test_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            self.sp.validate_text(123)  # type: ignore[arg-type]


# ===========================================================================
# BaseSplitter.validate_chunks
# ===========================================================================

class TestBaseSplitterValidateChunks:
    """Tests for BaseSplitter.validate_chunks."""

    def setup_method(self) -> None:
        self.sp = FakeSplitter()

    def test_valid_chunks(self) -> None:
        self.sp.validate_chunks(["a", "b", "c"])  # should not raise

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            self.sp.validate_chunks([])

    def test_non_list_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a list"):
            self.sp.validate_chunks("not a list")  # type: ignore[arg-type]

    def test_non_string_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="not a string"):
            self.sp.validate_chunks(["ok", 42])  # type: ignore[list-item]

    def test_empty_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="empty or whitespace"):
            self.sp.validate_chunks(["ok", "   "])


# ===========================================================================
# FakeSplitter behavior
# ===========================================================================

class TestFakeSplitter:
    """Tests for FakeSplitter as a BaseSplitter implementation."""

    def test_split_basic(self) -> None:
        sp = FakeSplitter()
        assert sp.split_text("hello world") == ["hello", "world"]

    def test_split_validates_input(self) -> None:
        sp = FakeSplitter()
        with pytest.raises(ValueError, match="empty or whitespace"):
            sp.split_text("   ")

    def test_split_increments_call_count(self) -> None:
        sp = FakeSplitter()
        assert sp.call_count == 0
        sp.split_text("a b")
        assert sp.call_count == 1
        sp.split_text("c d e")
        assert sp.call_count == 2

    def test_split_multiple_spaces(self) -> None:
        sp = FakeSplitter()
        assert sp.split_text("a   b   c") == ["a", "b", "c"]


# ===========================================================================
# SplitterFactory
# ===========================================================================

class TestSplitterFactory:
    """Tests for the Splitter factory routing logic."""

    def setup_method(self) -> None:
        from src.libs.splitter.splitter_factory import SplitterFactory

        self.factory = SplitterFactory()

    def test_register_and_create(self) -> None:
        self.factory.register_provider("fake", FakeSplitter)
        sp = self.factory.create("fake")
        assert isinstance(sp, FakeSplitter)

    def test_case_insensitive_registration(self) -> None:
        self.factory.register_provider("Recursive", FakeSplitter)
        sp = self.factory.create("recursive")
        assert isinstance(sp, FakeSplitter)

    def test_case_insensitive_create(self) -> None:
        self.factory.register_provider("fake", FakeSplitter)
        sp = self.factory.create("FAKE")
        assert isinstance(sp, FakeSplitter)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Splitter provider"):
            self.factory.create("nonexistent")

    def test_error_lists_available_providers(self) -> None:
        self.factory.register_provider("fake", FakeSplitter)
        with pytest.raises(ValueError, match="fake"):
            self.factory.create("missing")

    def test_register_non_subclass_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a subclass"):
            self.factory.register_provider("bad", dict)  # type: ignore[arg-type]

    def test_list_providers(self) -> None:
        self.factory.register_provider("semantic", FakeSplitter)
        self.factory.register_provider("fixed", FakeSplitter)
        self.factory.register_provider("recursive", FakeSplitter)
        assert self.factory.list_providers() == ["fixed", "recursive", "semantic"]

    def test_create_with_kwargs(self) -> None:
        self.factory.register_provider("fake", FakeSplitter)
        sp = self.factory.create("fake", chunk_size=500, chunk_overlap=50)
        assert sp.config["chunk_size"] == 500
        assert sp.config["chunk_overlap"] == 50

    def test_create_from_settings(self) -> None:
        from src.core.settings import IngestionSettings

        self.factory.register_provider("fake", FakeSplitter)

        settings = IngestionSettings(
            chunk_size=1000,
            chunk_overlap=200,
            splitter="fake",
        )
        sp = self.factory.create_from_settings(settings)
        assert isinstance(sp, FakeSplitter)

    def test_create_from_settings_forwards_fields(self) -> None:
        """create_from_settings should forward non-None fields as kwargs."""
        from src.core.settings import IngestionSettings

        self.factory.register_provider("fake", FakeSplitter)

        settings = IngestionSettings(
            chunk_size=500,
            chunk_overlap=100,
            splitter="recursive",
            batch_size=50,
        )
        # The splitter name "recursive" routes, but we registered "fake" for it
        self.factory.register_provider("recursive", FakeSplitter)
        sp = self.factory.create_from_settings(settings)
        assert isinstance(sp, FakeSplitter)
        assert sp.config["chunk_size"] == 500
        assert sp.config["chunk_overlap"] == 100
        assert sp.config["batch_size"] == 50

    def test_empty_registry_lists_none(self) -> None:
        with pytest.raises(ValueError, match="\\(none\\)"):
            self.factory.create("anything")
