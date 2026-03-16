"""Tests for SourceCodeLoader — language detection, raw text loading, Document contract."""

from __future__ import annotations

import pytest

from src.core.types import Document
from src.libs.loader.source_code_loader import SourceCodeLoader


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def loader():
    return SourceCodeLoader()


@pytest.fixture()
def python_file(tmp_path):
    content = """\
#!/usr/bin/env python
\"\"\"Example module.\"\"\"

def hello():
    print("Hello, world!")

if __name__ == "__main__":
    hello()
"""
    path = tmp_path / "hello.py"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def cpp_file(tmp_path):
    content = """\
#include <iostream>

int main() {
    std::cout << "Hello" << std::endl;
    return 0;
}
"""
    path = tmp_path / "main.cpp"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def header_file(tmp_path):
    content = """\
#pragma once
class Foo {
public:
    void bar();
};
"""
    path = tmp_path / "foo.h"
    path.write_text(content, encoding="utf-8")
    return path


# ── Tests ──────────────────────────────────────────────────────────


class TestLanguageDetection:
    """Language is correctly identified from extension."""

    def test_python_detected(self, loader, python_file):
        doc = loader.load(python_file)
        assert doc.metadata["language"] == "Python"

    def test_cpp_detected(self, loader, cpp_file):
        doc = loader.load(cpp_file)
        assert doc.metadata["language"] == "C++"

    def test_header_detected(self, loader, header_file):
        doc = loader.load(header_file)
        assert doc.metadata["language"] == "C++"

    @pytest.mark.parametrize(
        "ext,expected_lang",
        [
            (".c", "C++"),
            (".cpp", "C++"),
            (".cxx", "C++"),
            (".cc", "C++"),
            (".h", "C++"),
            (".hxx", "C++"),
            (".py", "Python"),
        ],
    )
    def test_all_extensions(self, loader, tmp_path, ext, expected_lang):
        path = tmp_path / f"test{ext}"
        path.write_text("code content", encoding="utf-8")
        doc = loader.load(path)
        assert doc.metadata["language"] == expected_lang


class TestDocumentContract:
    """Required metadata fields and Document structure."""

    def test_source_path(self, loader, python_file):
        doc = loader.load(python_file)
        assert "source_path" in doc.metadata
        assert str(python_file) in doc.metadata["source_path"]

    def test_doc_type(self, loader, python_file):
        doc = loader.load(python_file)
        assert doc.metadata["doc_type"] == "source_code"

    def test_doc_hash(self, loader, python_file):
        doc = loader.load(python_file)
        assert "doc_hash" in doc.metadata
        assert len(doc.metadata["doc_hash"]) == 64

    def test_doc_id_format(self, loader, python_file):
        doc = loader.load(python_file)
        assert doc.id.startswith("doc_")
        assert len(doc.id) == 20

    def test_filename(self, loader, python_file):
        doc = loader.load(python_file)
        assert doc.metadata["filename"] == "hello.py"

    def test_line_count(self, loader, python_file):
        doc = loader.load(python_file)
        assert doc.metadata["line_count"] == 8

    def test_text_content(self, loader, python_file):
        doc = loader.load(python_file)
        assert 'def hello():' in doc.text


class TestValidation:
    """Extension and file validation."""

    def test_unsupported_extension(self, loader, tmp_path):
        path = tmp_path / "file.rs"
        path.write_text("fn main() {}")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(path)

    def test_file_not_found(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "missing.py")


class TestKwargs:
    """LoaderFactory compatibility."""

    def test_accepts_kwargs(self):
        loader = SourceCodeLoader(extract_images=True, some_option="value")
        assert isinstance(loader, SourceCodeLoader)


class TestBriefExtraction:
    """Extract file-level description from header comments."""

    def test_python_brief_from_docstring(self, loader, tmp_path):
        content = '"""Dimuon invariant mass analysis using NanoAOD."""\n\nimport ROOT\n'
        path = tmp_path / "analysis.py"
        path.write_text(content)
        doc = loader.load(path)
        assert doc.metadata["brief"] == "Dimuon invariant mass analysis using NanoAOD."

    def test_python_brief_from_hash_comments(self, loader, tmp_path):
        content = (
            "## \\file\n"
            "## \\ingroup tutorial_fit\n"
            "## \\brief Fitting a Gaussian to histogram data\n"
            "##\n"
            "## \\macro_code\n"
            "import ROOT\n"
        )
        path = tmp_path / "fit.py"
        path.write_text(content)
        doc = loader.load(path)
        assert "Fitting a Gaussian to histogram data" in doc.metadata["brief"]

    def test_cpp_brief_from_triple_slash(self, loader, tmp_path):
        content = (
            "/// \\file\n"
            "/// \\ingroup tutorial_hist\n"
            "/// \\brief Draw a histogram with random Gaussian values\n"
            "///\n"
            "/// \\macro_image\n"
            "#include <TH1F.h>\n"
        )
        path = tmp_path / "hist.C"
        path.write_text(content)
        doc = loader.load(path)
        assert "Draw a histogram with random Gaussian values" in doc.metadata["brief"]

    def test_cpp_brief_from_block_comment(self, loader, tmp_path):
        content = (
            "/*\n"
            " * Compute dimuon invariant mass from NanoAOD\n"
            " */\n"
            "#include <ROOT/RDataFrame.hxx>\n"
        )
        path = tmp_path / "dimuon.C"
        path.write_text(content)
        doc = loader.load(path)
        assert "Compute dimuon invariant mass from NanoAOD" in doc.metadata["brief"]

    def test_no_brief_returns_empty(self, loader, tmp_path):
        content = "import ROOT\nh = ROOT.TH1F('h','',100,-5,5)\n"
        path = tmp_path / "bare.py"
        path.write_text(content)
        doc = loader.load(path)
        assert doc.metadata["brief"] == ""

    def test_python_hash_comment_header(self, loader, tmp_path):
        content = (
            "# This script analyzes Z boson decay\n"
            "# using CMS open data\n"
            "import ROOT\n"
        )
        path = tmp_path / "z_boson.py"
        path.write_text(content)
        doc = loader.load(path)
        assert "analyzes Z boson decay" in doc.metadata["brief"]
