"""
PDF 文本提取测试 —— 使用 PyMuPDF 从内存创建含已知内容的假 PDF。

覆盖 extract_pdf() 和 count_pages() 两个函数。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from paper_review.extractor import count_pages, extract_pdf

# 用 fitz（PyMuPDF）在内存中生成测试 PDF
pytest.importorskip("fitz")


def _make_pdf_text(text: str) -> bytes:
    """Create a PDF containing the given text, return bytes.

    Places text at y=150 (below the 50pt HEADER_FOOTER_MARGIN threshold)
    so the extractor doesn't filter it out as a header.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 150), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_multi_page_pdf(pages_content: list[str]) -> bytes:
    """Create a multi-page PDF, return bytes."""
    import fitz

    doc = fitz.open()
    for text in pages_content:
        page = doc.new_page()
        page.insert_text((50, 150), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _write_pdf(pdf_bytes: bytes) -> str:
    """Write PDF bytes to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        return f.name


class TestExtractPdf:
    """extract_pdf() 核心功能"""

    def test_extract_simple_text(self):
        """提取简单文本内容（使用 ASCII 文本，CJK 在 PDF 内联渲染不稳定）。"""
        text = "Abstract: This paper proposes a new method."
        pdf_bytes = _make_pdf_text(text)
        path = _write_pdf(pdf_bytes)
        try:
            result = extract_pdf(path)
            assert "Abstract" in result
            assert "new method" in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_extract_multi_page(self):
        """多页 PDF 提取，内容按页码拼接。"""
        pages = ["Page One Content", "Page Two Content", "Page Three Content"]
        pdf_bytes = _make_multi_page_pdf(pages)
        path = _write_pdf(pdf_bytes)
        try:
            result = extract_pdf(path)
            assert "Page One Content" in result
            assert "Page Three Content" in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_extract_preserves_reading_order(self):
        """同一页内文字按阅读顺序（y/x）排列。"""
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        # Insert text at different y positions
        page.insert_text((50, 200), "Second paragraph", fontsize=12)
        page.insert_text((50, 100), "First paragraph", fontsize=12)
        pdf_bytes = doc.tobytes()
        doc.close()

        path = _write_pdf(pdf_bytes)
        try:
            result = extract_pdf(path)
            first_pos = result.find("First paragraph")
            second_pos = result.find("Second paragraph")
            assert first_pos >= 0
            assert second_pos >= 0
            # "First paragraph" at y=50 should appear before "Second paragraph" at y=100
            assert first_pos < second_pos, "reading order should be top-to-bottom"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_extract_empty_pdf(self):
        """空 PDF 提取结果为空或不抛异常。"""
        import fitz

        doc = fitz.open()
        _page = doc.new_page()  # blank page
        pdf_bytes = doc.tobytes()
        doc.close()

        path = _write_pdf(pdf_bytes)
        try:
            result = extract_pdf(path)
            assert result == "" or result is not None
        finally:
            Path(path).unlink(missing_ok=True)


class TestCountPages:
    """count_pages() 功能"""

    def test_count_single_page(self):
        """单页 PDF 返回 1。"""
        pdf_bytes = _make_pdf_text("test")
        path = _write_pdf(pdf_bytes)
        try:
            assert count_pages(path) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_count_multiple_pages(self):
        """多页 PDF 返回正确页数。"""
        pdf_bytes = _make_multi_page_pdf(["p1", "p2", "p3"])
        path = _write_pdf(pdf_bytes)
        try:
            assert count_pages(path) == 3
        finally:
            Path(path).unlink(missing_ok=True)

    def test_count_no_pymupdf_fallback(self):
        """PyMuPDF 不可用时返回 0。"""

        # Temporarily simulate fitz not being available
        original = __import__("fitz", globals(), locals(), [], 0)

        def reload_without_fitz():
            import sys

            if "fitz" in sys.modules:
                del sys.modules["fitz"]

        # Just verify the function handles None gracefully
        # By patching the module-level flag
        import paper_review.extractor as ext

        original_has = ext.HAS_PYMUPDF
        try:
            ext.HAS_PYMUPDF = False
            assert count_pages("nonexistent.pdf") == 0
        finally:
            ext.HAS_PYMUPDF = original_has
