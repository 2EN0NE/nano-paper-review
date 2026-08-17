"""
PDF 文本提取 + 文件名元数据解析

Extractor 负责从 PDF 提取文本、从文件名解析元数据。
"""

from __future__ import annotations

import logging
import re
import threading

from paper_review.search.store import PaperMeta

logger = logging.getLogger(__name__)

# ============================================================================
# 文件名模式
# ============================================================================

# 01.提案表-基于深度学习-张三.pdf（用户实际命名格式）
PAT_USER = re.compile(r"^\d+(?:\.)\s*[^\-]+\-(.+)\-([^\d]+?)(?:\-(\d{4}))?(?:\.pdf)?$")
# 2023_张三_深度学习.pdf
PAT_YEAR_AUTHOR_TITLE = re.compile(r"(\d{4})[_-]([^_\-]+)[_-](.+?)(?:\.pdf)?$")
# 张三_深度学习_2023.pdf
PAT_AUTHOR_TITLE_YEAR = re.compile(r"([^_\-]+)[_-](.+?)[_-](\d{4})(?:\.pdf)?$")
# arXiv_2310.07554.pdf 或 1704.01279.pdf
PAT_ARXIV = re.compile(r"(?:arXiv[_-])?(\d{4}\.\d{4,5})(?:v\d+)?.*(?:\.pdf)?$")
# 纯标题（兜底）
PAT_FALLBACK = re.compile(r"(.+?)(?:\.pdf)?$")


def extract_meta(filename: str) -> PaperMeta:
    """
    从文件名尽力解析元数据。

    支持格式（优先级递减）：
    1. {序号}.{文档类型}-{标题}-{作者}.pdf  → 用户命名格式
    2. {年份}_{作者}_{标题}.pdf
    3. 作者_标题_年份
    4. arXiv ID
    5. 纯标题兜底
    """
    name = filename.removesuffix(".pdf").removesuffix(".PDF").strip()
    meta = PaperMeta(filename=filename)

    # 1. 用户命名格式：01.提案表-基于深度学习-张三.pdf
    m = PAT_USER.match(name)
    if m:
        meta.title_hint = m.group(1).replace("_", " ").strip()
        meta.author_hint = m.group(2).strip()
        if m.group(3):
            try:
                meta.year = int(m.group(3))
            except ValueError:
                pass
        return meta

    # 2. 年份_作者_标题
    m = PAT_YEAR_AUTHOR_TITLE.match(name)
    if m:
        try:
            meta.year = int(m.group(1))
        except ValueError:
            pass
        meta.author_hint = m.group(2)
        meta.title_hint = m.group(3).replace("_", " ").strip()
        return meta

    # 3. 作者_标题_年份
    m = PAT_AUTHOR_TITLE_YEAR.match(name)
    if m:
        meta.author_hint = m.group(1)
        meta.title_hint = m.group(2).replace("_", " ").strip()
        try:
            meta.year = int(m.group(3))
        except ValueError:
            pass
        return meta

    # 4. arXiv ID
    m = PAT_ARXIV.match(name)
    if m:
        meta.arxiv_id = m.group(1)
        meta.title_hint = name.replace("_", " ").replace("-", " ")
        return meta

    # 5. 纯标题兜底
    m = PAT_FALLBACK.match(name)
    if m:
        meta.title_hint = name.replace("_", " ").replace("-", " ").strip()
        return meta

    meta.title_hint = name
    return meta


# ============================================================================
# PDF 文本提取
# ============================================================================

try:
    import fitz  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


HEADER_FOOTER_MARGIN = 50  # pt，页面顶部/底部过滤阈值
SHORT_TEXT_THRESHOLD = 50  # 字符数，低于此值的文本块可能是页眉页脚


def extract_pdf(pdf_path: str) -> str:
    """
    从 PDF 提取纯文本。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        提取后的纯文本
    """
    if not HAS_PYMUPDF:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF extraction. Install with: pip install pymupdf"
        )

    doc = fitz.open(pdf_path)
    pages_text: list[str] = []

    for page in doc:
        blocks = page.get_text("blocks")
        # 按阅读顺序排序（先 y 后 x）
        blocks.sort(key=lambda b: (round(b[1], 1), b[0]))

        page_height = page.rect.height
        texts: list[str] = []

        for b in blocks:
            txt = b[4].strip()
            if not txt:
                continue

            # 页眉页脚过滤：短文本 + 位于页面顶部/底部
            y0 = b[1]
            y1 = b[3]
            is_margin = y0 < HEADER_FOOTER_MARGIN or y1 > page_height - HEADER_FOOTER_MARGIN
            if is_margin and len(txt) < SHORT_TEXT_THRESHOLD:
                continue

            texts.append(txt)

        pages_text.append("\n".join(texts))

    doc.close()
    return "\n\n".join(pages_text)


def count_pages(pdf_path: str) -> int:
    """返回 PDF 的页数"""
    if not HAS_PYMUPDF:
        return 0
    doc = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count


# 软超时默认值（秒）：PyMuPDF 同步调用无超时，卡死 PDF 不阻塞批次
def extract_pdf_with_timeout(pdf_path: str, timeout: int | float = 60) -> tuple[str, bool]:
    """提取 PDF 文本，带软超时（卡死 PDF 不阻塞调用方）。

    PyMuPDF 同步调用无超时——用 daemon 线程 + join(timeout) 做软超时：
    超时后返回 ("", True)，卡死的线程不阻塞主流程（进程退出时回收）。
    异常/无 PyMuPDF 返回 ("", False)。

    Returns:
        (text, timed_out)：timed_out=True 表示线程超时仍在运行（调用方应
        避免继续累积卡死线程）。
    """
    result: dict[str, object] = {}

    def _run() -> None:
        try:
            result["text"] = extract_pdf(pdf_path)
        except Exception as e:  # noqa: BLE001 — 单篇失败隔离，吞掉继续批次
            result["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        logger.warning("extract_pdf 超时（>%ds）: %s", timeout, pdf_path)
        return "", True
    if "error" in result:
        logger.warning("extract_pdf 失败 %s: %s", pdf_path, result["error"])
        return "", False
    return str(result.get("text", "")), False
