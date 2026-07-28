"""文件名元数据提取测试"""

from paper_review.extractor import extract_meta


def test_year_author_title_format():
    """2023_张三_深度学习信用风险评估.pdf"""
    meta = extract_meta("2023_张三_深度学习信用风险评估.pdf")
    assert meta.year == 2023
    assert meta.author_hint == "张三"
    assert meta.title_hint == "深度学习信用风险评估"
    assert meta.arxiv_id == ""


def test_author_title_year_format():
    """李四_图神经网络推荐系统_2024.pdf"""
    meta = extract_meta("李四_图神经网络推荐系统_2024.pdf")
    assert meta.year == 2024
    assert meta.author_hint == "李四"
    assert meta.title_hint == "图神经网络推荐系统"


def test_arxiv_id_format():
    """2310.07554.pdf"""
    meta = extract_meta("2310.07554.pdf")
    assert meta.arxiv_id == "2310.07554"

    meta2 = extract_meta("arXiv_2310.07554_v2.pdf")
    assert meta2.arxiv_id == "2310.07554"


def test_bare_title_format():
    """基于对比学习的文本表示方法研究.pdf"""
    meta = extract_meta("基于对比学习的文本表示方法研究.pdf")
    assert meta.title_hint == "基于对比学习的文本表示方法研究"
    assert meta.year == 0
    assert meta.author_hint == ""
    assert meta.arxiv_id == ""


def test_user_format():
    """01.提案表-基于深度学习的信用评估-张三.pdf"""
    meta = extract_meta("01.提案表-基于深度学习的信用评估-张三.pdf")
    assert meta.title_hint == "基于深度学习的信用评估"
    assert meta.author_hint == "张三"
    assert meta.year == 0


def test_user_format_with_year():
    """01.提案表-基于深度学习的信用评估-张三-2023.pdf"""
    meta = extract_meta("01.提案表-基于深度学习的信用评估-张三-2023.pdf")
    assert meta.title_hint == "基于深度学习的信用评估"
    assert meta.author_hint == "张三"
    assert meta.year == 2023


def test_no_extension():
    """没有扩展名的文件名应该也能解析"""
    meta = extract_meta("2023_王五_系统调度.pdf")
    assert meta.year == 2023
    assert meta.author_hint == "王五"
    assert meta.title_hint == "系统调度"


def test_garbled_filename():
    """无规律文件名退回兜底"""
    meta = extract_meta("paper_final_v3_revised_bob.pdf")
    assert meta.title_hint  # 应该至少有个标题推测
    assert meta.year == 0
    assert meta.arxiv_id == ""
