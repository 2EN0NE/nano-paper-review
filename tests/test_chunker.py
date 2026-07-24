"""分块逻辑测试"""

import pytest

from paper_rag.chunker import chunk_paper
from paper_rag.store import Paper, PaperMeta


@pytest.fixture
def sample_paper():
    """生成一篇模拟论文，包含中文段落和参考文献"""
    text = "\n\n".join(
        [
            "中文论文标题：基于深度学习的信用评估方法研究",
            "",
            "摘  要",
            "本文提出了一种新的信用评估方法，结合了深度神经网络和特征工程。"
            "在多个数据集上取得了优异表现。",
            "",
            "1  引言",
            "近年来，信用评估领域取得了长足进展。传统方法依赖特征工程，"
            "而深度学习方法能自动学习高层特征。",
            "",
            "2  方法",
            "本文提出的方法包含三个步骤：(1) 数据预处理 (2) 特征提取 (3) 模型训练。"
            "其中特征提取采用了自注意力机制。",
            "",
            "3  实验",
            "在三个公开数据集上，我们的方法相比基线提升了 5.7%。消融实验验证了每个模块的有效性。",
            "",
            "4  结论",
            "本文提出了面向信用评估的深度学习方法。未来工作将探索跨领域迁移。",
            "",
            "参考文献",
            "[1] Zhang et al., 2020",
            "[2] Li et al., 2021",
        ]
    )
    return Paper(
        paper_id="test123",
        filepath="data/history/2023_张三_信用评估.pdf",
        meta=PaperMeta(
            filename="2023_张三_信用评估.pdf",
            title_hint="信用评估",
            year=2023,
            author_hint="张三",
        ),
        raw_text=text,
        pages=4,
        pool="history",
    )


def test_chunk_count(sample_paper):
    """确保分块产生至少一个 chunk"""
    chunks = chunk_paper(sample_paper)
    assert len(chunks) > 0


def test_reference_filtering(sample_paper):
    """参考文献后的内容不应出现在任何 chunk 中"""
    chunks = chunk_paper(sample_paper)
    for c in chunks:
        assert "Zhang et al., 2020" not in c.text
        assert "Li et al., 2021" not in c.text


def test_chunk_has_paper_id(sample_paper):
    """每个 chunk 必须关联到源论文"""
    chunks = chunk_paper(sample_paper)
    for c in chunks:
        assert c.paper_id == "test123"
        assert c.chunk_id.startswith("test123#")


def test_chunk_overlap(sample_paper):
    """连续 chunk 之间应有重叠"""
    chunks = chunk_paper(sample_paper)
    if len(chunks) >= 2:
        # 检查前一个 chunk 的尾部是否在后一个 chunk 头部附近
        # 这是滑动窗口的基本性质
        assert chunks[0].end_pos - chunks[1].start_pos >= 0


def test_chunk_size_bound(sample_paper):
    """每个 chunk 不应超过指定大小太多"""
    chunks = chunk_paper(sample_paper, chunk_size=30, overlap=10)
    for c in chunks:
        assert len(c.text) <= 45  # 允许少许超出的边界情况


def test_position_weights(sample_paper):
    """chunk 的位置权重应该是 head/body/tail 之一"""
    chunks = chunk_paper(sample_paper)
    for c in chunks:
        assert c.position_weight in (5.0, 2.0, 4.0)


def test_sequential_page_numbers(sample_paper):
    """页码应该是合理的（1起算）"""
    chunks = chunk_paper(sample_paper)
    for c in chunks:
        assert c.page_num >= 1
        assert c.page_num <= sample_paper.pages
