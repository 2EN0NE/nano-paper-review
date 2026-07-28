"""
E2E: 完整用户流程测试 — 使用合成 PDF 验证 index → status → search 闭环。

测试策略：
- 使用 PyMuPDF（fitz）在内存生成含已知中文内容的合成 PDF
- 使用临时 data_dir 避免污染用户数据
- 通过 subprocess 调用 paper-review CLI（不 mock）
- 验证 index → status → search 三阶段的输出正确性

运行方式：
    python -m pytest tests/e2e/test_cli_user_flow.py -v
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

# 用 fitz（PyMuPDF）在内存生成测试 PDF
pytest.importorskip("fitz")


# ── 合成 PDF 生成 ─────────────────────────────────


def _make_paper_pdf(pdf_path: str, title: str, body_paragraphs: list[str]):
    """生成一篇含标题 + 正文段落的测试 PDF。

    Args:
        pdf_path: 输出文件路径
        title: 论文标题（大字号，出现在第一页上方）
        body_paragraphs: 正文段落列表，每段单独一页（模拟真实论文的页面）
    """
    import fitz

    doc = fitz.open()

    # 第 1 页：标题 + 摘要
    page = doc.new_page()
    # 标题（大字体）
    page.insert_text((50, 80), title, fontsize=18)
    page.insert_text((50, 130), "摘  要", fontsize=14)
    for i, para in enumerate(body_paragraphs[:2]):
        page.insert_text((50, 170 + i * 40), para[:80], fontsize=11)

    # 正文页（每段一页，模拟论文分页）
    for para in body_paragraphs:
        page = doc.new_page()
        lines = []
        # 简单换行：每行约 60 字
        for j in range(0, len(para), 60):
            lines.append(para[j : j + 60])
        for i, line in enumerate(lines[:12]):  # 每页最多 12 行
            page.insert_text((50, 80 + i * 28), line, fontsize=11)

    # 参考文献页
    page = doc.new_page()
    page.insert_text((50, 80), "参考文献", fontsize=14)
    page.insert_text((50, 130), "[1] 张三, 李四. " + title + "方法研究. 2025.", fontsize=11)

    doc.save(pdf_path)
    doc.close()


# ── 生成多篇互不相同的测试 PDF ──────────────────


_SAMPLE_PAPERS: list[dict] = [
    {
        "filename": "2025_张三_深度学习在NLP中的应用.pdf",
        "title": "深度学习在自然语言处理中的应用研究",
        "body": [
            "本文提出了一种基于Transformer的深度学习方法，用于解决自然语言处理中的序列建模问题。"
            "该方法结合了注意力机制和位置编码，在大规模语料库上取得了显著效果。",
            "实验结果表明，本方法在文本分类、命名实体识别和机器翻译三个任务上均超越了基线模型。"
            "尤其是在低资源场景下，本方法的泛化能力表现突出。",
            "与传统的循环神经网络相比，Transformer架构具有更好的并行计算能力。"
            "我们还引入了知识蒸馏技术，将大模型的知识迁移到轻量级模型中。",
            "未来工作将探索多模态学习，将文本与图像信息进行融合，进一步提升模型的理解能力。",
        ],
    },
    {
        "filename": "2025_李四_强化学习在游戏中的应用.pdf",
        "title": "强化学习在实时策略游戏中的应用探索",
        "body": [
            "本文研究了强化学习在实时策略游戏中的应用，提出了一种基于深度Q网络的决策框架。"
            "该框架能够处理高维状态空间和连续动作空间，在多个游戏场景中表现出色。",
            "我们设计了一种新的奖励函数，结合了短期收益和长期战略目标。"
            "实验结果表明，该方法在星际争霸II的微操作任务中达到了专业玩家的水平。",
            "与传统的基于规则的方法相比，强化学习能够自动发现新的战术策略。"
            "我们在多个基准地图上进行了广泛的实验验证。",
            "未来的工作将集中在多智能体协作和迁移学习方向，使模型能够适应更复杂的游戏环境。",
        ],
    },
    {
        "filename": "2025_王五_推荐系统中的图神经网络.pdf",
        "title": "图神经网络在推荐系统中的应用综述",
        "body": [
            "推荐系统是信息过滤的重要工具，图神经网络为建模用户-物品交互提供了新的思路。"
            "本文系统综述了图神经网络在推荐系统中的应用，包括协同过滤和知识图谱增强。",
            "我们总结了近年来提出的主要模型架构，如NGCF、LightGCN和KGAT。"
            "这些模型通过高阶连通性捕获用户偏好，显著提升了推荐的准确性和多样性。",
            "实验分析表明，图神经网络在冷启动场景下尤其有效，能够缓解数据稀疏问题。"
            "我们还讨论了现有方法的局限性，并展望了未来的研究方向。",
        ],
    },
]


def _generate_test_pdfs(target_dir: Path) -> list[Path]:
    """在 target_dir 下生成测试 PDF 文件，返回文件路径列表。"""
    paths: list[Path] = []
    for paper in _SAMPLE_PAPERS:
        pdf_path = target_dir / paper["filename"]
        _make_paper_pdf(str(pdf_path), paper["title"], paper["body"])
        paths.append(pdf_path)
    return paths


# ── CLI 辅助 ──────────────────────────────────────


def _paper_review_bin() -> str:
    """返回 paper-review 可执行文件路径。"""
    bindir = Path(sys.executable).parent
    candidate = bindir / "paper-review"
    if candidate.exists():
        return str(candidate)
    which = subprocess.run(["which", "paper-review"], capture_output=True, text=True, check=False)
    if which.returncode == 0:
        return which.stdout.strip()
    return f"{sys.executable} -m paper_review"


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_paper_review_bin(), *args],
        capture_output=True,
        text=True,
        check=check,
    )


# ── Fixtures ──────────────────────────────────────


@pytest.fixture(scope="module")
def pdf_dir(tmp_path_factory):
    """生成测试用合成 PDF。测试结束时自动清理。"""
    dest = tmp_path_factory.mktemp("pdfs")
    _generate_test_pdfs(dest)
    yield dest
    shutil.rmtree(dest, ignore_errors=True)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """测试专用空数据目录。测试结束时自动清理。"""
    dest = tmp_path_factory.mktemp("paper-review-data")
    yield dest
    shutil.rmtree(dest, ignore_errors=True)


# ── 测试类 ────────────────────────────────────────


class TestUserFlow:
    """完整用户流程：index → status → search（单方法内顺序执行）"""

    def test_complete_user_flow(self, pdf_dir: Path, data_dir: Path):
        """
        完整用户流程演练：index → status → search → pool-filter → idempotent。

        作为单个测试方法避免依赖 pytest 测试顺序，确保独立可重复。
        """
        import re

        # ════════════════════════════════════════
        # Step 1: index
        # ════════════════════════════════════════
        result = _run(
            "--data-dir",
            str(data_dir),
            "index",
            "--pdf-dir",
            str(pdf_dir),
            "--pool",
            "history",
            "--epoch-size",
            "10",
            check=False,
        )
        assert result.returncode == 0, f"Index 失败 (rc={result.returncode})\n{result.stdout[:500]}"
        for paper in _SAMPLE_PAPERS:
            assert paper["filename"] in result.stdout
        assert "索引完成" in result.stdout

        # ════════════════════════════════════════
        # Step 2: status
        # ════════════════════════════════════════
        result = _run("--data-dir", str(data_dir), "status", check=False)
        assert result.returncode == 0
        assert "论文总数" in result.stdout
        assert "history" in result.stdout
        assert "3" in result.stdout

        # ════════════════════════════════════════
        # Step 3: search
        # ════════════════════════════════════════
        result = _run(
            "--data-dir",
            str(data_dir),
            "search",
            "深度学习",
            "--skip-warnings",
            check=False,
        )
        assert result.returncode == 0
        assert "找到" in result.stdout
        assert "深度学习" in result.stdout or "自然语言处理" in result.stdout

        # ════════════════════════════════════════
        # Step 4: multi-keyword search
        # ════════════════════════════════════════
        for query, expect in [
            ("强化学习 游戏", "强化学习"),
            ("图神经网络 推荐", "图神经网络"),
        ]:
            r = _run(
                "--data-dir",
                str(data_dir),
                "search",
                query,
                "--skip-warnings",
                check=False,
            )
            assert r.returncode == 0
            assert "找到" in r.stdout
            assert expect in r.stdout, f"查询 {query!r} 应匹配 {expect}:\n{r.stdout}"

        # ════════════════════════════════════════
        # Step 5: pool filter
        # ════════════════════════════════════════
        rh = _run(
            "--data-dir",
            str(data_dir),
            "search",
            "深度学习",
            "--pool",
            "history",
            "--skip-warnings",
            check=False,
        )
        assert rh.returncode == 0
        assert "找到" in rh.stdout

        rp = _run(
            "--data-dir",
            str(data_dir),
            "search",
            "深度学习",
            "--pool",
            "pending",
            "--skip-warnings",
            check=False,
        )
        assert rp.returncode == 0
        assert "无匹配结果" in rp.stdout

        # ════════════════════════════════════════
        # Step 6: idempotent re-index
        # ════════════════════════════════════════
        result = _run(
            "--data-dir",
            str(data_dir),
            "index",
            "--pdf-dir",
            str(pdf_dir),
            "--pool",
            "history",
            "--epoch-size",
            "10",
            check=False,
        )
        assert result.returncode == 0

        sr = _run("--data-dir", str(data_dir), "status", check=False)
        assert sr.returncode == 0
        for line in sr.stdout.split("\n"):
            if "论文总数" in line:
                nums = re.findall(r"\d+", line)
                if nums:
                    assert int(nums[0]) == 3, f"去重失败: 期望 3 篇, 实际 {nums[0]}"
                    break
        else:
            pytest.fail(f"未找到 '论文总数' 行.\n{sr.stdout}")


class TestEdgeCases:
    """边缘场景测试"""

    def test_index_empty_dir(self, data_dir: Path, tmp_path: Path):
        """空目录索引应报错退出。"""
        empty_dir = tmp_path / "empty-pdfs"
        empty_dir.mkdir()

        result = _run(
            "--data-dir",
            str(data_dir),
            "index",
            "--pdf-dir",
            str(empty_dir),
            check=False,
        )
        assert result.returncode != 0, "空目录应返回非零退出码"
        assert "未找到" in result.stdout

    def test_search_empty_index(self, data_dir: Path):
        """空索引搜索应返回空结果而不是崩溃。"""
        empty_data = data_dir / "empty"
        empty_data.mkdir()

        result = _run(
            "--data-dir",
            str(empty_data),
            "search",
            "深度学习",
            "--skip-warnings",
            check=False,
        )
        assert result.returncode == 0
        empty_msg = ("无匹配结果" in result.stdout) or ("空" in result.stdout)
        assert empty_msg, f"空索引搜索应提示无结果.\n{result.stdout}"
