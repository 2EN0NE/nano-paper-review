"""04-extract-features.py 管线模板测试 —— 标签库自更新（替代写死词表）。

验证：
  1. ``_load_tag_library`` 从 papers.tags 聚合标签库（去重保序）
  2. 无 index.sqlite 时返回空（冷启动由种子词表兜底）
  3. ``extract_keywords`` 用标签库做子串匹配
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import Store

TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "src/paper_review/templates/pre-review/04-extract-features.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("extract_keywords", TEMPLATE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()


def _add_paper_with_tags(store: Store, fid: str, tags: list[str]) -> str:
    paper = make_sample_paper(fid)
    chunks = chunk_paper(paper)
    cvs = make_mock_chunk_vecs(chunks)
    store.add_paper(paper, cvs)
    store.update_tags(paper.paper_id, tags)
    return paper.paper_id


class TestLoadTagLibrary:
    def test_aggregates_tags_dedup_preserves_order(self, tmp_path):
        """从 papers.tags 聚合标签库，去重且保持首次出现顺序。"""
        index_dir = tmp_path / "index"
        store = Store(str(index_dir / "index.sqlite"))
        _add_paper_with_tags(store, "信用评估", ["数据库", "流量回放", "SQL"])
        _add_paper_with_tags(store, "系统调度", ["流量回放", "时序控制", "数据库"])
        store.close()

        tags = _mod._load_tag_library(str(index_dir))
        assert tags == ["数据库", "流量回放", "SQL", "时序控制"]

    def test_empty_when_no_db(self, tmp_path):
        """index.sqlite 不存在时返回空列表（冷启动由种子词表兜底）。"""
        assert _mod._load_tag_library(str(tmp_path / "no-index")) == []

    def test_empty_when_no_tags(self, tmp_path):
        """有库但无任何标签时返回空列表。"""
        index_dir = tmp_path / "index"
        store = Store(str(index_dir / "index.sqlite"))
        paper = make_sample_paper("信用评估")
        store.add_paper(paper, make_mock_chunk_vecs(chunk_paper(paper)))
        store.close()

        assert _mod._load_tag_library(str(index_dir)) == []


class TestExtractKeywords:
    def test_matches_tag_library(self):
        """用标签库做子串匹配，去重保序。"""
        lib = ["数据库", "流量回放", "SQL", "时序控制"]
        text = "本文讨论数据库容量评估与流量回放技术，并给出 SQL 时序控制方案"
        assert _mod.extract_keywords(text, lib) == ["数据库", "流量回放", "SQL", "时序控制"]

    def test_no_match_returns_empty(self):
        assert _mod.extract_keywords("完全无关的内容", ["数据库", "流量回放"]) == []


class TestMergeFeatureSets:
    """技术特征集并集汇总（LLM 主线 + 词表兜底，ADR 0015）。"""

    def test_union_dedup_llm_first(self):
        """并集去重，LLM 结果优先、词表结果补漏。"""
        assert _mod.merge_feature_sets(["向量化执行", "MPP"], ["MPP", "列式存储"]) == [
            "向量化执行",
            "MPP",
            "列式存储",
        ]

    def test_llm_only(self):
        assert _mod.merge_feature_sets(["向量化执行"], []) == ["向量化执行"]

    def test_keyword_fallback_only(self):
        """LLM 失败（空）时，词表兜底独占。"""
        assert _mod.merge_feature_sets([], ["数据库", "流量回放"]) == ["数据库", "流量回放"]

    def test_both_empty(self):
        assert _mod.merge_feature_sets([], []) == []


class TestBuildFeaturePrompt:
    """LLM 抽取 prompt 构建（粒度约束前置，ADR 0015）。"""

    def test_contains_text_and_constraints(self):
        prompt = _mod._build_feature_prompt("本文讨论向量化执行引擎")
        assert "向量化执行引擎" in prompt
        assert "技术方法" in prompt
        assert "JSON" in prompt

    def test_constrains_to_method_not_domain(self):
        """prompt 要求只抽技术方法，不抽宽泛领域词。"""
        prompt = _mod._build_feature_prompt("x")
        assert "不要抽取宽泛的领域词" in prompt


class TestParseFeatureJson:
    """从 pi 输出提取 JSON 数组（容忍 Markdown 包裹，ADR 0014 Bug C）。"""

    def test_plain_json_array(self):
        assert _mod._parse_feature_json('["向量化执行", "MPP"]') == ["向量化执行", "MPP"]

    def test_fenced_code_block(self):
        assert _mod._parse_feature_json('```json\n["向量化执行", "MPP"]\n```') == [
            "向量化执行",
            "MPP",
        ]

    def test_markdown_wrapped(self):
        """JSON 被说明文字包裹 + fenced block，仍能提取。"""
        assert _mod._parse_feature_json('结果是：\n```json\n["向量化执行"]\n```\n以上') == [
            "向量化执行"
        ]

    def test_invalid_returns_empty(self):
        assert _mod._parse_feature_json("没有 JSON") == []

    def test_empty_returns_empty(self):
        assert _mod._parse_feature_json("") == []


class TestLoadSubjectPaperIds:
    """读 02-auto-index 的 subject name → paper_id 映射。"""

    def test_reads_mapping(self, tmp_path):
        intermediates = tmp_path / "intermediates"
        out = intermediates / "pre" / "02-auto-index" / "output.json"
        out.parent.mkdir(parents=True)
        out.write_text(
            json.dumps({"data": {"subject_paper_ids": {"a": "pid1", "b": "pid2"}}}),
            encoding="utf-8",
        )
        assert _mod._load_subject_paper_ids(str(intermediates)) == {"a": "pid1", "b": "pid2"}

    def test_no_file_returns_empty(self, tmp_path):
        assert _mod._load_subject_paper_ids(str(tmp_path / "nonexistent")) == {}


class TestLlmExtractFeatures:
    """LLM 主线抽取（mock pi 二进制）。"""

    def test_mock_pi_returns_json_array(self, tmp_path):
        mock_pi = tmp_path / "pi"
        mock_pi.write_text('#!/bin/sh\necho \'["向量化执行", "MPP"]\'\n')
        mock_pi.chmod(0o755)
        assert _mod._llm_extract_features("本文讨论向量化执行", pi_binary=str(mock_pi)) == [
            "向量化执行",
            "MPP",
        ]

    def test_pi_nonzero_exit_returns_empty(self, tmp_path):
        mock_pi = tmp_path / "pi"
        mock_pi.write_text("#!/bin/sh\nexit 1\n")
        mock_pi.chmod(0o755)
        assert _mod._llm_extract_features("x", pi_binary=str(mock_pi)) == []

    def test_pi_not_found_returns_empty(self):
        assert _mod._llm_extract_features("x", pi_binary="/nonexistent/pi") == []
