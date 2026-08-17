"""Pre 模板步骤单篇超时与失败隔离（T7）。

验证：
1. extractor.extract_pdf_with_timeout：正常 / 异常 / 超时三种路径（卡死篇不阻塞批次）
2. 04._extract_pdf_with_timeout：卡死阈值（_MAX_STUCK_PDFS）后不再启动新线程
3. 04._process_subject：LLM 失败词表兜底 + per-subject 产物写入 + 写回计数
4. 05._search_subject：hybrid_search 异常 → 空结果 + error 消息（不中断批次）
5. 02._stale_existing_names：索引重建后续做时产物复用校验（重新索引缺失篇）
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

TEMPLATE_02 = (
    Path(__file__).resolve().parent.parent
    / "src/paper_review/templates/pre-review/02-auto-index.py"
)
TEMPLATE_04 = (
    Path(__file__).resolve().parent.parent
    / "src/paper_review/templates/pre-review/04-extract-features.py"
)
TEMPLATE_05 = (
    Path(__file__).resolve().parent.parent
    / "src/paper_review/templates/pre-review/05-batch-search.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestExtractorPdfTimeout:
    """extractor.extract_pdf_with_timeout —— 共享软超时实现（04/05 共用）。"""

    def _mod(self):
        import paper_review.extractor as ex

        return ex

    def test_normal_returns_text(self, monkeypatch):
        mod = self._mod()
        monkeypatch.setattr(mod, "extract_pdf", lambda p: "paper text")
        assert mod.extract_pdf_with_timeout("x.pdf", timeout=5) == ("paper text", False)

    def test_exception_returns_empty(self, monkeypatch):
        mod = self._mod()

        def boom(p):
            raise RuntimeError("corrupt pdf")

        monkeypatch.setattr(mod, "extract_pdf", boom)
        assert mod.extract_pdf_with_timeout("x.pdf", timeout=5) == ("", False)

    def test_timeout_returns_empty_and_flags(self, monkeypatch):
        """卡死的 extract_pdf（超时后线程仍运行）→ (空串, True)，主流程继续。"""
        mod = self._mod()

        def slow(p):
            time.sleep(1.0)
            return "late text"

        monkeypatch.setattr(mod, "extract_pdf", slow)
        assert mod.extract_pdf_with_timeout("x.pdf", timeout=0.05) == ("", True)


class TestExtractPdfTimeout:
    """04._extract_pdf_with_timeout —— 卡死篇不阻塞批次 + 阈值防线程累积。"""

    def _mod(self):
        return _load_module(TEMPLATE_04, "pre04")

    def test_normal_returns_text(self, monkeypatch):
        mod = self._mod()
        monkeypatch.setattr(
            mod, "extract_pdf_with_timeout", lambda p, timeout=60: ("paper text", False)
        )
        assert mod._extract_pdf_with_timeout("x.pdf", timeout=5) == "paper text"

    def test_exception_returns_empty(self, monkeypatch):
        mod = self._mod()
        monkeypatch.setattr(mod, "extract_pdf_with_timeout", lambda p, timeout=60: ("", False))
        assert mod._extract_pdf_with_timeout("x.pdf", timeout=5) == ""

    def test_timeout_counts_stuck(self, monkeypatch):
        """超时（timed_out=True）→ 返回空串且卡死计数 +1。"""
        mod = self._mod()
        monkeypatch.setattr(mod, "extract_pdf_with_timeout", lambda p, timeout=60: ("", True))
        assert mod._extract_pdf_with_timeout("x.pdf", timeout=0.05) == ""
        assert mod._stuck_count == 1

    def test_stuck_threshold_skips_extraction(self, monkeypatch):
        """累计卡死达 _MAX_STUCK_PDFS 后不再启动新线程（直接返回空串）。"""
        mod = self._mod()
        calls = []

        def timed_out(p, timeout=60):
            calls.append(p)
            return ("", True)

        monkeypatch.setattr(mod, "extract_pdf_with_timeout", timed_out)
        limit = mod._MAX_STUCK_PDFS
        # 前 limit 次启动线程（都超时），第 limit+1 次起不再调用 extractor
        for _ in range(limit):
            assert mod._extract_pdf_with_timeout("x.pdf") == ""
        assert len(calls) == limit
        assert mod._extract_pdf_with_timeout("x.pdf") == ""
        assert len(calls) == limit  # 阈值后不再启动
        assert mod._stuck_count == limit

    def test_timeout_default_is_bounded(self, monkeypatch):
        """默认超时是有限值（防意外无界等待）。"""
        mod = self._mod()
        assert mod._EXTRACT_PDF_TIMEOUT > 0
        assert mod._EXTRACT_PDF_TIMEOUT <= 300


class TestProcessSubject:
    """04._process_subject —— 单篇处理（提取→词表→LLM→写回→产物）。"""

    def _mod(self):
        return _load_module(TEMPLATE_04, "pre04")

    def _run(
        self, tmp_path, monkeypatch, *, llm_features=None, extract_text="深度学习 神经网络 技术"
    ):
        mod = self._mod()
        monkeypatch.setattr(
            mod, "extract_pdf_with_timeout", lambda p, timeout=60: (extract_text, False)
        )
        monkeypatch.setattr(
            mod,
            "_llm_extract_features",
            lambda *a, **k: llm_features if llm_features is not None else ["Transformer"],
        )
        intermediates = tmp_path / "intermediates"
        written, llm_count, _feature_count = mod._process_subject(
            name="paper-A",
            pdf_path=Path("/tmp/paper-A.pdf"),
            subject_paper_ids={},
            tag_library=["深度学习", "Transformer"],
            store=None,
            intermediates_dir=str(intermediates),
            pi_binary="pi",
        )
        out = json.loads(
            (intermediates / "paper-A" / "04-extract-features" / "output.json").read_text(
                encoding="utf-8"
            )
        )
        return mod, written, llm_count, out

    def test_writes_product_with_llm_and_keyword_union(self, tmp_path, monkeypatch):
        mod, written, llm_count, out = self._run(tmp_path, monkeypatch)
        assert out["status"] == "ok"
        # 并集去重：LLM 词 + 词表兜底词
        assert set(out["data"]["features"]) == {"深度学习", "Transformer"}
        assert out["data"]["llm_features"] == ["Transformer"]
        assert llm_count == 1
        assert written == 0  # store=None → 不写回

    def test_llm_failure_falls_back_to_keywords(self, tmp_path, monkeypatch):
        """LLM 失败（空列表）→ 词表兜底，产物仍有关键词。"""
        mod, written, llm_count, out = self._run(tmp_path, monkeypatch, llm_features=[])
        assert out["status"] == "ok"
        assert out["data"]["llm_features"] == []
        assert "深度学习" in out["data"]["features"]  # 词表兜底保证已知词不丢

    def test_empty_extract_still_writes_product(self, tmp_path, monkeypatch):
        """extract_pdf 返回空（失败/超时）→ 产物照写，不中断批次。"""
        mod, written, llm_count, out = self._run(tmp_path, monkeypatch, extract_text="")
        assert out["status"] == "ok"
        # 空文本无词表命中，但 LLM 词仍在（text_with_name 含篇名非空）
        assert out["data"]["features"] == ["Transformer"]
        assert out["data"]["keyword_count"] == 0

    def test_store_writeback_counts(self, tmp_path, monkeypatch):
        """store 提供且 paper_id 存在 → features 写回 +1。"""
        mod = self._mod()
        monkeypatch.setattr(
            mod, "extract_pdf_with_timeout", lambda p, timeout=60: ("深度学习 技术", False)
        )
        monkeypatch.setattr(mod, "_llm_extract_features", lambda *a, **k: [])
        calls: list[list[str]] = []

        class FakeStore:
            def update_features(self, paper_id, features):
                calls.append([paper_id, features])
                return True

        intermediates = tmp_path / "intermediates"
        written, _llm, _feat = mod._process_subject(
            name="paper-A",
            pdf_path=Path("/tmp/paper-A.pdf"),
            subject_paper_ids={"paper-A": "pid-1"},
            tag_library=["深度学习"],
            store=FakeStore(),
            intermediates_dir=str(intermediates),
            pi_binary="pi",
        )
        assert written == 1
        assert calls == [["pid-1", ["深度学习"]]]


class TestSearchSubjectIsolation:
    """05._search_subject —— 单篇检索失败不中断批次。"""

    def _mod(self):
        return _load_module(TEMPLATE_05, "pre05")

    def test_hybrid_search_exception_returns_error(self, monkeypatch):
        """hybrid_search 抛异常 → 空结果 + error 消息（批次继续）。"""
        mod = self._mod()

        def boom(*a, **k):
            raise RuntimeError("faiss broken")

        monkeypatch.setattr(mod, "hybrid_search", boom)
        subj_data, error = mod._search_subject(
            store=object(),
            query="q",
            name="paper-A",
            embed_model=None,
            reranker=None,
            exclude_hash=None,
            subject_features=None,
        )
        assert error is not None
        assert "faiss broken" in error
        assert subj_data["history"] == []
        assert subj_data["pending"] == []
        assert subj_data["history_count"] == 0
        assert subj_data["pending_count"] == 0

    def test_normal_path_serializes_results(self, monkeypatch):
        """正常路径：结果按 pool 分组序列化。"""
        mod = self._mod()

        class FakeResult:
            def __init__(self, pool):
                self.pool = pool
                self.paper_id = "p1"
                self.source_file = "a.pdf"
                self.title_hint = "t"
                self.author_hint = ""
                self.year = None
                self.combined_score = 0.9
                self.bm25_score = 0.1
                self.vector_score = 0.2
                self.rrf_score = 0.3
                self.rerank_score = 0.4
                self.matched_chunks = []

        monkeypatch.setattr(
            mod,
            "hybrid_search",
            lambda *a, **k: [FakeResult("history"), FakeResult("pending")],
        )
        subj_data, error = mod._search_subject(
            store=object(),
            query="q",
            name="paper-A",
            embed_model=None,
            reranker=None,
            exclude_hash=None,
            subject_features=None,
        )
        assert error is None
        assert subj_data["history_count"] == 1
        assert subj_data["pending_count"] == 1
        assert subj_data["history"][0]["paper_id"] == "p1"


class TestAutoIndexStoreValidation:
    """02._stale_existing_names —— 索引重建/清空后续做时产物复用校验。"""

    def _mod(self):
        return _load_module(TEMPLATE_02, "pre02")

    def test_all_present_no_stale(self):
        """store 中 paper_id 齐全 → 无 stale，全部复用。"""
        mod = self._mod()

        class FakeStore:
            def paper_exists(self, pid):
                return True

        products = {"a": {"data": {"paper_id": "pid-a"}}, "b": {"data": {"paper_id": "pid-b"}}}
        assert mod._stale_existing_names(products, FakeStore()) == []

    def test_missing_in_store_flagged(self):
        """store 缺失的 paper_id → 该篇标记 stale（重新索引）。"""
        mod = self._mod()

        class FakeStore:
            def paper_exists(self, pid):
                return pid != "pid-b"

        products = {"a": {"data": {"paper_id": "pid-a"}}, "b": {"data": {"paper_id": "pid-b"}}}
        assert mod._stale_existing_names(products, FakeStore()) == ["b"]

    def test_missing_paper_id_ignored(self):
        """产物缺 paper_id（旧版产物）→ 不判 stale（由调用方按无映射处理）。"""
        mod = self._mod()

        class FakeStore:
            def paper_exists(self, pid):
                return True

        products = {"a": {"data": {}}}
        assert mod._stale_existing_names(products, FakeStore()) == []
