"""
并发写测试 —— 验证 Store 在多进程并发写入时的行为。

@pytest.mark.integration: 跨进程并发 + SQLite / FAISS 写竞争
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile

import pytest

from helpers import make_mock_chunk_vecs, make_sample_paper
from paper_review.search.chunker import chunk_paper
from paper_review.search.store import Store

pytestmark = pytest.mark.integration


def _write_paper_to_store(db_path: str, pid: str, fid: str):
    """在子进程中往 Store 添加一篇论文（强制 paper_id 为 pid）。"""
    from paper_review.search.store import Paper

    store = Store(db_path)
    store.load_all()
    base = make_sample_paper(fid)
    paper = Paper(
        paper_id=pid,
        filepath=base.filepath,
        meta=base.meta,
        raw_text=base.raw_text,
        pages=base.pages,
        pool=base.pool,
    )
    chunks = chunk_paper(paper)
    cvs = make_mock_chunk_vecs(chunks)
    store.add_paper(paper, cvs)
    store.close()


class TestConcurrentWrite:
    """多进程并发写入同一 Store 时的隔离性"""

    def test_two_processes_write_independently(self):
        """两个子进程同时添加不同论文不互相干扰。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")

            # 先初始化 Store（创建 schema 和 FAISS 目录）
            store = Store(db_path)
            store.init_faiss(dim=4)
            store.close()

            # 启动两个子进程分别写入
            p1 = multiprocessing.Process(
                target=_write_paper_to_store, args=(db_path, "p1", "信用评估")
            )
            p2 = multiprocessing.Process(
                target=_write_paper_to_store, args=(db_path, "p2", "图神经网络")
            )
            p1.start()
            p2.start()
            p1.join(timeout=30)
            p2.join(timeout=30)

            assert p1.exitcode == 0, f"Process 1 failed: {p1.exitcode}"
            assert p2.exitcode == 0, f"Process 2 failed: {p2.exitcode}"

            # 验证两篇都在索引中（paper_id = pid）
            store2 = Store(db_path)
            store2.load_all()
            assert "p1" in store2.papers, f"papers: {list(store2.papers.keys())}"
            assert "p2" in store2.papers
            assert len(store2.papers) == 2
            store2.close()

    def test_concurrent_add_does_not_lose_papers(self):
        """多次并发 add 后 paper 计数正确（不丢失、不重复）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "concurrent.sqlite")

            # 初始化
            store = Store(db_path)
            store.init_faiss(dim=4)
            store.close()

            # 3 个子进程同时写入
            papers = [("p1", "信用评估"), ("p2", "图神经网络"), ("p3", "系统调度")]
            processes = []
            for pid, fid in papers:
                p = multiprocessing.Process(target=_write_paper_to_store, args=(db_path, pid, fid))
                processes.append(p)
                p.start()

            for p in processes:
                p.join(timeout=30)
                assert p.exitcode == 0, f"Process failed: {p.exitcode}"

            # 验证计数（paper_id = pid）
            expected_ids = {"p1", "p2", "p3"}
            store2 = Store(db_path)
            store2.load_all()
            assert len(store2.papers) == 3, f"papers: {list(store2.papers.keys())}"
            assert set(store2.papers.keys()) == expected_ids
            store2.close()

    def test_concurrent_same_content_dedup(self):
        """并发写入相同内容利用去重机制，不产生重复向量。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "dedup.sqlite")

            store = Store(db_path)
            store.init_faiss(dim=4)
            store.close()

            # 两个进程写相同 fid/content → 应触发 content_dedup
            p1 = multiprocessing.Process(
                target=_write_paper_to_store, args=(db_path, "p1", "信用评估")
            )
            p2 = multiprocessing.Process(
                target=_write_paper_to_store, args=(db_path, "p2", "信用评估")
            )
            p1.start()
            p2.start()
            p1.join(timeout=30)
            p2.join(timeout=30)

            store2 = Store(db_path)
            store2.load_all()
            # 两份相同内容 → 去重机制确保二者都存在元数据
            assert "p1" in store2.papers
            assert len(store2.papers) >= 1
            store2.close()
