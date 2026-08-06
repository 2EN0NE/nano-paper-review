"""
01-auto-index 单元测试 — 哨兵检查、冲突文件重命名、index 配置解析。

测试 seam: 纯函数，无需 mock（遵守 SPEC.md 红线）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from paper_review.auto_index import (
    check_sentinel,
    migrate_legacy_pdfs_dir,
    resolve_copy_path,
    resolve_index_config,
    write_sentinel,
)

# ============================================================================
# Sentinel
# ============================================================================


class TestSentinel:
    def test_check_returns_false_when_missing(self, tmp_path: Path):
        assert not check_sentinel(tmp_path)

    def test_check_returns_true_after_write(self, tmp_path: Path):
        write_sentinel(tmp_path)
        assert check_sentinel(tmp_path)

    def test_write_creates_empty_file(self, tmp_path: Path):
        write_sentinel(tmp_path)
        sentinel = tmp_path / ".auto-index-done"
        assert sentinel.exists()
        assert sentinel.read_text() == ""


# ============================================================================
# Resolve copy path
# ============================================================================


def _make_tmp_pdf(dir_path: Path, name: str, content: str) -> Path:
    """在 dir_path 下创建一个最小 PDF 并写入 content。"""
    p = dir_path / name
    p.write_text(content)
    return p


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


class TestResolveCopyPath:
    def test_no_conflict_returns_original_stem(self, tmp_path: Path):
        """dest_dir 空 → 直接用原名。"""
        src = _make_tmp_pdf(tmp_path, "paper.pdf", "hello")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        target, skipped = resolve_copy_path(src, dest_dir)

        assert not skipped
        assert target == dest_dir / "paper.pdf"

    def test_same_hash_skips(self, tmp_path: Path):
        """同名 + 同 hash → 跳过复制。"""
        content = "identical content"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        existing = _make_tmp_pdf(dest_dir, "paper.pdf", content)

        src = _make_tmp_pdf(tmp_path, "paper.pdf", content)

        target, skipped = resolve_copy_path(src, dest_dir)

        assert skipped
        assert target == existing

    def test_same_name_different_hash_renames(self, tmp_path: Path):
        """同名 + 不同内容 → 重命名。"""
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        _make_tmp_pdf(dest_dir, "paper.pdf", "old content")

        src_content = "different content"
        src = _make_tmp_pdf(tmp_path, "paper.pdf", src_content)

        target, skipped = resolve_copy_path(src, dest_dir)

        assert not skipped
        # 应重命名为 {stem}_{timestamp}_{hash[:8]}.pdf
        assert target.parent == dest_dir
        assert target.name.startswith("paper_")
        assert target.name.endswith(".pdf")
        assert _sha256(src_content)[:8] in target.name

    def test_different_name_no_conflict(self, tmp_path: Path):
        """不同名 → 直接用原名。"""
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        _make_tmp_pdf(dest_dir, "other.pdf", "something")

        src = _make_tmp_pdf(tmp_path, "paper.pdf", "hello")

        target, skipped = resolve_copy_path(src, dest_dir)

        assert not skipped
        assert target == dest_dir / "paper.pdf"

    def test_same_hash_different_name_still_skips(self, tmp_path: Path):
        """不同名但同 hash → 仍然跳过复制（去重优先）。"""
        content = "same stuff"
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        existing = _make_tmp_pdf(dest_dir, "old_name.pdf", content)

        src = _make_tmp_pdf(tmp_path, "new_name.pdf", content)

        target, skipped = resolve_copy_path(src, dest_dir)

        assert skipped
        assert target == existing  # 复用已有路径


# ============================================================================
# Index config resolution
# ============================================================================


class TestResolveIndexConfig:
    def test_empty_config_uses_defaults(self, tmp_path: Path):
        """无 index 段 → 全部取默认值。"""
        data_dir = tmp_path
        config = resolve_index_config(None, data_dir)

        assert config.store_dir == data_dir / "index"
        assert config.reference_dir == data_dir / "origin" / "pdf"
        assert config.auto_index
        assert config.copy_subjects

    def test_explicit_paths_override_defaults(self, tmp_path: Path):
        """显式路径 → 覆盖默认值。"""
        data_dir = tmp_path
        raw = {
            "index": {
                "store_dir": "/custom/store",
                "reference_dir": "/custom/refs",
            }
        }
        config = resolve_index_config(raw, data_dir)

        assert config.store_dir == Path("/custom/store")
        assert config.reference_dir == Path("/custom/refs")

    def test_relative_paths_resolve_against_data_dir(self, tmp_path: Path):
        """相对路径 → 相对于 data_dir 解析。"""
        data_dir = tmp_path
        raw = {
            "index": {
                "store_dir": "my_index",
                "reference_dir": "my_pdfs",
            }
        }
        config = resolve_index_config(raw, data_dir)

        assert config.store_dir == data_dir / "my_index"
        assert config.reference_dir == data_dir / "my_pdfs"

    def test_bools_default_to_true(self, tmp_path: Path):
        """auto_index 和 copy_subjects 默认 true。"""
        config = resolve_index_config({}, data_dir=tmp_path)

        assert config.auto_index
        assert config.copy_subjects

    def test_bools_can_be_disabled(self, tmp_path: Path):
        """显式设为 false。"""
        raw = {
            "index": {
                "auto_index": False,
                "copy_subjects": False,
            }
        }
        config = resolve_index_config(raw, data_dir=tmp_path)

        assert not config.auto_index
        assert not config.copy_subjects


# ============================================================================
# Migration
# ============================================================================


class TestMigration:
    def test_migrates_when_old_exists_new_missing(self, tmp_path: Path):
        old = tmp_path / "pdfs"
        old.mkdir()
        (old / "dummy.pdf").write_text("test")

        assert migrate_legacy_pdfs_dir(tmp_path)
        assert not old.exists()
        assert (tmp_path / "origin" / "pdf" / "dummy.pdf").exists()

    def test_noop_when_new_already_exists(self, tmp_path: Path):
        old = tmp_path / "pdfs"
        old.mkdir()
        new = tmp_path / "origin" / "pdf"
        new.mkdir(parents=True)

        assert not migrate_legacy_pdfs_dir(tmp_path)
        assert old.exists()

    def test_noop_when_old_missing(self, tmp_path: Path):
        assert not migrate_legacy_pdfs_dir(tmp_path)
