"""
Auto-index 辅助函数 —— Sentinel 管理、PDF 复制冲突处理、Index 配置解析。

由 01-auto-index.py Pre Phase 步骤导入使用。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENTINEL_NAME = ".auto-index-done"


# ============================================================================
# IndexConfig（配置解析用轻量 dataclass）
# ============================================================================


@dataclass
class ResolvedIndexConfig:
    """已解析的 Index 配置（默认值已应用）。"""

    store_dir: Path
    reference_dir: Path
    auto_index: bool
    copy_subjects: bool


def resolve_index_config(
    pipeline_raw: dict[str, Any] | None,
    data_dir: Path,
) -> ResolvedIndexConfig:
    """从 pipeline YAML 原始 dict 解析 index 配置，应用默认值。

    Args:
        pipeline_raw: pipeline.yaml 的原始 dict（可为 None）。
        data_dir: .paper-review/ 数据目录，用于默认路径推导。

    Returns:
        ResolvedIndexConfig with all defaults applied.
    """
    raw = (pipeline_raw or {}).get("index", {}) or {}

    store_dir_str = raw.get("store_dir", "")
    reference_dir_str = raw.get("reference_dir", "")

    store_dir = Path(store_dir_str) if store_dir_str else (data_dir / "index")
    reference_dir = Path(reference_dir_str) if reference_dir_str else (data_dir / "origin" / "pdf")

    # 相对路径相对于 data_dir
    if store_dir_str and not store_dir.is_absolute():
        store_dir = data_dir / store_dir
    if reference_dir_str and not reference_dir.is_absolute():
        reference_dir = data_dir / reference_dir

    return ResolvedIndexConfig(
        store_dir=store_dir,
        reference_dir=reference_dir,
        auto_index=raw.get("auto_index", True),
        copy_subjects=raw.get("copy_subjects", True),
    )


# ============================================================================
# Sentinel
# ============================================================================


def check_sentinel(data_dir: Path) -> bool:
    """检查首次批量索引是否已完成。"""
    return (data_dir / SENTINEL_NAME).exists()


def write_sentinel(data_dir: Path) -> None:
    """标记首次批量索引已完成。"""
    (data_dir / SENTINEL_NAME).touch()


def migrate_legacy_pdfs_dir(data_dir: Path) -> bool:
    """将旧的 pdfs/ 目录迁移到 origin/pdf/。

    只在 pdfs/ 存在且 origin/pdf/ 不存在时执行。
    Returns:
        True 如果执行了迁移。
    """
    old_dir = data_dir / "pdfs"
    new_dir = data_dir / "origin" / "pdf"

    if old_dir.is_dir() and not new_dir.exists():
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        old_dir.rename(new_dir)
        return True
    return False


# ============================================================================
# PDF 复制冲突处理
# ============================================================================


def resolve_copy_path(src: Path, dest_dir: Path) -> tuple[Path, bool]:
    """决定 PDF 复制目标路径，处理同名冲突。

    规则（按优先级）：
    1. 计算 SHA-256 → 扫 dest_dir 下同 hash 文件 → 有则复用，跳过复制
    2. 同名但不同 hash → 重命名为 {stem}_{YYYYMMDD_HHmmss}_{hash[:8]}.pdf
    3. 无冲突 → 直接用原名

    Args:
        src: 源 PDF 路径。
        dest_dir: 目标目录。

    Returns:
        (target_path, skipped): target_path 为最终物理路径（可能是已有文件），
        skipped=True 表示可跳过复制。
    """
    content = src.read_bytes()
    src_hash = hashlib.sha256(content).hexdigest()

    # 1. 同 hash 去重
    for existing in dest_dir.glob("*.pdf"):
        try:
            existing_content = existing.read_bytes()
            if hashlib.sha256(existing_content).hexdigest() == src_hash:
                return existing, True
        except OSError:
            continue

    # 2. 检查同名冲突
    direct_path = dest_dir / src.name
    if direct_path.exists():
        # 同名但不同 hash（否则上面已跳过）
        short_hash = src_hash[:8]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = src.stem
        new_name = f"{stem}_{ts}_{short_hash}.pdf"
        return dest_dir / new_name, False

    # 3. 无冲突
    return direct_path, False
