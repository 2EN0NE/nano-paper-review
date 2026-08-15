"""scaffold.py — Scaffold Template 版本检测与托管清单（manifest）。

SCAFFOLD_VERSION 是 Scaffold Template（``src/paper_review/templates/``）的版本号，
仅在模板内容**实际变化**时递增（与包版本解耦，避免"发版但脚手架没变"时的误报）。

``init`` 生成 Pipelines Directory 时写入 manifest（``{data_dir}/.scaffold-manifest``），
记录版本号与脚手架写入的全部文件（相对 data_dir 的路径）。``review`` / ``status``
启动时读取 manifest 与当前 SCAFFOLD_VERSION 对比，检测脚手架漂移——Scaffold
Template 升级后，用户侧 data_dir 的实例化副本（Pipelines Directory）未同步。

孤儿文件清理依赖 manifest 的 files 清单：``init --reset`` 时，manifest 中记录、
当前模板中已不存在的文件视为孤儿，备份后删除；不在 manifest 中的文件视为用户
自定义，保留不动。详见 docs/adr/0012-scaffold-version-detection.md。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Scaffold Template 当前版本。仅在 src/paper_review/templates/ 内容实际变化时递增。
SCAFFOLD_VERSION = "0.2.0"

# manifest 文件名（相对 data_dir）
MANIFEST_FILENAME = ".scaffold-manifest"

# init 生成的 phase 子目录名（相对 pipelines/standard/）
PHASE_DIRS = ["pre-review", "review-pipeline", "post-review"]


def build_scaffold_files(templates_dir: Path) -> list[str]:
    """扫描 Scaffold Template，返回 init 将写入的全部文件（相对 data_dir 的路径）。

    映射关系（Scaffold Template → Pipelines Directory）：
      ``config.yaml``            → ``{data_dir}/config.yaml``
      ``pipeline.yaml``          → ``{data_dir}/pipelines/standard/pipeline.yaml``
      ``{phase}/*.py|*.md``      → ``{data_dir}/pipelines/standard/{phase}/*``
    """
    files: list[str] = []
    if (templates_dir / "config.yaml").is_file():
        files.append("config.yaml")
    if (templates_dir / "pipeline.yaml").is_file():
        files.append("pipelines/standard/pipeline.yaml")
    for phase in PHASE_DIRS:
        src = templates_dir / phase
        if not src.is_dir():
            continue
        for f in sorted(src.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                files.append(f"pipelines/standard/{phase}/{f.name}")
    return sorted(files)


def load_manifest(data_dir: Path) -> dict | None:
    """读取 manifest。不存在或损坏时返回 None。"""
    path = data_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("scaffold manifest 损坏，忽略: %s (%s)", path, e)
        return None
    if not isinstance(data, dict) or "version" not in data:
        logger.warning("scaffold manifest 结构无效（缺 version），忽略: %s", path)
        return None
    return data


def write_manifest(data_dir: Path, files: list[str]) -> None:
    """写入 manifest（当前版本 + 脚手架文件清单）。"""
    path = data_dir / MANIFEST_FILENAME
    payload = {"version": SCAFFOLD_VERSION, "files": sorted(files)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_scaffold(data_dir: Path) -> str:
    """对比 manifest 版本与当前 SCAFFOLD_VERSION。

    Returns:
        ``"ok"``       — 版本一致，或 data_dir 无脚手架生成的 standard 管线。
        ``"missing"``  — 有 pipelines/standard/ 但无 manifest：旧快照/残留。
        ``"outdated"`` — manifest 版本 != 当前 SCAFFOLD_VERSION。
    """
    manifest = load_manifest(data_dir)
    if manifest is None:
        # 脚手架只生成 pipelines/standard/；用户自定义管线（非 standard）不视为漂移。
        if (data_dir / "pipelines" / "standard").is_dir():
            return "missing"
        return "ok"
    if manifest.get("version") != SCAFFOLD_VERSION:
        return "outdated"
    return "ok"


def find_orphan_files(data_dir: Path, templates_dir: Path) -> list[Path]:
    """孤儿文件：Pipelines Directory 中、当前 Scaffold Template 已不存在的文件。

    - 有 manifest：精准判断——manifest 记录、但模板已无的文件。
    - 无 manifest（旧快照首次升级）：退化为无差别扫描——phase 目录里实际存在、
      模板没有的 .py/.md 文件。用户自定义文件也会被列为潜在孤儿，但备份可恢复，
      且 ``init --reset`` 交互确认时会逐个列出。

    仅 ``init --reset`` 时调用。返回绝对路径列表，按路径排序。
    """
    manifest = load_manifest(data_dir)
    current = set(build_scaffold_files(templates_dir))

    if manifest is not None:
        recorded = set(manifest.get("files", []))
        orphans_rel = recorded - current
    else:
        # 无 manifest：扫描 phase 目录，找出模板没有的 .py/.md 文件
        orphans_rel: set[str] = set()
        pipeline_dir = data_dir / "pipelines" / "standard"
        for phase in PHASE_DIRS:
            target = pipeline_dir / phase
            if not target.is_dir():
                continue
            for f in target.iterdir():
                if f.is_file() and f.suffix in (".py", ".md") and not f.name.startswith("."):
                    rel = f"pipelines/standard/{phase}/{f.name}"
                    if rel not in current:
                        orphans_rel.add(rel)

    # 防御性检查：manifest 可能被手工编辑，拒绝越出 data_dir 的路径。
    dd = data_dir.resolve()
    result: list[Path] = []
    for rel in sorted(orphans_rel):
        candidate = (data_dir / rel).resolve()
        if candidate == dd or dd in candidate.parents:
            result.append(data_dir / rel)
        else:
            logger.warning("manifest 记录了越出 data_dir 的路径，忽略: %s", rel)
    return result
