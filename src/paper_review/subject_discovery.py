"""
from __future__ import annotations

Subject 发现模块

从 manifest JSON 或 CLI PDF 扫描发现 Subject 列表，
应用去重和排序策略。

Public API: discover_subjects()
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from paper_review.logging_config import get_logger
from paper_review.pipeline_models import PipelineConfig, _order_subjects

logger = get_logger("orchestrator")

_PARSE_TIME_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


# ============================================================================
# 公开 API
# ============================================================================


def discover_subjects(
    config: PipelineConfig,
    input_path: Path,
    output_dir: Path,
) -> list[str]:
    """从 PipelineConfig 和输入路径发现 Subject 列表。

    流程：
    1. 找到第一个 per_subject 阶段
    2. 如果 subject_source.type == "manifest"，从 manifest JSON 加载
    3. 否则从 CLI 输入路径扫描 PDF
    4. 应用 duplicate_policy 去重
    5. 应用 subject_order 排序
    """
    per_subject_phases = [p for p in config.phases if p.mode == "per_subject"]
    first_per_subject = per_subject_phases[0] if per_subject_phases else None

    # Subject 来源
    if first_per_subject and first_per_subject.subject_source:
        use_manifest = first_per_subject.subject_source.type == "manifest"
    else:
        use_manifest = False

    if use_manifest and first_per_subject:
        manifest_path_str = _resolve_config_vars(
            first_per_subject.subject_source.path,  # type: ignore[union-attr]
            output_dir,
        )
        manifest_data = _load_manifest(Path(manifest_path_str))
        if manifest_data:
            subjects_data = manifest_data.get("subjects", [])
            if subjects_data:
                raw_subjects = [s["name"] for s in subjects_data]
            else:
                logger.warning("Manifest contains 0 subjects — falling back to CLI discovery")
                raw_subjects = _scan_pdfs(input_path)
        else:
            logger.info("Falling back to CLI subject discovery from %s", input_path)
            raw_subjects = _scan_pdfs(input_path)
    else:
        raw_subjects = _scan_pdfs(input_path)

    # 去重
    dup_policy = first_per_subject.duplicate_policy if first_per_subject else "skip"
    raw_subjects = _apply_duplicate_policy(raw_subjects, dup_policy)

    # 排序
    if first_per_subject and first_per_subject.subject_order:
        subjects = _order_subjects(raw_subjects, first_per_subject.subject_order)
    else:
        subjects = raw_subjects

    return subjects


# ============================================================================
# 内部函数
# ============================================================================


def _load_manifest(manifest_path: Path) -> dict | None:
    """加载 subject-manifest.json。"""
    if not manifest_path.exists():
        logger.info("Manifest not found at %s — will fall back to CLI discovery", manifest_path)
        return None
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        logger.info("Loaded manifest: %d subjects", len(data.get("subjects", [])))
        return data
    except json.JSONDecodeError as e:
        logger.error("Manifest file exists at %s but JSON is corrupt: %s", manifest_path, e)
        return None
    except OSError as e:
        logger.error("Failed to read manifest at %s: %s", manifest_path, e)
        return None


def _resolve_config_vars(value: str, output_dir: Path) -> str:
    """解析配置中的 {{ output_dir }} 等变量。"""
    var_map = {"output_dir": str(output_dir.absolute())}
    return _resolve_parse_time_vars(value, var_map)


def _resolve_parse_time_vars(value: str, var_map: dict[str, str]) -> str:
    """替换字符串中的 {{ var_name }} 模板变量。"""

    def _replacer(m: re.Match) -> str:
        var_name = m.group(1)
        if var_name in var_map:
            return var_map[var_name]
        logger.warning("Unknown template variable '{{ %s }}' in '%s' — left as-is", var_name, value)
        return m.group(0)

    return _PARSE_TIME_VAR.sub(_replacer, value)


def _scan_pdfs(input_path: Path) -> list[str]:
    """从 CLI 输入路径扫描 PDF。"""
    if input_path.is_dir():
        return sorted(f.stem for f in input_path.iterdir() if f.is_file() and f.suffix == ".pdf")
    return [input_path.stem]


def _apply_duplicate_policy(subjects: list[str], policy: str) -> list[str]:
    """按 duplicate_policy 处理同名 Subject。"""
    if policy == "error":
        seen: set[str] = set()
        for s in subjects:
            if s in seen:
                raise ValueError(f"Duplicate subject '{s}' detected with duplicate_policy=error")
            seen.add(s)
        return subjects

    if policy == "rename":
        result: list[str] = []
        counter: dict[str, int] = {}
        for s in subjects:
            if s in counter:
                counter[s] += 1
                result.append(f"{s}-{counter[s]}")
            else:
                counter[s] = 0
                result.append(s)
        return result

    # skip（默认）
    if policy != "skip":
        logger.warning(
            "Unknown duplicate_policy '%s' — falling back to 'skip'. Valid: skip, rename, error",
            policy,
        )

    result: list[str] = []
    seen: set[str] = set()
    for s in subjects:
        if s in seen:
            continue
        seen.add(s)
        result.append(s)
    return result
