"""
08-summarize.py — 间接标准修正直接标准，输出最终汇总评分

修正矩阵注释：
  间接标准各维度偏离中性值（3）时，按系数修正直接标准各维度：
  - 间接分 < 3（偏离1-2单位）：惩罚（扣分）
  - 间接分 > 3（偏离1-2单位）：奖励（加分）
  - 间接分 = 3：不影响

  修正量 = (indirect_score - 3) × coefficient
    coefficient 取惩罚系数（indirect<3时）或奖励系数（indirect>3时）

  多个间接维度对同一直接维度的修正累加，但单个直接维度的总修正量
  上限为 ±1.5。最终分数 clamp 在 [1, 5]。

  ┌──────────────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
  │ 间接 ↓ → 直接 → │ 创新性 │ 质量   │ 效能   │ 风险   │ 难度   │ 业务价值 │
  ├──────────────────┼────────┼────────┼────────┼────────┼────────┼──────────┤
  │ 行文严谨性       │-.8/+.2 │-.4/+.1 │-.4/+.1 │-.3/+.1 │-.3/+.1 │ -.4/+.1  │
  │ 问题关键性       │-1./+.3 │-.5/+.2 │-.3/+.1 │  0/ 0  │-.3/+.1 │-1.5/+.5  │
  │ 公式堆砌度       │-.8/+.3 │-.2/+.1 │-.2/ 0  │  0/ 0  │-.5/+.4 │ -.2/+.1  │
  │ 源码深度         │-.5/+.4 │-.4/+.3 │-.3/+.2 │  0/ 0  │-1./+.8 │ -.2/+.1  │
  │ 业务规模真实性   │-.2/ 0  │-.4/+.2 │-.8/+.4 │  0/ 0  │-.2/+.1 │-1.5/+.8  │
  │ 前人调研充分度   │-1.2/+.4│-.3/+.1 │-.2/+.1 │  0/ 0  │-.4/+.2 │ -.3/+.1  │
  └──────────────────┴────────┴────────┴────────┴────────┴────────┴──────────┘
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# ============================================================================
# 修正矩阵
# ============================================================================

CORRECTION_MATRIX: dict[str, dict[str, tuple[float, float]]] = {
    "行文严谨性": {
        "创新性": (-0.8, 0.2),
        "质量提升效果": (-0.4, 0.1),
        "效能提升效果": (-0.4, 0.1),
        "风险敏感性": (-0.3, 0.1),
        "难度": (-0.3, 0.1),
        "业务价值提升效果": (-0.4, 0.1),
    },
    "问题关键性": {
        "创新性": (-1.0, 0.3),
        "质量提升效果": (-0.5, 0.2),
        "效能提升效果": (-0.3, 0.1),
        "风险敏感性": (0.0, 0.0),
        "难度": (-0.3, 0.1),
        "业务价值提升效果": (-1.5, 0.5),
    },
    "公式堆砌度": {
        "创新性": (-0.8, 0.3),
        "质量提升效果": (-0.2, 0.1),
        "效能提升效果": (-0.2, 0.0),
        "风险敏感性": (0.0, 0.0),
        "难度": (-0.5, 0.4),
        "业务价值提升效果": (-0.2, 0.1),
    },
    "源码深度": {
        "创新性": (-0.5, 0.4),
        "质量提升效果": (-0.4, 0.3),
        "效能提升效果": (-0.3, 0.2),
        "风险敏感性": (0.0, 0.0),
        "难度": (-1.0, 0.8),
        "业务价值提升效果": (-0.2, 0.1),
    },
    "业务规模真实性": {
        "创新性": (-0.2, 0.0),
        "质量提升效果": (-0.4, 0.2),
        "效能提升效果": (-0.8, 0.4),
        "风险敏感性": (0.0, 0.0),
        "难度": (-0.2, 0.1),
        "业务价值提升效果": (-1.5, 0.8),
    },
    "前人调研充分度": {
        "创新性": (-1.2, 0.4),
        "质量提升效果": (-0.3, 0.1),
        "效能提升效果": (-0.2, 0.1),
        "风险敏感性": (0.0, 0.0),
        "难度": (-0.4, 0.2),
        "业务价值提升效果": (-0.3, 0.1),
    },
}

DIRECT_DIMS = ["创新性", "质量提升效果", "效能提升效果", "风险敏感性", "难度", "业务价值提升效果"]
INDIRECT_DIMS = [
    "行文严谨性",
    "问题关键性",
    "公式堆砌度",
    "源码深度",
    "业务规模真实性",
    "前人调研充分度",
]

# Agent 可能输出与标准名有细微差异的维度名（如 "源码研究深度" → "源码深度"）
_DIM_ALIASES: dict[str, str] = {
    "源码研究深度": "源码深度",
    "问题识别关键性": "问题关键性",
}

MAX_CORRECTION_PER_DIM = 1.5


# ============================================================================
# 核心函数
# ============================================================================


def clamp(value: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, value))


def _normalize_dim(name: str) -> str:
    return _DIM_ALIASES.get(name, name)


def apply_correction(
    original_scores: dict[str, float],
    indirect_scores: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """应用修正矩阵，返回 (final_scores, corrections)。"""
    final: dict[str, float] = {}
    corrections: dict[str, float] = {}

    for dim in DIRECT_DIMS:
        orig = original_scores.get(dim, 3.0)
        total_correction = 0.0

        for indirect_dim, mapping in CORRECTION_MATRIX.items():
            ind_score = indirect_scores.get(indirect_dim, 3.0)
            penalty, reward = mapping.get(dim, (0.0, 0.0))
            delta = ind_score - 3.0
            if delta < 0:
                total_correction += delta * abs(penalty)
            elif delta > 0:
                total_correction += delta * reward

        total_correction = clamp(total_correction, -MAX_CORRECTION_PER_DIM, MAX_CORRECTION_PER_DIM)
        final[dim] = clamp(orig + total_correction)
        corrections[dim] = round(total_correction, 2)

    return final, corrections


def _extract_scores_from_markdown(md: str) -> dict[str, float]:
    """从 Markdown 表格中提取 维度→分数 映射。

    支持: | 维度名 | 4 |, | **维度名** | **3/5** |, | 维度名 | 3.0/10 | 等。
    分数含 / 时归一化到 1-5 范围。
    """
    scores: dict[str, float] = {}
    pat = re.compile(
        r"\|\s*\*{0,2}([^*|]+?)\*{0,2}\s*\|"
        r"\s*\*{0,2}([\d.]+(?:/\s*[\d.]+)?)\*{0,2}\s*\|"
    )

    for m in pat.finditer(md):
        dim = _normalize_dim(m.group(1).strip())
        score_str = m.group(2).strip()
        try:
            if "/" in score_str:
                num, denom = score_str.split("/")
                score = float(num) / float(denom) * 5.0
            else:
                score = float(score_str)
        except (ValueError, ZeroDivisionError):
            continue
        scores[dim] = round(score, 1)

    return scores


def _parse_scoring_output(data: dict) -> dict[str, float]:
    """从前序 scoring step 的 data 中提取 {维度: 分数}。

    兼容:
    1) {"创新性": 4, ...}
    2) {"创新性": {"score": 4}, ...}
    3) {"raw_output": "| 维度 | 分数 |\\n| 创新性 | 4 |..."}
    """
    scores: dict[str, float] = {}
    for key, value in data.items():
        if key == "raw_output":
            continue
        if isinstance(value, dict) and "score" in value:
            try:
                scores[_normalize_dim(key)] = float(value["score"])
            except (ValueError, TypeError):
                raise ValueError(f"Cannot parse score for '{key}': {value['score']!r}") from None
        elif isinstance(value, (int, float)):
            # pi-lens-ignore: unchecked-throwing-call-python
            scores[_normalize_dim(key)] = float(value)

    if not scores and "raw_output" in data:
        scores = _extract_scores_from_markdown(data["raw_output"])

    return scores


def _extract_rationale_from_markdown(md: str, dims: list[str]) -> dict[str, str]:
    """从 Markdown 表格第三列提取 维度→理由 文字。

    表格形如 | 维度 | 分数 | 理由/依据 |，第三列为评分理由。
    """
    rationale: dict[str, str] = {}
    pat = re.compile(
        r"\|\s*\*{0,2}([^*|\n]+?)\*{0,2}\s*\|"
        r"\s*\*{0,2}([\d.]+(?:/\s*[\d.]+)?)\*{0,2}\s*\|"
        r"\s*([^|\n]*?)\s*\|"
    )
    for m in pat.finditer(md):
        dim = _normalize_dim(m.group(1).strip())
        reason = m.group(3).strip()
        if dim in dims and reason:
            rationale[dim] = reason
    return rationale


def _parse_scoring_rationale(data: dict, dims: list[str]) -> dict[str, str]:
    """从前序 scoring step 的 data 提取 {维度: rationale 文字}。

    兼容:
    1) {"创新性": {"score": 4, "rationale": "..."}}
    2) {"raw_output": "| 维度 | 分数 | 理由 |"}
    """
    rationale: dict[str, str] = {}
    for key, value in data.items():
        if key == "raw_output":
            continue
        if isinstance(value, dict) and "rationale" in value:
            r = value.get("rationale")
            if isinstance(r, str) and r.strip():
                rationale[_normalize_dim(key)] = r.strip()
    if not rationale and "raw_output" in data:
        rationale = _extract_rationale_from_markdown(data["raw_output"], dims)
    return rationale


def _parse_tags(data: dict) -> list[str]:
    """从 scoring step 的 data 提取 tags 列表（缺失/非法 → 空列表）。"""
    tags = data.get("tags")
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if isinstance(t, str) and t.strip()]
    return []


# ============================================================================
# 主逻辑
# ============================================================================


def main():
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")
    intermediates_dir = os.environ.get("PIPELINE_INTERMEDIATES", ".")
    subject_name = os.environ.get("PIPELINE_SUBJECT", "")

    base = Path(intermediates_dir) / subject_name
    direct_path = base / "06-direct-scoring" / "output.json"
    indirect_path = base / "07-indirect-scoring" / "output.json"

    original_direct: dict[str, float] = {}
    indirect_scores: dict[str, float] = {}
    direct_data: dict = {}
    direct_rationale: dict[str, str] = {}
    tags: list[str] = []

    if direct_path.exists():
        # pi-lens-ignore: unchecked-throwing-call-python
        with open(direct_path) as f:
            # pi-lens-ignore: unchecked-throwing-call-python
            direct_data = json.load(f).get("data", {})
        original_direct = _parse_scoring_output(direct_data)
        direct_rationale = _parse_scoring_rationale(direct_data, DIRECT_DIMS)
        tags = _parse_tags(direct_data)
    else:
        print(f"Warning: {direct_path} not found, using defaults")

    if indirect_path.exists():
        # pi-lens-ignore: unchecked-throwing-call-python
        with open(indirect_path) as f:
            # pi-lens-ignore: unchecked-throwing-call-python
            indirect_scores = _parse_scoring_output(json.load(f).get("data", {}))
    else:
        print(f"Warning: {indirect_path} not found, using defaults")

    # 补全缺失维度默认值
    for dim in DIRECT_DIMS:
        if dim not in original_direct:
            original_direct[dim] = 3.0
    for dim in INDIRECT_DIMS:
        if dim not in indirect_scores:
            indirect_scores[dim] = 3.0

    # 应用修正
    final_scores, corrections = apply_correction(original_direct, indirect_scores)

    # 字段级降级标记（本次只判「有没有」：rationale/tags 是否缺失，不判内容空洞）
    rationale_missing = [dim for dim in DIRECT_DIMS if not direct_rationale.get(dim)]
    tags_missing = not tags
    degraded = bool(rationale_missing or tags_missing)
    degradation_reason = ""
    if degraded:
        parts = []
        if rationale_missing:
            parts.append("缺证据:" + "/".join(rationale_missing))
        if tags_missing:
            parts.append("tags缺失")
        degradation_reason = "⚠ " + "；".join(parts)

    # 输出
    data = {
        "final_scores": {dim: round(final_scores[dim], 1) for dim in DIRECT_DIMS},
        "corrections": {dim: corrections[dim] for dim in DIRECT_DIMS},
        "indirect_scores": {dim: indirect_scores.get(dim, 3) for dim in INDIRECT_DIMS},
        "original_direct_scores": {dim: original_direct.get(dim, 3) for dim in DIRECT_DIMS},
        "rationale": {dim: direct_rationale.get(dim, "") for dim in DIRECT_DIMS},
        "tags": tags,
        "evidence": {
            "rationale_missing": rationale_missing,
            "tags_missing": tags_missing,
        },
        "degraded": degraded,
        "degradation_reason": degradation_reason,
    }
    output = {"step": "08-summarize", "status": "ok", "error": None, "data": data}

    # pi-lens-ignore: unchecked-throwing-call-python
    os.makedirs(step_dir, exist_ok=True)
    # pi-lens-ignore: unchecked-throwing-call-python
    with open(os.path.join(step_dir, "output.json"), "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("08-summarize: correction applied")
    for dim in DIRECT_DIMS:
        corr = corrections[dim]
        orig = original_direct.get(dim, 3)
        fin = final_scores[dim]
        tag = f"(Δ{corr:+.1f})" if abs(corr) > 0.01 else "(unchanged)"
        print(f"  {dim}: {orig} → {fin} {tag}")


if __name__ == "__main__":
    main()
