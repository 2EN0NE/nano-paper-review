"""
模板引擎 —— .md Agent 步骤的变量替换与前缀生成

变量语法: {variable.name} 或 {intermediates.STEPNAME.data.KEY}
替换在所有 .md 文件提交给 pi 之前完成。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

_VARIABLE_PATTERN = re.compile(r"\{([a-zA-Z_0-9][\w.-]*)\}")


@dataclass
class PriorStepOutput:
    """前序步骤的 output.json 信息（用于变量替换）。"""

    step_name: str
    status: str
    data: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TemplateContext:
    """模板变量解析的上下文。"""

    # Subject 信息
    subject_name: str = ""
    subject_path: str = ""
    subject_text: str = ""
    subject_meta: str = "{}"

    # 路径信息
    output_dir: str = ""
    intermediates_dir: str = ""
    step_dir: str = ""
    reports_dir: str = ""

    # 前序步骤输出（用于 intermediates.xxx 变量）
    prior_step_outputs: dict[str, dict] = field(default_factory=dict)


def resolve_variables(template: str, ctx: TemplateContext) -> str:
    """将模板中的变量替换为实际值。

    支持的变量（按优先级）：
    1. subject.name, subject.path, subject.text, subject.meta
    2. output_dir, intermediates_dir, step_dir, reports_dir
    3. intermediates.STEPNAME.output  — 整份 output.json
    4. intermediates.STEPNAME.data.KEY — data.KEY 字段
    5. intermediates.STEPNAME.status   — status 字段
    """

    def _replacer(m: re.Match) -> str:
        var_path = m.group(
            1
        )  # e.g. "subject.name" or "intermediates.01-search.data.refs"
        parts = var_path.split(".")

        # Subject 变量
        if parts[0] == "subject":
            key = parts[1] if len(parts) > 1 else ""
            return _resolve_subject_var(key, ctx)

        # 路径变量（单段）
        if len(parts) == 1:
            return _resolve_path_var(parts[0], ctx)

        # intermediates 变量
        if parts[0] == "intermediates" and len(parts) >= 2:
            step_name = parts[1]  # e.g. "01-search"
            step_output = ctx.prior_step_outputs.get(step_name)
            if step_output is None:
                return m.group(0)  # 原样保留

            if len(parts) == 2:
                # {intermediates.01-search} — 整份 output（兼容短格式）
                return json.dumps(step_output, ensure_ascii=False)

            # 深入 data、status 或 output
            rest = parts[2:]  # e.g. ["data", "references"] or ["output"] or ["status"]
            if rest == ["output"] or not rest:
                return json.dumps(step_output, ensure_ascii=False)
            if rest == ["status"]:
                return str(step_output.get("status", ""))
            elif rest and rest[0] == "data":
                data = step_output.get("data", {})
                if len(rest) == 1:
                    return json.dumps(data, ensure_ascii=False)
                # 深入 data.key.subkey
                current: Any = data
                for k in rest[1:]:
                    if isinstance(current, dict) and k in current:
                        current = current[k]
                    else:
                        return m.group(0)
                if isinstance(current, (dict, list)):
                    return json.dumps(current, ensure_ascii=False)
                return str(current)
            else:
                return m.group(0)

        return m.group(0)  # 未知变量原样保留

    return _VARIABLE_PATTERN.sub(_replacer, template)


def _resolve_subject_var(key: str, ctx: TemplateContext) -> str:
    mapping = {
        "name": ctx.subject_name,
        "path": ctx.subject_path,
        "text": ctx.subject_text,
        "meta": ctx.subject_meta,
    }
    val = mapping.get(key)
    if val is None:
        return f"{{subject.{key}}}"
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _resolve_path_var(key: str, ctx: TemplateContext) -> str:
    mapping = {
        "output_dir": ctx.output_dir,
        "intermediates_dir": ctx.intermediates_dir,
        "step_dir": ctx.step_dir,
        "reports_dir": ctx.reports_dir,
    }
    val = mapping.get(key)
    if val is None:
        return f"{{{key}}}"
    return val


def build_agent_prefix(
    step_name: str,
    step_dir: str,
    prior_results: list,
) -> str:
    """构建 Agent 步骤的前缀。

    包含：
    1. 步骤标识（当前是第几步）
    2. 前序步骤摘要
    3. 输出路径约束
    4. output.json 格式要求

    Args:
        step_name: 当前步骤名。
        step_dir: 当前步骤的 intermediates 目录（绝对路径）。
        prior_results: 前序步骤的 StepResult 列表。

    Returns:
        前缀文本。
    """
    parts: list[str] = []

    parts.append(
        f"你正在执行评审流水线中的第 {_guess_step_number(step_name)} 个步骤：{step_name}。"
    )
    parts.append("")

    if prior_results:
        parts.append("## 前序步骤的中间产物")
        parts.append(
            "以下是你前面步骤产出的 output.json 内容摘要。你在需要时可以自行查阅这些信息。"
        )
        parts.append("")
        for r in prior_results:
            status_icon = (
                "✅" if r.status == "ok" else "⚠️" if r.status == "skipped" else "❌"
            )
            parts.append(f"- **{r.step_name}** {status_icon} status=`{r.status}`")
            if r.data:
                snippet = json.dumps(r.data, ensure_ascii=False)
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                parts.append(f"  data: {snippet}")
        parts.append("")

    parts.append("## 本步骤的约束")
    parts.append(f"- 你的输出应写入以下目录：**{step_dir}**")
    parts.append("- 输出文件名应为 `output.json`，格式如下：")
    parts.append("  ```json")
    parts.append(
        '  {"step": "'
        + step_name
        + '", "status": "ok|error|skipped", "error": null|"原因", "data": {...}}'
    )
    parts.append("  ```")
    parts.append(
        "- `status` 字段：`ok`（成功）、`error`（执行失败）、`skipped`（跳过）"
    )
    parts.append(
        "- 如果信息不足以完成评审，将 status 设为 `skipped` 并在 error 中写明原因。"
    )
    parts.append("")

    parts.append("---")
    parts.append("")
    parts.append("## 本步骤的评审规则")
    parts.append("")

    return "\n".join(parts)


def _guess_step_number(step_name: str) -> str:
    """尝试从前缀数字猜测步骤序号。"""
    if "-" in step_name:
        parts = step_name.split("-", 1)
        try:
            return str(int(parts[0]))
        except ValueError:
            pass
    return "?"
