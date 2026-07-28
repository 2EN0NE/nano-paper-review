"""
管线 Steps 执行器

负责执行单个 Pipeline Step：
- .py → subprocess.run([python, step_path])
- .md → subprocess.run([pi, -p, @prompt_file])
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from paper_review.logging_config import get_logger
from paper_review.pipeline_models import StepFile, StepResult

logger = get_logger("orchestrator")


def _run_py_step(step: StepFile, step_dir: Path, env: dict) -> StepResult:
    """执行一个 .py 步骤。"""
    logger.debug("Running .py step: %s", step.stem)

    os.makedirs(step_dir, exist_ok=True)
    step_env = {**env, "PIPELINE_STEP_DIR": str(step_dir)}

    try:
        proc = subprocess.run(
            [sys.executable, str(step.path)],
            env=step_env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        logger.error("Step %s timed out", step.stem)
        return StepResult(
            step_name=step.stem,
            status="error",
            error="Timed out after 600s",
        )

    if proc.returncode != 0:
        error_msg = proc.stderr.strip() or f"exit code {proc.returncode}"
        logger.warning("Step %s failed: %s", step.stem, error_msg)

        # 即使失败也检查是否有 output.json
        output_file = step_dir / "output.json"
        if output_file.exists():
            try:
                with open(output_file) as f:
                    data = json.load(f)
                return StepResult(
                    step_name=step.stem,
                    status="error",
                    error=error_msg,
                    data=data.get("data", {}),
                )
            except (OSError, json.JSONDecodeError):
                pass

        return StepResult(
            step_name=step.stem,
            status="error",
            error=error_msg,
        )

    # 成功 — 读取 output.json
    output_file = step_dir / "output.json"
    data: dict = {}
    if output_file.exists():
        try:
            with open(output_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to parse output.json for %s: %s", step.stem, e)

    step_status = data.get("status", "ok")
    return StepResult(
        step_name=step.stem,
        status=step_status,
        data=data.get("data", {}),
    )


def _run_md_step(
    step: StepFile,
    step_dir: Path,
    env: dict,
    prior_results: list[StepResult],
    subject_name: str = "",
    subject_text: str = "",
    subject_meta: str = "{}",
) -> StepResult:
    """执行一个 .md Agent 步骤（通过 subprocess 调用 pi）。

    流程：
    1. 读取 .md 文件内容
    2. 构建 TemplateContext，解析模板变量
    3. 构建 Agent 前缀
    4. 拼接前缀 + 已替换的 prompt → 写入临时 .md 文件
    5. subprocess.run(["pi", "-p", "@prompt_file.md"])
    6. 解析 stdout → output.json
    """
    logger.debug("Running .md step: %s", step.stem)

    os.makedirs(step_dir, exist_ok=True)
    step_env = {**env, "PIPELINE_STEP_DIR": str(step_dir)}

    # 1. 读取 .md 文件内容
    md_content = step.path.read_text(encoding="utf-8")

    # 2. 构建变量替换上下文
    prior_step_outputs: dict[str, dict] = {}
    for r in prior_results:
        prior_step_outputs[r.step_name] = {
            "step": r.step_name,
            "status": r.status,
            "error": r.error,
            "data": r.data,
        }

    from paper_review.template_engine import TemplateContext, resolve_variables

    ctx = TemplateContext(
        subject_name=subject_name,
        subject_text=subject_text,
        subject_meta=subject_meta,
        step_dir=str(step_dir),
        intermediates_dir=env.get("PIPELINE_INTERMEDIATES", ""),
        output_dir=env.get("PIPELINE_OUTPUT_DIR", ""),
        prior_step_outputs=prior_step_outputs,
    )

    resolved_md = resolve_variables(md_content, ctx)

    # 3. 构建 Agent 前缀
    from paper_review.template_engine import build_agent_prefix

    prefix = build_agent_prefix(
        step_name=step.stem,
        step_dir=str(step_dir),
        prior_results=prior_results,
    )

    # 4. 拼接 final prompt → 写入临时文件（避免 CLI 参数过长 + shell 转义问题）
    final_prompt = prefix + resolved_md
    prompt_file = step_dir / "prompt.md"
    prompt_file.write_text(final_prompt, encoding="utf-8")

    # 5. 调用 pi（-p 非交互模式）
    pi_binary = env.get("PIPELINE_PI_BINARY", "pi")
    try:
        proc = subprocess.run(
            [pi_binary, "-p", f"@{prompt_file}"],
            env=step_env,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        logger.error("Agent step %s timed out (900s)", step.stem)
        return StepResult(
            step_name=step.stem,
            status="error",
            error="Agent step timed out (900s)",
        )
    except FileNotFoundError:
        # pi 不在 PATH 中 — 降级为 skipped
        logger.warning(
            "pi binary not found at '%s'; marking step %s as skipped",
            pi_binary,
            step.stem,
        )
        output_file = step_dir / "output.json"
        placeholder = {
            "step": step.stem,
            "status": "skipped",
            "error": f"pi binary '{pi_binary}' not found",
            "data": {},
        }
        with open(output_file, "w") as f:
            json.dump(placeholder, f)
        return StepResult(
            step_name=step.stem,
            status="skipped",
            error=f"pi binary '{pi_binary}' not found",
        )

    # 6. 检查 pi 是否成功执行
    output_file = step_dir / "output.json"
    if proc.returncode != 0:
        stderr_tail = proc.stderr.strip()[-500:] if proc.stderr.strip() else "<no stderr>"
        logger.error(
            "Agent step %s failed (exit %d): %s",
            step.stem,
            proc.returncode,
            stderr_tail,
        )
        output_json = {
            "step": step.stem,
            "status": "error",
            "error": f"pi exited with code {proc.returncode}: {stderr_tail}",
            "data": {},
        }
        with open(output_file, "w") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        return StepResult(
            step_name=step.stem,
            status="error",
            error=output_json["error"],
        )

    # 7. 处理 stdout → output.json
    output_text = proc.stdout.strip()

    # 7. 解析 stdout 作为 JSON
    try:
        parsed = json.loads(output_text) if output_text else {}
        if not isinstance(parsed, dict):
            parsed = {"raw_output": output_text}
    except json.JSONDecodeError:
        # stdout 不是纯 JSON — 包装成 data.raw_output
        parsed = {
            "step": step.stem,
            "status": "ok",
            "data": {"raw_output": output_text},
        }

    # 确保 output.json 包含必要字段
    output_json = {
        "step": parsed.get("step", step.stem),
        "status": parsed.get("status", "ok"),
        "error": parsed.get("error"),
        "data": parsed.get("data", {}),
    }
    with open(output_file, "w") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    result_status = output_json["status"]
    return StepResult(
        step_name=step.stem,
        status=result_status,
        data=output_json["data"],
    )


def _run_step(
    step: StepFile,
    step_dir: Path,
    env: dict,
    prior_results: list[StepResult] | None = None,
    subject_name: str = "",
    subject_text: str = "",
    subject_meta: str = "{}",
) -> StepResult:
    """根据步骤类型分派执行。"""
    if step.step_type == "py":
        return _run_py_step(step, step_dir, env)
    else:
        return _run_md_step(
            step,
            step_dir,
            env,
            prior_results=prior_results or [],
            subject_name=subject_name,
            subject_text=subject_text,
            subject_meta=subject_meta,
        )
