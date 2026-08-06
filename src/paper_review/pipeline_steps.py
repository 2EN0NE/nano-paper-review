"""
管线 Steps 执行器

负责执行单个 Pipeline Step：
- .py → PyStepRunner（runpy.run_path() 进程内执行）
- .md → MdStepExecutor（PromptBuilder + AgentRunner，subprocess 调用 pi）

提供 StepExecutor Protocol 作为测试 seam。
"""

from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import threading
from pathlib import Path
from typing import Protocol

from paper_review.logging_config import get_logger
from paper_review.pipeline_models import StepFile, StepResult

logger = get_logger("orchestrator")

# 进程内执行 .py 步骤时的环境变量互斥锁（Worker 池并发时保护 os.environ）
_py_step_lock = threading.Lock()


# ============================================================================
# StepExecutor — 核心 seam
# ============================================================================


class StepExecutor(Protocol):
    """步骤执行器的协议。

    生产实现：PyStepRunner / MdStepExecutor。
    测试实现：InMemoryExecutor。
    """

    def execute(
        self,
        step: StepFile,
        step_dir: Path,
        env: dict,
        prior_results: list[StepResult],
        subject_name: str,
    ) -> StepResult: ...


# ============================================================================
# PyStepRunner — .py 步骤执行
# ============================================================================


class PyStepRunner:
    """通过 runpy.run_path() 在进程内执行 .py 步骤。

    Worker 池并发时使用 _py_step_lock 保护 os.environ 互斥访问。
    """

    def execute(
        self,
        step: StepFile,
        step_dir: Path,
        env: dict,
        prior_results: list[StepResult] | None = None,
        subject_name: str = "",
    ) -> StepResult:
        logger.info("  [.py] %s — starting", step.stem)

        os.makedirs(step_dir, exist_ok=True)
        step_env = {**os.environ, **env, "PIPELINE_STEP_DIR": str(step_dir)}

        with _py_step_lock:
            _prev = {k: os.environ.get(k) for k in step_env if k in os.environ}
            try:
                os.environ.update(step_env)
                runpy.run_path(str(step.path), run_name="__main__")
            except (Exception, SystemExit) as e:
                if isinstance(e, SystemExit):
                    logger.warning("  [.py] %s — exited with code %s", step.stem, e.code)
                else:
                    logger.warning("  [.py] %s — failed: %s", step.stem, e)
                error_msg = str(e)

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
            finally:
                for k, v in _prev.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                for k in step_env:
                    if k not in _prev:
                        os.environ.pop(k, None)

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
        logger.info("  [.py] %s — %s", step.stem, step_status)
        return StepResult(
            step_name=step.stem,
            status=step_status,
            data=data.get("data", {}),
        )


# ============================================================================
# PromptBuilder — .md 步骤的 prompt 构建
# ============================================================================


class PromptBuilder:
    """构建 .md Agent 步骤的完整 prompt（模板替换 + Agent 前缀）。

    纯计算，无 I/O。可在不运行 pi 的情况下单独测试。
    """

    def build(
        self,
        step: StepFile,
        prior_results: list[StepResult],
        subject_name: str = "",
        subject_text: str = "",
        subject_meta: str = "{}",
        intermediates_dir: str = "",
        output_dir: str = "",
        step_dir: str = "",
    ) -> str:
        """读取 .md 文件，解析模板变量，拼接 Agent 前缀，返回完整 prompt。"""
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
            step_dir=step_dir,
            intermediates_dir=intermediates_dir,
            output_dir=output_dir,
            prior_step_outputs=prior_step_outputs,
        )

        resolved_md = resolve_variables(md_content, ctx)

        # 3. 构建 Agent 前缀
        from paper_review.template_engine import build_agent_prefix

        prefix = build_agent_prefix(
            step_name=step.stem,
            step_dir=step_dir,
            prior_results=prior_results,
        )

        return prefix + resolved_md


# ============================================================================
# AgentRunner — .md 步骤的 subprocess 执行
# ============================================================================

_ANSI_ESCAPE_RE = __import__("re").compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\]8;.*?\x1b\\\\")


def _strip_ansi(text: str) -> str:
    """移除 ANSI 转义码，保留可读文本。"""
    return _ANSI_ESCAPE_RE.sub("", text)


def _classify_stderr_error(stderr_clean: str, step_timeout: int) -> str:
    """根据 pi stderr 内容识别具体错误类型，返回更精确的错误描述。

    返回的错误消息中会包含 429/503 状态码，供 DynamicPool 识别。
    """
    if not stderr_clean:
        return f"Agent step timed out ({step_timeout}s) — no output from pi"

    # API 认证不可用（视为 503）
    if "auth_unavailable" in stderr_clean or "no auth available" in stderr_clean:
        provider_info = ""

        m = re.search(r"providers=([^,)]+)", stderr_clean)
        if m:
            provider_info = f" (provider: {m.group(1)})"
        return f"API auth unavailable (503){provider_info} — check DeepSeek proxy credentials"

    # 429 速率限制（优先级高于 503，因为可能同时出现）
    if (
        "429" in stderr_clean
        or "rate_limit" in stderr_clean.lower()
        or "too many requests" in stderr_clean.lower()
    ):
        return f"API rate limited (429){_extract_error_message(stderr_clean)}"

    # 503 服务端错误
    if "503" in stderr_clean:
        return f"API server error (503){_extract_error_message(stderr_clean)}"

    # 兜底：截取 stderr 尾部作为参考
    stderr_tail = stderr_clean.strip()[-200:] if stderr_clean.strip() else ""
    if stderr_tail:
        return f"Agent step timed out ({step_timeout}s) — stderr tail: {stderr_tail}"
    return f"Agent step timed out ({step_timeout}s)"


def _extract_error_message(stderr_clean: str) -> str:
    """从 stderr 中提取 JSON 错误消息。"""

    m = re.search(r'"message"\s*:\s*"([^"]+)"', stderr_clean)
    if m:
        return f": {m.group(1)}"
    return ""


class AgentRunner:
    """通过 subprocess 调用 pi 执行 Agent 步骤。

    接收已构建的 prompt 字符串，写临时文件，调 pi，解析输出。
    """

    def run(
        self,
        prompt: str,
        step_stem: str,
        step_dir: Path,
        env: dict,
        timeout: int = 900,
    ) -> StepResult:
        """执行 Agent 步骤。

        Args:
            prompt: 已构建的完整 prompt（模板已替换 + Agent 前缀已拼接）。
            step_stem: 步骤名（用于日志和 output.json）。
            step_dir: 步骤 intermediates 目录。
            env: 环境变量。
            timeout: subprocess 超时秒数。
        """
        os.makedirs(step_dir, exist_ok=True)
        step_env = {**env, "PIPELINE_STEP_DIR": str(step_dir)}

        # 写入临时 prompt 文件（避免 CLI 参数过长 + shell 转义问题）
        prompt_file = step_dir / "prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")

        # 调用 pi（--no-session 一次性执行，-p 非交互模式）
        pi_binary = env.get("PIPELINE_PI_BINARY", "pi")
        step_timeout = timeout  # default
        timeout_str = env.get("PIPELINE_STEP_TIMEOUT", "")
        if timeout_str:
            try:
                val = int(timeout_str)
                if val > 0:
                    step_timeout = val
                else:
                    logger.warning(
                        "PIPELINE_STEP_TIMEOUT must be >0, got %s; using default %d",
                        timeout_str,
                        step_timeout,
                    )
            except ValueError:
                logger.warning(
                    "Invalid PIPELINE_STEP_TIMEOUT value: %s; using default %d",
                    timeout_str,
                    step_timeout,
                )

        prompt_size_kb = len(prompt) / 1024
        logger.info(
            "  [%s] ▶ pi %s --no-session -p @%s (%.1fKB, timeout=%ds)",
            step_stem,
            pi_binary,
            prompt_file,
            prompt_size_kb,
            step_timeout,
        )
        proc = None
        try:
            proc = subprocess.Popen(  # noqa: S603 — pi_binary is user-configurable
                [pi_binary, "--no-session", "-p", f"@{prompt_file}"],
                env=step_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout_data, stderr_data = proc.communicate(timeout=step_timeout)
            # 构造兼容 subprocess.run 返回值的 CompletedProcess
            result = subprocess.CompletedProcess(
                args=proc.args,
                returncode=proc.returncode,
                stdout=stdout_data or "",
                stderr=stderr_data or "",
            )
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
                stdout_data, stderr_data = proc.communicate()
            else:
                stdout_data, stderr_data = "", ""

            # 检查 stderr 中的已知错误模式，提供更精确的失败原因
            stderr_clean = _strip_ansi(stderr_data) if stderr_data else ""
            error_detail = _classify_stderr_error(stderr_clean, step_timeout)

            logger.error("Agent step %s timed out (%ds): %s", step_stem, step_timeout, error_detail)
            return StepResult(
                step_name=step_stem,
                status="error",
                error=error_detail,
            )
        except FileNotFoundError:
            logger.warning(
                "pi binary not found at '%s'; marking step %s as skipped",
                pi_binary,
                step_stem,
            )
            output_file = step_dir / "output.json"
            placeholder = {
                "step": step_stem,
                "status": "skipped",
                "error": f"pi binary '{pi_binary}' not found",
                "data": {},
            }
            with open(output_file, "w") as f:
                json.dump(placeholder, f)
            return StepResult(
                step_name=step_stem,
                status="skipped",
                error=f"pi binary '{pi_binary}' not found",
            )

        return self._parse_output(result, step_stem, step_dir)

    def _parse_output(
        self,
        proc: subprocess.CompletedProcess,
        step_stem: str,
        step_dir: Path,
    ) -> StepResult:
        """解析 pi 的 stdout/stderr，写入 output.json，返回 StepResult。"""
        output_file = step_dir / "output.json"

        if proc.returncode != 0:
            # 清理 pi TUI 的 ANSI 转义码，提取可读错误信息
            stderr_clean = _strip_ansi(proc.stderr) if proc.stderr else ""
            stderr_tail = stderr_clean.strip()[-500:] if stderr_clean.strip() else "<no stderr>"
            logger.error(
                "Agent step %s failed (exit %d): %s",
                step_stem,
                proc.returncode,
                stderr_tail,
            )
            output_json = {
                "step": step_stem,
                "status": "error",
                "error": f"pi exited with code {proc.returncode}: {stderr_tail}",
                "data": {},
            }
            with open(output_file, "w") as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)
            return StepResult(
                step_name=step_stem,
                status="error",
                error=output_json["error"],
            )

        output_text = proc.stdout.strip()

        try:
            parsed = json.loads(output_text) if output_text else {}
            if not isinstance(parsed, dict):
                parsed = {"raw_output": output_text}
        except json.JSONDecodeError:
            parsed = {
                "step": step_stem,
                "status": "ok",
                "data": {"raw_output": output_text},
            }

        output_json = {
            "step": parsed.get("step", step_stem),
            "status": parsed.get("status", "ok"),
            "error": parsed.get("error"),
            "data": parsed.get("data", {}),
        }
        with open(output_file, "w") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        result_status = output_json["status"]

        # 成功时也输出 pi 的 stdout 预览便于调试
        stdout_preview = output_text[:200].replace("\n", " ") if output_text else "<empty>"
        stderr_info = ""
        if proc.stderr and proc.stderr.strip():
            stderr_info = f", stderr={len(proc.stderr)}B"
        logger.info(
            "  [%s] ✓ pi done (exit=%d, stdout=%dB%s): %s…",
            step_stem,
            proc.returncode,
            len(output_text),
            stderr_info,
            stdout_preview,
        )

        return StepResult(
            step_name=step_stem,
            status=result_status,
            data=output_json["data"],
        )


# ============================================================================
# MdStepExecutor — .md 步骤执行（组合 PromptBuilder + AgentRunner）
# ============================================================================


class MdStepExecutor:
    """组合 PromptBuilder 和 AgentRunner，执行 .md Agent 步骤。"""

    def __init__(
        self, prompt_builder: PromptBuilder | None = None, agent_runner: AgentRunner | None = None
    ):
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.agent_runner = agent_runner or AgentRunner()

    def execute(
        self,
        step: StepFile,
        step_dir: Path,
        env: dict,
        prior_results: list[StepResult] | None = None,
        subject_name: str = "",
        subject_text: str = "",
        subject_meta: str = "{}",
    ) -> StepResult:
        """构建 prompt 并执行 Agent。"""
        logger.debug("Running .md step: %s", step.stem)

        prompt = self.prompt_builder.build(
            step=step,
            prior_results=prior_results or [],
            subject_name=subject_name,
            subject_text=subject_text,
            subject_meta=subject_meta,
            intermediates_dir=env.get("PIPELINE_INTERMEDIATES", ""),
            output_dir=env.get("PIPELINE_OUTPUT_DIR", ""),
            step_dir=str(step_dir),
        )

        return self.agent_runner.run(
            prompt=prompt,
            step_stem=step.stem,
            step_dir=step_dir,
            env=env,
        )


# ============================================================================
# InMemoryExecutor — 测试用
# ============================================================================


class InMemoryExecutor:
    """返回预置 StepResult 的测试 executor。

    用法::

        results = {"01-test": StepResult(step_name="01-test", status="ok")}
        executor = InMemoryExecutor(results)
        result = executor.execute(step, step_dir, env, [], "subject-01")
    """

    def __init__(self, results: dict[str, StepResult] | None = None):
        self.results = results or {}

    def execute(
        self,
        step: StepFile,
        step_dir: Path,
        env: dict,
        prior_results: list[StepResult] | None = None,
        subject_name: str = "",
    ) -> StepResult:
        return self.results.get(
            step.stem,
            StepResult(step_name=step.stem, status="ok"),
        )


# ============================================================================
# 分派函数
# ============================================================================


def _execute_step(
    step: StepFile,
    step_dir: Path,
    env: dict,
    prior_results: list[StepResult] | None,
    subject_name: str,
    py_runner: PyStepRunner,
    md_executor: MdStepExecutor,
) -> StepResult:
    """根据步骤类型分派到 PyStepRunner 或 MdStepExecutor。"""
    if step.step_type == "py":
        return py_runner.execute(step, step_dir, env, prior_results, subject_name)
    return md_executor.execute(step, step_dir, env, prior_results, subject_name)
