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

from paper_review.agent import (
    AgentConfig,
    build_command,
    build_command_without_model,
    is_model_config_error,
)
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
        subject_text: str = "",
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
        subject_text: str = "",
    ) -> StepResult:
        logger.info("  [.py] %s — starting", step.stem)

        # pi-lens-ignore: unchecked-throwing-call-python
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
        subject_path: str = "",
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
            subject_path=subject_path,
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


# 默认 pi 额外参数：禁用扩展以加速启动
_DEFAULT_PI_ARGS = ["-ne"]


class _AgentTimeoutError(Exception):
    """Agent 子进程超时（携带文本 stderr 供调用方分类错误）。"""

    def __init__(self, stderr: str) -> None:
        super().__init__("agent subprocess timed out")
        self.stderr = stderr


def _run_agent_subprocess(
    cmd: list[str],
    step_env: dict,
    step_timeout: int,
) -> subprocess.CompletedProcess:
    """执行一次 Agent 子进程（Popen + communicate，超时杀进程）。

    超时时抛出 _AgentTimeoutError（携带文本 stderr 供分类）。
    二进制不存在时 FileNotFoundError 原样上抛（调用方处理为 skipped）。
    """
    proc = subprocess.Popen(  # noqa: S603 — binary 来自环境配置，用户可控
        cmd,
        env=step_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout_data, stderr_data = proc.communicate(timeout=step_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr_data = proc.communicate()
        raise _AgentTimeoutError(stderr_data)
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout=stdout_data or "",
        stderr=stderr_data or "",
    )


def _output_json_is_valid(output_file: Path) -> bool:
    """检查 output.json 是否已由 pi 写入且有有效数据。

    当 pi 子进程超时时，prompt 模板要求 pi 将结果写入 output.json 文件，
    pi 可能已完成写入但未及时退出。此函数检查文件是否包含有效的评分数据。
    """
    return _read_output_json_if_valid(output_file) is not None


def _read_output_json_if_valid(output_file: Path) -> dict | None:
    """读取并返回有效的 output.json 数据，无效则返回 None。

    返回解析后的完整 dict（避免调用方重复读取文件）。
    """
    if not output_file.exists():
        return None
    try:
        data = json.loads(output_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    # 状态为 ok，且 data 中有实际内容（不只是空 dict）
    if data.get("status") != "ok":
        return None
    inner = data.get("data", {})
    if not inner or not isinstance(inner, dict):
        return None
    # 至少有一个非空值（区分 None/"" 与 0/False）
    if not any(v is not None and v != "" for v in inner.values()):
        return None
    return data


def _extract_braced_json(text: str) -> str | None:
    """从文本中提取第一个「平衡花括号」包裹的 JSON 对象原始片段。

    用 balance 扫描（正确处理嵌套花括号与字符串字面量内的花括号），
    从首个 `{` 匹配到对应 `}`。相比非贪婪正则，能容忍「JSON 后附说明
    文字再闭合围栏」的常见 LLM 输出形态。
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_parse_agent_json(output_text: str) -> dict | None:
    """从 pi stdout 提取结构化 JSON（dict）。

    1) 直接 json.loads（成功且是 dict → 返回）
    2) 首末花括号 balance 扫描提取再 loads（容忍代码围栏 / 前后说明文字）
    3) 均失败 → None（调用方视为格式失败，触发重试）
    """
    if not output_text:
        return None
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    # 首末花括号 balance 扫描（容忍嵌套 JSON 与「JSON 后附文字」）
    candidate = _extract_braced_json(output_text)
    if candidate is None:
        return None
    try:
        inner = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(inner, dict):
        return inner
    return None


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
            env: 环境变量，可包含：
                - PIPELINE_PI_BINARY: pi 可执行文件路径（默认 "pi"）
                - PIPELINE_PI_ARGS: 空格分隔的 pi 额外参数（覆盖默认值 -ne）
                - PIPELINE_AGENT_TYPE/PROVIDER/MODEL: Agent 类型与模型配置
                  （留空不传 flag；显式配置报错时回退为不传）
                - PIPELINE_STEP_TIMEOUT: 步骤超时秒数
            timeout: subprocess 超时秒数。
        """
        # pi-lens-ignore: unchecked-throwing-call-python
        os.makedirs(step_dir, exist_ok=True)
        step_env = {**env, "PIPELINE_STEP_DIR": str(step_dir)}

        # 失败反馈注入：上一次 attempt 因格式不合格失败时，把失败原因追加到 prompt
        feedback_file = step_dir / "_format_error.txt"
        if feedback_file.exists():
            try:
                feedback = feedback_file.read_text(encoding="utf-8").strip()
            except OSError:
                feedback = ""
            if feedback:
                prompt += (
                    "\n\n---\n\n"
                    "⚠ 重要：你上一次的回复因未遵循「只输出一个 JSON 对象」的要求被判为不合格。"
                    "这次请务必严格只输出一个 JSON 对象（首字符 `{`、末字符 `}`，"
                    "禁止 Markdown、表格、代码围栏、解释性文字）。\n"
                    "上一次的失败诊断：\n" + feedback
                )
            try:
                feedback_file.unlink()
            except OSError:
                pass

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

        # 解析 pi 额外参数
        pi_args_str = env.get("PIPELINE_PI_ARGS", "")
        if pi_args_str:
            pi_extra_args = pi_args_str.split()
        else:
            pi_extra_args = list(_DEFAULT_PI_ARGS)

        # Agent 配置（type/provider/model）——留空不传 flag（继承 Agent 默认），
        # 显式配置报错时回退为不传（降低对特定 provider/model 的硬编码弱依赖）。
        agent_cfg = AgentConfig.from_env(env)
        cmd = build_command(agent_cfg, pi_binary, str(prompt_file), pi_extra_args)

        prompt_size_kb = len(prompt) / 1024
        logger.info(
            "  [%s] ▶ pi %s (%.1fKB, timeout=%ds)",
            step_stem,
            " ".join(cmd),
            prompt_size_kb,
            step_timeout,
        )
        try:
            result = _run_agent_subprocess(cmd, step_env, step_timeout)
            # 显式 provider/model 配置无效（402/模型不存在等）→ 回退为不传重试一次。
            # 其他非零退出（prompt 崩溃、网络抖动等）直接传播为 error，不静默换模型。
            if (
                result.returncode != 0
                and agent_cfg.has_explicit_model()
                and is_model_config_error(result.stderr or "")
            ):
                logger.warning(
                    "  [%s] ⚠ 显式 provider/model 配置无效（exit %d）——回退为 Agent 默认重试",
                    step_stem,
                    result.returncode,
                )
                fallback_cmd = build_command_without_model(
                    agent_cfg, pi_binary, str(prompt_file), pi_extra_args
                )
                result = _run_agent_subprocess(fallback_cmd, step_env, step_timeout)
        except _AgentTimeoutError as e:
            stderr_data = e.stderr

            # pi 可能已将结果写入 output.json（prompt 模板要求），检查文件是否存在有效数据
            output_file = step_dir / "output.json"
            timeout_data = _read_output_json_if_valid(output_file)
            if timeout_data is not None:
                logger.info(
                    "  [%s] ⚠ pi timed out but output.json already written — treating as ok",
                    step_stem,
                )
                return StepResult(
                    step_name=step_stem,
                    status="ok",
                    data=timeout_data.get("data", {}),
                )

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
            # pi-lens-ignore: unchecked-throwing-call-python
            with open(output_file, "w") as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)
            return StepResult(
                step_name=step_stem,
                status="error",
                error=output_json["error"],
            )

        output_text = proc.stdout.strip()

        parsed = _try_parse_agent_json(output_text)

        if parsed is None:
            # 输出不是合法 JSON 对象 → 不再静默兜底为 ok。
            # 标记 error 触发 _retry_step 重试；原文保留进 data.raw_output，
            # 供 08-summarize 在重试耗尽后走正则兜底（降级保分）。
            error_msg = "agent 输出不是合法 JSON 对象（未遵循结构化输出要求）"
            output_json = {
                "step": step_stem,
                "status": "error",
                "error": error_msg,
                "data": {"raw_output": output_text},
            }
            # pi-lens-ignore: unchecked-throwing-call-python
            with open(output_file, "w") as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)
            # 记录失败原因供下次 attempt 注入 prompt（失败反馈）
            try:
                (step_dir / "_format_error.txt").write_text(
                    error_msg + "\n\n你上一次输出的开头：\n" + output_text[:500],
                    encoding="utf-8",
                )
            except OSError:
                pass
            logger.warning(
                "  [%s] ✗ 输出不是合法 JSON 对象（stdout=%dB），触发重试",
                step_stem,
                len(output_text),
            )
            return StepResult(
                step_name=step_stem,
                status="error",
                error=error_msg,
                data={"raw_output": output_text},
            )

        output_json = {
            "step": parsed.get("step", step_stem),
            "status": parsed.get("status", "ok"),
            "error": parsed.get("error"),
            "data": parsed.get("data", {}),
        }
        # pi-lens-ignore: unchecked-throwing-call-python
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
            subject_path=env.get("PIPELINE_SUBJECT_PATH", ""),
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
        subject_text: str = "",
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
    subject_text: str = "",
) -> StepResult:
    """根据步骤类型分派到 PyStepRunner 或 MdStepExecutor。"""
    if step.step_type == "py":
        return py_runner.execute(step, step_dir, env, prior_results, subject_name)
    return md_executor.execute(
        step, step_dir, env, prior_results, subject_name, subject_text=subject_text
    )
