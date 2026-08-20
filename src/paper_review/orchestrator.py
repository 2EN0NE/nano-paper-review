"""
Orchestrator —— 评审流水线执行引擎

薄编排层：配置解析 → Subject 发现 → 遍历 phases → 报告生成。
模式函数：_execute_batch（批量） / _execute_per_subject（逐 Subject，支持池化）。

数据模型 → pipeline_models.py
Step 执行 → pipeline_steps.py
Subject 发现 → subject_discovery.py
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from paper_review.agent import (
    ENV_AGENT_COMMAND,
    ENV_AGENT_ESCALATE,
    ENV_AGENT_MAX_ATTEMPTS,
    ENV_AGENT_MODEL,
    ENV_AGENT_PROVIDER,
    ENV_AGENT_TYPE,
    AgentConfig,
    parse_escalation_chain,
    resolve_command_for_attempt,
)
from paper_review.agent_stats import (  # pyright: ignore[reportMissingImports] — 新模块，pyright 文件索引未刷新
    AgentStatsRecorder,
    classify_kind,
    compute_fingerprint,
    default_stats,
    load_stats,
    save_stats,
)
from paper_review.dynamic_pool import DynamicPool, _is_productive_timeout, _is_rate_or_server_error
from paper_review.logging_config import get_logger
from paper_review.pipeline_models import (
    PhaseConfig,
    PipelineConfig,
    PipelineResult,
    PoolConfig,
    PoolProgress,
    RetryConfig,
    StepFile,
    StepResult,
    discover_steps,
)
from paper_review.pipeline_steps import (
    MdStepExecutor,
    PyStepRunner,
    StepExecutor,
    _execute_step,
)
from paper_review.progress import PhaseProgressInfo, PipelineProgress
from paper_review.subject_discovery import discover_subjects
from paper_review.timeout_estimator import estimate_step_timeout

logger = get_logger("orchestrator")

# PDF 文件大小到文本字符数的经验比例（中文技术文章约 1 字节 PDF → 0.25-0.5 字符）
_PDF_BYTE_TO_CHAR_RATIO = 0.35
# 当无法读取 PDF 文件大小时的兜底估算（字符数）
_FALLBACK_CHARS = 5000
# 注入评分 prompt 的 Subject 全文字符上限（超过则截断并附加提醒，避免 prompt 过大逼近上下文窗口）
_FULLTEXT_MAX_CHARS = 30000
# 全文提取失败/为空时注入评分 prompt 的占位提醒（评分步骤据此明确「无全文可比对」，而非臆造原文引用）
_FULLTEXT_UNAVAILABLE_NOTE = (
    "（全文提取失败或内容为空：评分时禁止引用原文具体证据，"
    "相关维度请基于公开常识与检索参考评分，并在 rationale 中注明「无全文可比对」）"
)


def _estimate_subject_chars(subjects: list[str], output_dir: Path) -> list[int]:
    """根据 manifest 中的 pdf_path 估算每个 subject 的文本字符数。

    读取 subject-manifest.json，取每个 subject 的 pdf_path，
    用文件大小 × _PDF_BYTE_TO_CHAR_RATIO 近似文本量。
    文件不存在时使用 _FALLBACK_CHARS 兜底。
    """
    manifest_path = output_dir / "subject-manifest.json"
    subject_chars: list[int] = []
    pdf_map: dict[str, Path] = {}

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest.get("subjects", []):
                name = entry.get("name", "")
                pdf = entry.get("pdf_path", "")
                if name and pdf:
                    pdf_map[name] = Path(pdf)
        except (json.JSONDecodeError, OSError):
            pass

    for subject in subjects:
        pdf_path = pdf_map.get(subject)
        if pdf_path and pdf_path.exists():
            try:
                size_bytes = pdf_path.stat().st_size
                chars = max(int(size_bytes * _PDF_BYTE_TO_CHAR_RATIO), _FALLBACK_CHARS // 2)
            except OSError:
                chars = _FALLBACK_CHARS
        else:
            chars = _FALLBACK_CHARS
        subject_chars.append(chars)
        logger.debug(
            "Subject '%s': estimated %d chars (pdf=%s)",
            subject,
            chars,
            pdf_path if pdf_path and pdf_path.exists() else "<not found>",
        )

    return subject_chars


def _load_subject_text(subject: str, output_dir: Path) -> tuple[str, str]:
    """从 manifest 拿 subject 的 pdf_path 并提取全文，返回 (text, pdf_path)。

    - 全文超过 _FULLTEXT_MAX_CHARS 时截断，并在开头附加提醒（评分步骤据此知道
      看到的不是完整原文，避免臆造后半部分证据）。
    - 提取失败/空文本/找不到 PDF 时返回占位提醒文本（非空），评分 prompt 据此
      明确「无全文可比对」，而非让 {subject.text} 静默变空。
    """
    manifest_path = output_dir / "subject-manifest.json"
    if not manifest_path.exists():
        return _FULLTEXT_UNAVAILABLE_NOTE, ""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _FULLTEXT_UNAVAILABLE_NOTE, ""
    for entry in manifest.get("subjects", []):
        if entry.get("name") != subject:
            continue
        pdf_path = entry.get("pdf_path", "")
        if not pdf_path or not Path(pdf_path).exists():
            return _FULLTEXT_UNAVAILABLE_NOTE, ""
        try:
            from paper_review.extractor import extract_pdf

            text = extract_pdf(str(pdf_path))
        except Exception as e:
            logger.warning("提取 Subject '%s' 全文失败：%s", subject, e)
            return _FULLTEXT_UNAVAILABLE_NOTE, ""
        if not text:
            return _FULLTEXT_UNAVAILABLE_NOTE, ""
        if len(text) > _FULLTEXT_MAX_CHARS:
            logger.info(
                "Subject '%s' 全文 %d 字超过 %d 字上限，截断并附加提醒",
                subject,
                len(text),
                _FULLTEXT_MAX_CHARS,
            )
            return (
                f"⚠ 注意：论文全文共 {len(text)} 字，超过 {_FULLTEXT_MAX_CHARS} 字上限，已截断。"
                f"以下仅提供前 {_FULLTEXT_MAX_CHARS} 字（后半部分未提供，评分请基于已提供内容）。\n\n"
                + text[:_FULLTEXT_MAX_CHARS],
                str(pdf_path),
            )
        return text, str(pdf_path)
    return _FULLTEXT_UNAVAILABLE_NOTE, ""


# ============================================================================
# _retry_step — 共享重试逻辑
# ============================================================================


def _retry_step(
    step: StepFile,
    step_dir: Path,
    env: dict,
    prior_results: list[StepResult],
    subject_name: str,
    retry_cfg: RetryConfig,
    executor: StepExecutor,
    step_timeout: int = 0,
    subject_text: str = "",
) -> StepResult:
    """执行一个 Step，按 RetryConfig 重试。

    唯一的重试实现点——_execute_batch 和 _execute_per_subject 都调用此函数。
    step_timeout 通过 env['PIPELINE_STEP_TIMEOUT'] 传递给 executor。
    """
    timed_env = {
        **env,
        "PIPELINE_STEP_TIMEOUT": str(step_timeout),
        ENV_AGENT_MAX_ATTEMPTS: str(retry_cfg.max_attempts),
    }
    # 升级链：.md 步骤每次尝试用第 N 条命令（单调推进 + 顶部饱和，ADR 0017）；
    # .py 步骤（04-extract-features 等）从 ENV_AGENT_ESCALATE 读整链自行按 subject 迭代。
    # 注：.py 步骤内部按链迭代 max_attempts 次，且 _retry_step 又在步骤级重试
    # max_attempts 次——若某 .py 步骤硬失败（返回 error），最坏为 N² 次子进程调用。
    # 当前模板 .py 步骤均为软失败（词表兜底返回空）故不触发；属潜在放大风险而非 bug。
    escalation_chain = parse_escalation_chain(AgentConfig.from_env(env).escalate)
    result: StepResult | None = None
    last_command = ""
    attempt_history: list[dict] = []
    t0 = time.monotonic()

    for attempt in range(1, retry_cfg.max_attempts + 1):
        attempt_env = timed_env
        attempt_command = ""
        if step.step_type == "md":
            cmd = resolve_command_for_attempt(escalation_chain, attempt)
            if cmd is not None:
                attempt_env = {**timed_env, ENV_AGENT_COMMAND: json.dumps(cmd)}
                attempt_command = " ".join(cmd)
                last_command = attempt_command

        logger.info(
            "  [%s] ▶ step '%s' attempt %d/%d (timeout=%ds)",
            subject_name,
            step.stem,
            attempt,
            retry_cfg.max_attempts,
            step_timeout,
        )

        try:
            result = executor.execute(  # pi-lens-ignore: python-sql-injection — StepExecutor 非 SQL
                step, step_dir, attempt_env, prior_results, subject_name, subject_text=subject_text
            )
            elapsed = time.monotonic() - t0
            result.subject = subject_name
            result.attempt = attempt
            attempt_history.append(
                {
                    "command": attempt_command,
                    "ok": result.status in ("ok", "skipped"),
                    "error": result.error or "",
                }
            )

            if result.status in ("ok", "skipped"):
                logger.info(
                    "  [%s] ✓ step '%s' %s in %.1fs (attempt %d)",
                    subject_name,
                    step.stem,
                    result.status,
                    elapsed,
                    attempt,
                )
                break

            logger.warning(
                "  [%s] ✗ step '%s' attempt %d failed (%.1fs elapsed): %s",
                subject_name,
                step.stem,
                attempt,
                elapsed,
                result.error,
            )
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                "  [%s] ✗ step '%s' attempt %d crashed (%.1fs elapsed): %s",
                subject_name,
                step.stem,
                attempt,
                elapsed,
                e,
            )
            attempt_history.append({"command": attempt_command, "ok": False, "error": str(e)})
            result = StepResult(
                step_name=step.stem,
                status="error",
                error=str(e),
                subject=subject_name,
                attempt=attempt,
            )

    if result is None:
        result = StepResult(
            step_name=step.stem,
            status="error",
            error="All attempts exhausted (no result)",
            subject=subject_name,
        )
    result.command = last_command
    result.attempt_history = attempt_history
    return result


# ============================================================================
# _execute_batch — 批量模式
# ============================================================================


def _apply_agent_overrides(base_env: dict, agent: AgentConfig | None) -> dict:
    """应用 phase 级 agent 覆盖到 step 环境变量（不改原 base_env）。

    phase.agent.provider/model/escalate 覆盖全局 agent 注入的值；空字段不覆盖
    （保留 base_env 里的全局值）。type 仅全局生效，phase 级不覆盖。

    phase 显式配置 provider/model 而未配置 escalate 时，清除全局注入的
    escalate——否则 _retry_step 会走升级链命令而静默忽略 provider/model。
    """
    if not agent or not (agent.has_explicit_model() or agent.escalate):
        return base_env
    env = dict(base_env)
    if agent.provider:
        env[ENV_AGENT_PROVIDER] = agent.provider
    if agent.model:
        env[ENV_AGENT_MODEL] = agent.model
    if agent.escalate:
        env[ENV_AGENT_ESCALATE] = json.dumps(agent.escalate)
    elif agent.has_explicit_model():
        env.pop(ENV_AGENT_ESCALATE, None)
    return env


def _execute_batch(
    phase: PhaseConfig,
    steps: list[StepFile],
    output_dir: Path,
    base_env: dict,
    executor: StepExecutor,
    pp: PipelineProgress | None = None,
    step_timeout: int = 0,
    skip_steps: set[str] | None = None,
    resume_skip_existing: bool = False,
) -> dict[str, list[StepResult]]:
    """批量模式：所有 Step 对 sentinel subject _batch_ 执行一次。

    用于 Pre / Post 阶段（mode=batch）。

    skip_steps: Resume 时产物已验证存在的步骤名集合——跳过（复用产物，不重跑）。
    resume_skip_existing: 注入 PIPELINE_RESUME_SKIP_EXISTING=1，步骤脚本内部跳过
        已有 per-subject 产物的 Subject（Pre 大步骤内部断点续做，T3/T4）。
    """
    results: list[StepResult] = []
    result_base = base_env.get("PIPELINE_RESULT_DIR", str(output_dir))

    # T1: batch 步骤内部子进度文件——orchestrator 注入路径，spinner 轮询显示。
    # 步骤脚本经 report_batch_progress() 写入；步骤执行前后清空/删除文件。
    progress_file: Path | None = None
    if pp is not None:
        progress_file = Path(result_base) / "progress" / f"{phase.name}-batch-step.json"
        try:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            progress_file = None
        if progress_file is not None:
            pp.set_batch_progress_file(phase.name, str(progress_file))

    skip_steps = skip_steps or set()
    for step in steps:
        step_dir = Path(result_base) / "intermediates" / phase.name / step.stem

        # T3: Resume 时跳过已验证完成的步骤（产物存在且 status ok/skipped）
        if step.stem in skip_steps:
            logger.info("  [_batch_] ⏭ step '%s' skipped (resume: pre product exists)", step.stem)
            results.append(StepResult(step_name=step.stem, status="skipped", subject="_batch_"))
            if pp:
                pp.phase_step_done(phase.name)
            continue

        env = _apply_agent_overrides(
            {
                **base_env,
                "PIPELINE_PHASE": phase.name,
                "PIPELINE_SUBJECT": "_batch_",
                "PIPELINE_STEP_NAME": step.stem,
                "PIPELINE_INTERMEDIATES": str(Path(result_base) / "intermediates"),
                "PIPELINE_DUPLICATE_POLICY": phase.duplicate_policy,
                # T3: Pre 大步骤内部断点续做标志（步骤脚本跳过已有 per-subject 产物）
                "PIPELINE_RESUME_SKIP_EXISTING": "1" if resume_skip_existing else "0",
            },
            phase.agent,
        )
        if progress_file is not None:
            env["PIPELINE_BATCH_PROGRESS_FILE"] = str(progress_file)
            _safe_unlink(progress_file)  # 清空上一步残留

        result = _retry_step(
            step=step,
            step_dir=step_dir,
            env=env,
            prior_results=results,
            subject_name="_batch_",
            retry_cfg=phase.retry,
            executor=executor,
            step_timeout=step_timeout,
        )
        results.append(result)

        if progress_file is not None:
            _safe_unlink(progress_file)  # 步骤结束删除 → spinner 清空子进度
        if pp:
            pp.clear_batch_detail(phase.name)  # 同步清除，保证最终渲染不残留

        # Progress — batch phase step done
        if pp:
            pp.phase_step_done(phase.name)

        if result.status == "error" and phase.retry.on_failure == "abort":
            logger.error("Aborting batch phase '%s' due to %s failure", phase.name, step.stem)
            break

    # 清除 batch 进度文件路径：spinner 不再轮询已删除的文件（防整个剩余运行
    # 每 100ms 一次无意义的 open 系统调用）
    if pp:
        pp.set_batch_progress_file(phase.name, None)
    return {"_batch_": results}


def _safe_unlink(path: Path) -> None:
    """尽力删除文件（缺失/权限失败静默）。"""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("failed to unlink %s", path)


# ============================================================================
# _execute_per_subject — 逐 Subject 模式（含池化）
# ============================================================================


def _execute_per_subject(
    phase: PhaseConfig,
    steps: list[StepFile],
    subjects: list[str],
    output_dir: Path,
    base_env: dict,
    executor: StepExecutor,
    pool_progress: PoolProgress | None = None,
    pp: PipelineProgress | None = None,
    step_timeout: int = 0,
    skip_completed: bool = False,
) -> dict[str, list[StepResult]]:
    """逐 Subject 模式：每个 Subject 顺序执行所有 Step。

    当 pool.workers > 1 且 subjects > 1 时启用 ThreadPoolExecutor 并发。
    skip_completed=True（Resume 续做）时，已有 output.json 的 Subject-Step 跳过。
    """
    base_env = _apply_agent_overrides(base_env, phase.agent)
    pool_cfg = phase.pool
    use_pool = pool_cfg is not None and pool_cfg.workers > 1 and len(subjects) > 1

    if use_pool:
        if pool_cfg.granularity == "step":  # type: ignore[union-attr]
            return _execute_per_step_pooled(
                phase=phase,
                steps=steps,
                subjects=subjects,
                output_dir=output_dir,
                base_env=base_env,
                executor=executor,
                pool_cfg=pool_cfg,  # type: ignore[arg-type]  # guard above
                pool_progress=pool_progress,
                pp=pp,
                step_timeout=step_timeout,
                skip_completed=skip_completed,
            )
        return _execute_per_subject_pooled(
            phase=phase,
            steps=steps,
            subjects=subjects,
            output_dir=output_dir,
            base_env=base_env,
            executor=executor,
            pool_cfg=pool_cfg,  # type: ignore[arg-type]  # guard above
            pool_progress=pool_progress,
            pp=pp,
            step_timeout=step_timeout,
            skip_completed=skip_completed,
        )

    # 顺序执行
    if pool_cfg is None and len(subjects) > 1:
        logger.info(
            "No pool configured for phase '%s' — running %d subject(s) sequentially",
            phase.name,
            len(subjects),
        )
    all_results: dict[str, list[StepResult]] = {}
    for subject in subjects:
        if pp:
            pp.phase_subject_running(phase.name, subject)
        if pool_progress:
            pool_progress.on_subject_start(subject)

        subject_results = _run_steps_for_subject(
            subject=subject,
            steps=steps,
            phase=phase,
            output_dir=output_dir,
            base_env=base_env,
            executor=executor,
            pp=pp,
            step_timeout=step_timeout,
            skip_completed=skip_completed,
        )
        all_results[subject] = subject_results

        if pool_progress:
            pool_progress.on_subject_complete(subject, subject_results)

    return all_results


def _run_steps_for_subject(
    subject: str,
    steps: list[StepFile],
    phase: PhaseConfig,
    output_dir: Path,
    base_env: dict,
    executor: StepExecutor,
    pp: PipelineProgress | None = None,
    step_timeout: int = 0,
    dyn_pool: DynamicPool | None = None,
    started: dict[str, float] | None = None,
    skip_completed: bool = False,
    seed_results: list[StepResult] | None = None,
) -> list[StepResult]:
    """对单个 Subject 执行全部 Step（顺序）。

    dyn_pool 不为 None 时，每个 step 完成后向 DynamicPool 报告观测结果。
    started 不为 None 时（池化模式），记录该 Subject 实际开始执行的时间——
    池化超时以此刻为起点（排队等待时间不计入超时）。
    skip_completed=True（Resume 续做）时，已有 output.json 的步骤跳过（复用产物）。
    seed_results 不为 None 时（step 粒度分波执行），以它作为前序波次已完成的
    StepResult 种子——step 粒度每个波次只执行一个 step，`.md` 步骤模板的
    `{intermediates.*}` 变量依赖 prior_results，必须把前序波次的产物带进来。
    """
    if started is not None:
        started[subject] = time.monotonic()

    subject_results: list[StepResult] = list(seed_results) if seed_results else []
    result_base = base_env.get("PIPELINE_RESULT_DIR", str(output_dir))

    # 提取 Subject 全文（供 .md 评分步骤的 {subject.text} 变量注入）。
    # 提取一次复用给所有 Step，避免每个 Step 重复 extract_pdf。
    # subject_path 通过 env 传给 MdStepExecutor（PIPELINE_SUBJECT_PATH），供 {subject.path}。
    subject_text, subject_path = _load_subject_text(subject, output_dir)

    # 加载 Pre Phase 为当前 Subject 写的 per-subject intermediates 作为 seed。
    # 批量预检索（05-batch-search / 04-extract-features）在 Pre Phase 执行，但按
    # Subject 布局写入 intermediates/{subject}/{step}/output.json。Review Phase 的
    # .md 步骤模板变量 {intermediates.STEP.data.KEY} 依赖 prior_results，必须把
    # 这些 Pre 产物带进来（否则评分 prompt 读不到检索结果）。
    review_step_names = {step.stem for step in steps}
    subject_intermediates = Path(result_base) / "intermediates" / subject
    if subject_intermediates.is_dir():
        loaded_names = {r.step_name for r in subject_results}
        for output_file in sorted(subject_intermediates.glob("*/output.json")):
            step_name = output_file.parent.name
            if step_name in review_step_names or step_name in loaded_names:
                continue
            try:
                out = json.loads(output_file.read_text(encoding="utf-8"))
                subject_results.append(
                    StepResult(
                        step_name=step_name,
                        status=out.get("status", "ok"),
                        error=out.get("error"),
                        subject=subject,
                        data=out.get("data", {}),
                    )
                )
                logger.debug("  [%s] ↳ seeded Pre intermediate '%s'", subject, step_name)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("  [%s] 跳过损坏的 Pre 产物 '%s': %s", subject, step_name, e)

    t0 = time.monotonic()

    logger.info(
        "  [%s] ▶ starting %d step(s) (timeout=%ds/step)", subject, len(steps), step_timeout
    )

    for step in steps:
        step_dir = Path(result_base) / "intermediates" / subject / step.stem

        # Resume 续做：已有 output.json 的步骤跳过（复用产物，不重跑）。
        # 仅当产物 status 为 ok/skipped 才跳过——status=error 的失败产物不跳过
        # （重跑重试），否则失败会被续做永久固化。
        if skip_completed and (step_dir / "output.json").exists():
            try:
                out = json.loads((step_dir / "output.json").read_text(encoding="utf-8"))
                result = None
                if out.get("status", "ok") in ("ok", "skipped"):
                    result = StepResult(
                        step_name=step.stem,
                        status=out.get("status", "ok"),
                        error=out.get("error"),
                        subject=subject,
                        data=out.get("data", {}),
                        resumed=True,
                    )
            except (json.JSONDecodeError, OSError):
                result = None  # 产物损坏 → 正常重跑
            if result is not None:
                subject_results.append(result)
                logger.info(
                    "  [%s] ⏭ step '%s' skipped (resume: output.json exists)",
                    subject,
                    step.stem,
                )
                if pp:
                    pp.phase_subject_step_done(phase.name, subject)
                continue

        # 动态池：用 with 保护槽位生命周期，异常时自动释放
        slot_cm = dyn_pool.worker_slot() if dyn_pool else nullcontext()

        # 动态池超时乘数：productive timeout 后自动上调后续 step 的时限
        adjusted_timeout = step_timeout
        if dyn_pool is not None and dyn_pool.timeout_multiplier > 1.0:
            adjusted_timeout = math.floor(step_timeout * dyn_pool.timeout_multiplier)

        with slot_cm:
            env = {
                **base_env,
                "PIPELINE_PHASE": phase.name,
                "PIPELINE_SUBJECT": subject,
                "PIPELINE_STEP_NAME": step.stem,
                "PIPELINE_INTERMEDIATES": str(Path(result_base) / "intermediates"),
                "PIPELINE_DUPLICATE_POLICY": phase.duplicate_policy,
                "PIPELINE_SUBJECT_PATH": subject_path,
            }

            result = _retry_step(
                step=step,
                step_dir=step_dir,
                env=env,
                prior_results=subject_results,
                subject_name=subject,
                retry_cfg=phase.retry,
                executor=executor,
                step_timeout=adjusted_timeout,
                subject_text=subject_text,
            )
            subject_results.append(result)

        # 槽位已释放，安全上报观测
        if dyn_pool is not None:
            is_429_503 = _is_rate_or_server_error(result.error)
            is_prod_timeout = _is_productive_timeout(result.error)
            is_success = result.status in ("ok", "skipped")
            dyn_pool.observe(is_429_503, is_prod_timeout, is_success)
            if pp:
                pp.update_dynamic_workers(
                    active=dyn_pool.active_workers,
                    current=dyn_pool.current_workers,
                    timeout_multiplier=dyn_pool.timeout_multiplier,
                )

        if pp:
            pp.phase_subject_step_done(phase.name, subject)

        if result.status == "error" and phase.retry.on_failure == "abort":
            logger.error("Aborting pipeline for %s due to %s failure", subject, step.stem)
            break

    elapsed = time.monotonic() - t0
    logger.info("  [%s] ✓ all %d step(s) done (%.1fs total)", subject, len(steps), elapsed)
    return subject_results


# ============================================================================
# 池化执行
# ============================================================================

# 超时 worker 排空等待上限（秒）：防御标准库 wait 对已 cancel future 的漏计数死锁
_TIMEOUT_DRAIN_FUTURES = 30


def _make_error_results(
    steps: list[StepFile],
    subject: str,
    error_msg: str,
) -> list[StepResult]:
    """为超时/失败的 subject 生成统一的 error StepResult 列表。"""
    return [
        StepResult(step_name=step.stem, status="error", error=error_msg, subject=subject)
        for step in steps
    ]


def _report_subject_failure(
    pool_progress: PoolProgress | None,
    subject_state: dict[str, str],
    subject: str,
    status: str,
    error: str,
) -> None:
    """step 粒度下向 PoolProgress 上报 subject 失败（仅首次失败上报，避免重复计数）。"""
    if pool_progress is None or subject_state.get(subject) == "failed":
        return
    pool_progress.on_subject_fail(subject, status, error)
    subject_state[subject] = "failed"


def _execute_per_subject_pooled(
    phase: PhaseConfig,
    steps: list[StepFile],
    subjects: list[str],
    output_dir: Path,
    base_env: dict,
    executor: StepExecutor,
    pool_cfg: PoolConfig,
    pool_progress: PoolProgress | None = None,
    pp: PipelineProgress | None = None,
    step_timeout: int = 0,
    skip_completed: bool = False,
) -> dict[str, list[StepResult]]:
    """Worker 池并发处理多个 Subject。

    profile='dynamic' 时启用基于贝叶斯估计的自适应并发控制。
    """
    is_dynamic = pool_cfg.profile == "dynamic"

    # dynamic 模式：线程池用 workers_max，DynamicPool 控制实际并发
    if is_dynamic:
        actual_workers = pool_cfg.workers_max
        total_steps = len(subjects) * len(steps)
        dyn_pool = DynamicPool(pool_cfg, total_steps)
    else:
        actual_workers = min(pool_cfg.workers, len(subjects))
        dyn_pool = None

    per_subject_timeout = pool_cfg.timeout if pool_cfg.timeout > 0 else None

    # 墙钟兜底上限：worker 卡死（.py 步骤进程内执行无 subprocess 超时）时，
    # 排队未开始的 Subject 永远记录不到 started → 永不超时 → 主循环无限挂起。
    # 上限 = 全部 subject 最坏串行化时间（ceil 波次 × 单 subject 预算）；
    # dynamic 用 workers_min（DynamicPool 可收缩并发）避免误杀。
    pool_start = time.monotonic()
    if per_subject_timeout is not None:
        wall_workers = actual_workers if not is_dynamic else max(1, pool_cfg.workers_min)
        pool_wall_limit = math.ceil(len(subjects) / wall_workers) * per_subject_timeout
    else:
        pool_wall_limit = None

    if is_dynamic:
        logger.info(
            "Pool mode: dynamic (initial=%d, min=%d, max=%d), %d subject(s), %d step(s)/subject (step_timeout=%ds)",
            pool_cfg.workers,
            pool_cfg.workers_min,
            pool_cfg.workers_max,
            len(subjects),
            len(steps),
            step_timeout,
        )
    else:
        logger.info(
            "Pool mode: %d worker(s) processing %d subject(s) (step_timeout=%ds)",
            actual_workers,
            len(subjects),
            step_timeout,
        )

    all_results: dict[str, list[StepResult]] = {}
    errors: list[str] = []
    # Subject 实际开始执行的时间（worker 线程入口记录）；排队等待不计入超时
    started: dict[str, float] = {}
    timed_out_futures: set[Future] = set()
    # 超时判定会立刻 cancel/写 error 占位，但 worker 仍可能实质完成（cancel 对
    # RUNNING future 无效，排空窗口内完成即“晚到成功”）。PoolProgress 的 fail 与
    # complete 互斥（pending = total - completed - failed），故超时的 fail 事件
    # 延迟到排空后按最终结果上报：恢复 → complete；未恢复 → fail（曾双报导致
    # pending 出现负值）。
    deferred_timeout_fails: dict[str, str] | None = {} if pool_progress else None

    tp_executor = ThreadPoolExecutor(max_workers=actual_workers)
    try:
        future_map: dict[Future, str] = {}
        for s in subjects:
            fut = tp_executor.submit(
                _run_steps_for_subject,
                s,
                steps,
                phase,
                output_dir,
                base_env,
                executor,
                pp,
                step_timeout,
                dyn_pool,
                started,
                skip_completed,
            )
            future_map[fut] = s
            if pool_progress:
                pool_progress.on_subject_start(s)
            if pp:
                pp.phase_subject_running(phase.name, s)

        pending = set(future_map.keys())
        while pending:
            done, pending = wait(pending, timeout=1.0)

            for future in done:
                subject = future_map[future]
                try:
                    results = future.result()
                    all_results[subject] = results
                    elapsed = time.monotonic() - started.get(subject, time.monotonic())
                    logger.info("  [%s] ✓ completed in %.0fs", subject, elapsed)
                    if pool_progress:
                        pool_progress.on_subject_complete(subject, results)
                except Exception as e:
                    logger.error("  [%s] ✗ failed: %s", subject, e)
                    errors.append(subject)
                    if pool_progress:
                        pool_progress.on_subject_fail(subject, "error", str(e))
                    all_results[subject] = _make_error_results(steps, subject, str(e))

            # 超时检查
            now = time.monotonic()
            for future in list(pending):
                subject = future_map[future]
                actual_start = started.get(subject)
                if (
                    per_subject_timeout is not None
                    and actual_start is not None
                    and (now - actual_start) > per_subject_timeout
                ):
                    pending.discard(future)
                    timed_out_futures.add(future)
                    future.cancel()
                    errors.append(subject)
                    logger.error("  [%s] ✗ timed out after %ds", subject, per_subject_timeout)
                    if deferred_timeout_fails is not None:
                        deferred_timeout_fails[subject] = f"Timed out after {per_subject_timeout}s"
                    all_results[subject] = _make_error_results(
                        steps, subject, f"Timed out after {per_subject_timeout}s"
                    )
                elif (
                    pool_wall_limit is not None
                    and actual_start is None
                    and (now - pool_start) > pool_wall_limit
                ):
                    # 排队超过墙钟上限仍未开始：worker 大概率卡死，止损放弃
                    pending.discard(future)
                    timed_out_futures.add(future)
                    future.cancel()
                    errors.append(subject)
                    logger.error(
                        "  [%s] ✗ queued but never started after %ds — worker may be stuck",
                        subject,
                        pool_wall_limit,
                    )
                    if deferred_timeout_fails is not None:
                        deferred_timeout_fails[subject] = (
                            f"Never started after {pool_wall_limit}s (worker stuck?)"
                        )
                    all_results[subject] = _make_error_results(
                        steps, subject, f"Never started after {pool_wall_limit}s (worker stuck?)"
                    )

            if not pp:
                logger.info(
                    "  Progress: %d/%d done, %d running",
                    len(all_results),
                    len(subjects),
                    len(pending),
                )
    finally:
        tp_executor.shutdown(wait=False, cancel_futures=True)

    # 等待超时 worker 实质完成
    # 兜底 timeout：标准库 wait 对"已 cancel（CANCELLED）且通知发生在 waiter 安装前"
    # 的 future 可能漏计数导致死锁——给上限避免卡死（正常场景 RUNNING worker 秒级完成）。
    if timed_out_futures:
        done_timed_out, not_done = wait(timed_out_futures, timeout=_TIMEOUT_DRAIN_FUTURES)
        for future in done_timed_out:
            subject = future_map[future]
            try:
                results = future.result()
                all_results[subject] = results
                elapsed = time.monotonic() - started.get(subject, time.monotonic())
                logger.info("  [%s] ✓ completed after timeout in %.0fs", subject, elapsed)
                # 恢复完成：上报 complete（fail 事件已延迟，此处不双报）并清理 errors
                if pool_progress:
                    pool_progress.on_subject_complete(subject, results)
                if subject in errors:
                    errors.remove(subject)
            except Exception as e:
                logger.debug("  [%s] timed-out future resolved with %s", subject, type(e).__name__)
                # 未恢复（CANCELLED / 异常结束）：按延迟记录的原始超时原因上报 fail
                if pool_progress and deferred_timeout_fails is not None:
                    pool_progress.on_subject_fail(
                        subject,
                        "timeout",
                        deferred_timeout_fails.pop(subject, "timed out"),
                    )

        if not_done:
            logger.warning(
                "  %d timed-out future(s) unresolved after drain timeout — results dropped",
                len(not_done),
            )
            if pool_progress and deferred_timeout_fails is not None:
                for future in not_done:
                    subject = future_map[future]
                    pool_progress.on_subject_fail(
                        subject,
                        "timeout",
                        deferred_timeout_fails.pop(subject, "timed out"),
                    )

    if pool_cfg.ordered:
        all_results = {s: all_results[s] for s in subjects if s in all_results}

    if errors:
        logger.warning("Pool mode finished with %d failed subject(s): %s", len(errors), errors)

    return all_results


def _execute_per_step_pooled(
    phase: PhaseConfig,
    steps: list[StepFile],
    subjects: list[str],
    output_dir: Path,
    base_env: dict,
    executor: StepExecutor,
    pool_cfg: PoolConfig,
    pool_progress: PoolProgress | None = None,
    pp: PipelineProgress | None = None,
    step_timeout: int = 0,
    skip_completed: bool = False,
) -> dict[str, list[StepResult]]:
    """step 级粒度：按 Step 分波次（barrier），波内多 Subject 并行。

    - 外层循环 Steps（顺序）：所有 Subject 完成当前 Step 后才进入下一 Step；
    - 波内：ThreadPoolExecutor 并行所有 Subject 的当前 Step（每波新建线程池）；
    - 单步预算 = pool.timeout（>0 时，模板契约“step 粒度下为单步超时上限”）
      否则 step_timeout；超时从 Subject 实际开始起算（排队等待不计入）；
      预算同时作为 executor 超时传入（PIPELINE_STEP_TIMEOUT），避免估算值
      小于配置上限时步骤被提前杀掉；
    - 波次墙钟上限 = ceil(subjects/workers) × 单步预算，兜底 cancel 无效的僵尸 worker
      （.py 步骤进程内执行无 subprocess 超时，卡死必须由这里兜底）；
    - profile='dynamic' 时 DynamicPool 跨波共享，自适应观测跨波累计；
    - 前序波次产物通过 seed_results 传入（prior_results 种子）：step 粒度每个
      波次只执行一个 step，`.md` 步骤模板的 {intermediates.*} 变量依赖
      prior_results，必须带上已完成的 StepResult（barrier 保证前序产物已落盘）。
    """
    is_dynamic = pool_cfg.profile == "dynamic"
    if is_dynamic:
        actual_workers = pool_cfg.workers_max
        total_steps = len(subjects) * len(steps)
        dyn_pool = DynamicPool(pool_cfg, total_steps)
    else:
        actual_workers = min(pool_cfg.workers, len(subjects))
        dyn_pool = None

    # step 粒度下单步预算：pool.timeout > 0 时优先（YAML 注释声明的契约），否则用
    # phase 估算的 step_timeout。波次墙钟兜底 = 全部 subject 最坏串行化时间（ceil 波次 ×
    # 单步预算），正常排队场景足够；僵尸 worker 卡死时由其收尾（不会无限等待）。
    # dynamic 与 subject 粒度一致用 workers_min（DynamicPool 可收缩并发）避免误杀
    # 排队中的 Subject——若按初始 workers 预算，池收缩后真实串行化时间会超出上限。
    effective_step_timeout = pool_cfg.timeout if pool_cfg.timeout > 0 else step_timeout
    if effective_step_timeout > 0:
        wave_workers = actual_workers if not is_dynamic else max(1, pool_cfg.workers_min)
        wave_wall_limit = math.ceil(len(subjects) / wave_workers) * effective_step_timeout
    else:
        # 与 subject 粒度一致：无超时配置（pool.timeout=0 且估算为 0）不设墙钟上限。
        # 曾回退 _TIMEOUT_DRAIN_FUTURES（30s）——"无超时"被静默改成 30s 硬上限。
        wave_wall_limit = None

    logger.info(
        "Granularity=step — %d step(s) × %d subject(s), %d worker(s)/wave (step_timeout=%ds)",
        len(steps),
        len(subjects),
        actual_workers,
        effective_step_timeout,
    )

    all_results: dict[str, list[StepResult]] = {}
    errors: set[str] = set()
    # on_failure=abort 的 subject：本波步骤失败后不再提交后续波次（与 subject 粒度
    # break 语义一致——曾每波无条件提交全部 subject，abort 被静默忽略）。
    aborted: set[str] = set()
    # PoolProgress 为 subject 级事件：step 模式每波提交全部 subject，只能
    # 首次波次上报 start、失败首次上报 fail、最后波次成功上报 complete
    # （按波次重复上报会虚增 completed/failed 计数）。
    subject_state: dict[str, str] = {}

    for step_index, step in enumerate(steps):
        last_wave = step_index == len(steps) - 1
        wave_results: dict[str, StepResult] = {}
        started: dict[str, float] = {}
        canceled: set[Future] = set()
        wave_start = time.monotonic()
        tp_executor = ThreadPoolExecutor(max_workers=actual_workers)
        # 提前初始化：submit 中途抛异常时 finally 内收割僵尸仍需访问（避免 unbound）
        future_map: dict[Future, str] = {}
        try:
            for s in subjects:
                if s in aborted:
                    continue  # abort：本 subject 不再参与后续波次
                fut = tp_executor.submit(
                    _run_steps_for_subject,
                    s,
                    [step],  # 单步波次：每个 Subject 只跑当前 Step
                    phase,
                    output_dir,
                    base_env,
                    executor,
                    pp,
                    effective_step_timeout,  # 单步预算同时作为 executor 超时（pool.timeout 契约）
                    dyn_pool,
                    started,
                    skip_completed,
                    all_results.get(s, []),  # 前序波次产物作为 prior_results 种子
                )
                future_map[fut] = s
                if s not in subject_state:
                    subject_state[s] = "started"
                    if pool_progress:
                        pool_progress.on_subject_start(s)
                    if pp:
                        pp.phase_subject_running(phase.name, s)

            # barrier：轮询等待波内全部完成。每个 Subject 的超时从“实际开始”起算
            # （worker 入口记录于 started），排队未开始的 Subject 不计入超时——
            # 与 subject 粒度修复语义一致，避免大批量排队被集体冤杀。
            pending = set(future_map.keys())
            while pending:
                done, pending = wait(pending, timeout=1.0)
                now = time.monotonic()
                for fut in done:
                    s = future_map[fut]
                    try:
                        res = fut.result()
                        # res = 前序波次种子 + 当前步骤结果；取最后一项（当前步骤）。
                        # 曾取 res[0]——第 2 波起取到的是前序波次产物，当前步骤结果被
                        # 丢弃，导致结果列表重复首步骤、报告/CLI 统计缺后续步骤。
                        wave_results[s] = (
                            res[-1] if res else _make_error_results([step], s, "no result")[0]
                        )
                        if pool_progress and last_wave and subject_state.get(s) == "started":
                            pool_progress.on_subject_complete(s, res)
                            subject_state[s] = "done"
                    except Exception as e:
                        wave_results[s] = StepResult(
                            step_name=step.stem, status="error", error=str(e), subject=s
                        )
                        errors.add(s)
                        _report_subject_failure(pool_progress, subject_state, s, "error", str(e))
                        if phase.retry.on_failure == "abort":
                            aborted.add(s)
                    else:
                        # 正常完成但状态为 error（重试耗尽）→ on_failure=abort 时停止该 subject
                        if (
                            phase.retry.on_failure == "abort"
                            and wave_results.get(s) is not None
                            and wave_results[s].status == "error"
                        ):
                            aborted.add(s)
                            # 上报失败：abort 后该 subject 不再进入最后波次，
                            # 不报 complete 也不报 fail 会让 PoolProgress 出现 pending 泄漏
                            # （曾只处理异常/超时分支，error-status 结果被漏报）。
                            _report_subject_failure(
                                pool_progress,
                                subject_state,
                                s,
                                "error",
                                wave_results[s].error or "step failed",
                            )

                # 已实际开始且超过单步预算的 Subject 判超时（排队未开始的不计）
                for fut in list(pending):
                    s = future_map[fut]
                    st = started.get(s)
                    if (
                        effective_step_timeout > 0
                        and st is not None
                        and (now - st) > effective_step_timeout
                    ):
                        pending.discard(fut)
                        canceled.add(fut)
                        fut.cancel()
                        errors.add(s)
                        wave_results[s] = _make_error_results(
                            [step], s, f"step timed out after {effective_step_timeout}s"
                        )[0]
                        logger.error(
                            "  [%s] ✗ step '%s' timed out after %ds",
                            s,
                            step.stem,
                            effective_step_timeout,
                        )
                        _report_subject_failure(
                            pool_progress,
                            subject_state,
                            s,
                            "timeout",
                            f"step timed out after {effective_step_timeout}s",
                        )
                        if phase.retry.on_failure == "abort":
                            # 与 error-status/异常分支一致：abort 策略对超时同样生效
                            aborted.add(s)

                # 波次墙钟兜底：僵尸 worker 占住线程/槽位时排队者永远无法开始，
                # 到上限后整体放弃剩余（僵尸场景无法真正 kill 线程，只能止损）
                if wave_wall_limit is not None and pending and (now - wave_start) > wave_wall_limit:
                    for fut in list(pending):
                        s = future_map[fut]
                        pending.discard(fut)
                        canceled.add(fut)
                        fut.cancel()
                        errors.add(s)
                        wave_results[s] = _make_error_results(
                            [step], s, f"wave timed out after {wave_wall_limit}s"
                        )[0]
                        logger.error(
                            "  [%s] ✗ step '%s' wave timeout after %ds",
                            s,
                            step.stem,
                            wave_wall_limit,
                        )
                        _report_subject_failure(
                            pool_progress,
                            subject_state,
                            s,
                            "timeout",
                            f"wave timed out after {wave_wall_limit}s",
                        )
                        if phase.retry.on_failure == "abort":
                            aborted.add(s)
        finally:
            tp_executor.shutdown(wait=False, cancel_futures=True)
            # 有界等待被 cancel 但仍在运行的 worker 结束：cancel 对 RUNNING future
            # 无效，僵尸继续跑会破坏 barrier（同 Subject 下一步与其并发）并占用
            # DynamicPool 槽位导致后续波次级联超时。等它完成以恢复顺序不变量；
            # 无限循环的僵尸由兜底超时放弃（daemon 线程随进程结束清理）。
            zombies = [f for f in canceled if not f.done()]
            if zombies:
                done_zombies, _ = wait(zombies, timeout=_TIMEOUT_DRAIN_FUTURES)
                # 收割实际完成的僵尸结果：波次超时后 worker 仍可能实质完成（cancel
                # 对 RUNNING future 无效，output.json 已落盘）。用真实结果覆盖 error
                # 占位，与 subject 粒度排空回收一致——曾静默丢弃，导致“报告 error 但
                # 磁盘产物 ok、续做又跳过该步骤”的视图分裂。
                for fut in done_zombies:
                    s = future_map[fut]
                    try:
                        res = fut.result()
                        if res and res[-1].status in ("ok", "skipped"):
                            wave_results[s] = res[-1]
                            errors.discard(s)
                    except Exception as e:
                        logger.debug("  [%s] zombie resolved with %s", s, type(e).__name__)
                        # 僵尸以异常结束：保留超时 error 占位

        # 按 Subject 原始顺序收集本波结果（ordered 语义）
        for s in subjects:
            if s in aborted and s not in wave_results:
                # abort 后的后续波次（未提交，无本波结果）：不追加，与 subject
                # 粒度 break 一致（剩余步骤无结果，不制造伪造 error）。
                # abort 发生的当波（失败步骤在 wave_results 中）正常追加。
                continue
            all_results.setdefault(s, []).append(
                wave_results.get(s, _make_error_results([step], s, "wave failed")[0])
            )

    if errors:
        logger.warning(
            "Step-granularity finished with %d failed subject-step(s): %s",
            len(errors),
            sorted(errors),
        )

    return all_results


# ============================================================================
# CLI 树形图 + 叶子输出
# ============================================================================


def _build_cli_tree(
    task_id: str,
    pipeline_name: str,
    config: PipelineConfig,
    all_phase_results: dict[str, dict[str, list[StepResult]]],
    pipeline_dir: Path,
    task_dir: Path,
) -> str:
    """构建 CLI 管线树形图、过程统计、叶子节点输出。"""
    lines: list[str] = []
    phases = config.phases

    # ── 全局统计 ──
    all_steps: list[StepResult] = []
    for pr in all_phase_results.values():
        for subj_results in pr.values():
            all_steps.extend(subj_results)

    total = len(all_steps)
    ok_count = sum(1 for r in all_steps if r.status == "ok")
    err_count = sum(1 for r in all_steps if r.status == "error")

    lines.append(f"📊 Pipeline: {pipeline_name}  ·  Task: {task_id}")
    lines.append(f"   合计 {total} 步（✅ {ok_count}  /  ❌ {err_count}）")
    lines.append("")

    # ── 终端 phase 判定（phases 列表最后一项）──
    if not phases:
        lines.append("(无 phase)")
        return "\n".join(lines)
    terminal_phase = phases[-1]

    # ── step_type 查表 + 缓存 step 顺序（避免重复 I/O）──
    step_type_map: dict[tuple[str, str], str] = {}
    phase_step_names: dict[str, list[str]] = {}  # phase.name → ordered step stems
    for phase in phases:
        step_files = discover_steps(pipeline_dir / phase.directory)
        phase_step_names[phase.name] = [sf.stem for sf in step_files]
        for sf in step_files:
            step_type_map[(phase.name, sf.stem)] = sf.step_type

    # ── 逐 phase 渲染 ──
    for i, phase in enumerate(phases):
        phase_results = all_phase_results.get(phase.name, {})
        is_last_phase = i == len(phases) - 1
        phase_prefix = "└──" if is_last_phase else "├──"
        indent = "    " if is_last_phase else "│   "

        # Phase 概览
        if phase.mode == "batch" and not phase_results:
            # batch 阶段无任何产物：续做跳过的 Pre（或 0 步骤阶段）——避免显示
            # 误导性的“✅ 0/0”（b_err==0 恒真，看起来像空批次成功）。
            lines.append(f"{phase_prefix} {phase.display_label} (batch) ⏭ skipped（无产物）")
            lines.append("")
            continue
        if phase.mode == "batch":
            batch = phase_results.get("_batch_", [])
            b_ok = sum(1 for r in batch if r.status == "ok")
            b_err = sum(1 for r in batch if r.status == "error")
            icon = "✅" if b_err == 0 else "❌"
            lines.append(f"{phase_prefix} {phase.display_label} (batch) {icon} {b_ok}/{len(batch)}")
        else:
            subjects_in_phase = [s for s in phase_results if s != "_batch_"]
            lines.append(
                f"{phase_prefix} {phase.display_label} (per_subject) "
                f"{len(subjects_in_phase)} subject(s)"
            )

        # 从缓存获取该 phase 的 step 顺序（无重复 I/O）
        step_names = phase_step_names.get(phase.name, [])

        for j, step_name in enumerate(step_names):
            is_last_step = j == len(step_names) - 1
            step_prefix = indent + ("└──" if is_last_step else "├──")
            out_indent = indent + ("    " if is_last_step else "│   ")

            s_type = step_type_map.get((phase.name, step_name), "py")

            # per-subject 统计
            subj_stats = {"ok": 0, "error": 0, "skipped": 0}
            for subj_results in phase_results.values():
                for sr in subj_results:
                    if sr.step_name == step_name:
                        if sr.status == "ok":
                            subj_stats["ok"] += 1
                        elif sr.status == "error":
                            subj_stats["error"] += 1
                        else:
                            subj_stats["skipped"] += 1

            icon = "❌" if subj_stats["error"] > 0 else ("⚠️" if subj_stats["ok"] == 0 else "✅")
            if phase.mode == "batch":
                lines.append(
                    f"{step_prefix} {step_name} ({s_type}) {icon} {subj_stats['ok']} ok"
                    f"{' / ❌ ' + str(subj_stats['error']) if subj_stats['error'] else ''}"
                )
            else:
                parts = []
                if subj_stats["ok"]:
                    parts.append(f"✅ {subj_stats['ok']} ok")
                if subj_stats["skipped"]:
                    parts.append(f"⚠️ {subj_stats['skipped']} skipped")
                if subj_stats["error"]:
                    parts.append(f"❌ {subj_stats['error']} error")
                lines.append(f"{step_prefix} {step_name} ({s_type})  {' / '.join(parts)}")

            # ── 叶子节点输出 ──
            is_terminal = phase == terminal_phase
            is_leaf_in_phase = (phase.mode == "batch" and is_last_step) or (
                phase.mode == "per_subject"
            )
            if is_terminal and is_leaf_in_phase and subj_stats["ok"] > 0:
                leaf_lines = _render_leaf_outputs(
                    phase, step_name, phase_results, task_dir, out_indent
                )
                lines.extend(leaf_lines)

        lines.append("")

    lines.append(f"完整报告: {task_dir / 'report.md'}")
    return "\n".join(lines)


# ── 已知文件键后缀（后缀匹配，避免误匹配 profile / empathy 等）──
_FILE_KEY_SUFFIXES = ("_path", "_file", "_dir")


def _render_leaf_outputs(
    phase: PhaseConfig,
    step_name: str,
    phase_results: dict[str, list[StepResult]],
    task_dir: Path,
    out_indent: str,
) -> list[str]:
    """渲染叶子步骤的输出内容（文字 + 文件路径）。"""
    out_lines: list[str] = []

    if phase.mode == "batch":
        batch = phase_results.get("_batch_", [])
        candidates = [r for r in batch if r.step_name == step_name and r.status == "ok"]
    else:
        candidates = []
        for subj, subj_results in phase_results.items():
            if subj == "_batch_":
                continue
            for sr in subj_results:
                if sr.step_name == step_name and sr.status == "ok":
                    candidates.append(sr)

    if not candidates:
        return out_lines

    # ── 从 output.json 提取数据和文件路径 ──
    if phase.mode == "batch":
        for sr in candidates[:1]:  # batch 只取一条
            data = sr.data
            if not data:
                continue
            # 文件路径（严格后缀匹配：_path / _file / _dir）
            file_keys = [k for k in data if k.endswith(_FILE_KEY_SUFFIXES)]
            for fk in file_keys:
                val = data[fk]
                if isinstance(val, str) and val:
                    out_lines.append(f"{out_indent}→ 文件: {val}")
                elif isinstance(val, list):
                    for v in val:
                        if isinstance(v, str):
                            out_lines.append(f"{out_indent}→ 文件: {v}")
            # 报告目录等约定路径
            if "archived_subjects" in data:
                out_lines.append(
                    f"{out_indent}→ 归档 {data.get('total', len(data['archived_subjects']))} 篇: "
                    f"{task_dir.parent / 'reports'}"
                )
            # 非文件标的文本数据
            text_keys = [
                k
                for k in data
                if k not in file_keys
                and k != "archived_subjects"
                and k != "total"
                and isinstance(data[k], (str, int, float, bool))
            ]
            for tk in text_keys:
                val = data[tk]
                out_lines.append(f"{out_indent}→ {tk}: {val}")
    else:
        # per_subject: 每 subject 显示一行摘要
        for sr in candidates:
            subj = sr.subject or "?"
            data = sr.data
            if not data:
                continue
            # 文件路径优先（严格后缀匹配）
            file_keys = [k for k in data if k.endswith(_FILE_KEY_SUFFIXES)]
            shown = False
            for fk in file_keys:
                val = data[fk]
                if isinstance(val, str) and val:
                    out_lines.append(f"{out_indent}  [{subj}] → {val}")
                    shown = True
            if not shown:
                # 显示关键文本
                text_keys = [
                    k
                    for k in data
                    if k not in file_keys and isinstance(data[k], (str, int, float, bool))
                ]
                for tk in text_keys[:2]:  # 最多显示 2 个字段
                    out_lines.append(f"{out_indent}  [{subj}] → {tk}: {data[tk]}")

    return out_lines


def _per_subject_step_has_error(task_dir: Path, step_stem: str) -> bool:
    """该步骤是否存在 status=error 的 per-subject 产物（resume 跳过判定用）。

    05-batch-search 单篇检索失败时写 status=error 的 per-subject 产物，而步骤级
    产物仍为 ok——若续做时跳过该步骤，失败篇的空引用会被静默固化到评审
    （ADR 0005“失败产物续做时重跑”原则要求重跑）。subject 目录排除 batch 汇总
    目录（约定 pre/post，与 _collect_degradation_warnings 一致）。
    """
    intermediates = task_dir / "intermediates"
    if not intermediates.is_dir():
        return False
    for subj_dir in intermediates.iterdir():
        if not subj_dir.is_dir() or subj_dir.name in ("pre", "post"):
            continue
        out = subj_dir / step_stem / "output.json"
        if not out.exists():
            continue
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") == "error":
            return True
    return False


# ============================================================================
# 报告生成
# ============================================================================


def _collect_degradation_warnings(task_dir: Path) -> list[str]:
    """扫描中间产物，收集 warn 级降级项（ADR 0014 哨兵）。

    与 abort 级哨兵（步骤失败）不同，这些是「步骤成功但结果为空」的信号，
    可能是合法冷启动（如第一篇无 history），也可能是闭环断裂（标签自更新
    失效）——只标注不中断，写入报告 + 终端双呈现。

    信号：
      1. history 参考恒空（05-batch-search 的 history_count 全 0）
      2. 技术特征恒空（04-extract-features 的 feature_count 全 0）
      3. 标签写回 0 篇 / 池提升失败或 0 篇（09-archive-reports 的 tags_written / promote_error / promoted）
      4. 评分标签缺失（06-direct-scoring 的 data.tags 空）
      5. L3 技术特征覆盖率低（04-extract-features 汇总的 l3_coverage < 50%）
      6. 评分证据降级（08-summarize 的 evidence 字段级标记：rationale/tags 缺失）
    """
    warnings: list[str] = []
    intermediates = task_dir / "intermediates"
    if not intermediates.is_dir():
        return warnings

    # subject 目录：排除 pre/post 两个 batch 汇总目录
    subject_dirs = [
        d for d in intermediates.iterdir() if d.is_dir() and d.name not in ("pre", "post")
    ]

    def _load(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    # 1. history 恒空
    checked = False
    all_empty = True
    for sd in subject_dirs:
        out = sd / "05-batch-search" / "output.json"
        if not out.exists():
            continue
        checked = True
        if (_load(out).get("data", {}).get("history_count") or 0) > 0:
            all_empty = False
            break
    if checked and all_empty:
        warnings.append("历史参考恒空（05-batch-search 的 history 池无已审论文可比对）")

    # 2. 技术特征恒空（feature_count = LLM 抽取 + 词表兜底并集，L3 数据源）
    checked = False
    all_empty = True
    for sd in subject_dirs:
        out = sd / "04-extract-features" / "output.json"
        if not out.exists():
            continue
        checked = True
        if (_load(out).get("data", {}).get("feature_count") or 0) > 0:
            all_empty = False
            break
    if checked and all_empty:
        warnings.append(
            "技术特征恒空（04-extract-features 的 LLM 抽取 + 词表兜底均无产出，L3 检索失效）"
        )

    # 3. tags_written == 0 / promoted == 0（09-archive-reports 写回闭环）
    archive_out = intermediates / "post" / "09-archive-reports" / "output.json"
    if archive_out.exists():
        archive_data = _load(archive_out).get("data", {})
        if (archive_data.get("tags_written") or 0) == 0:
            warnings.append("标签写回 0 篇（09-archive-reports tags_written=0，标签库不会自更新）")
        promote_error = archive_data.get("promote_error")
        if promote_error:
            warnings.append(
                f"池提升失败（09-archive-reports promote_to_history 异常: {promote_error}）"
            )
        elif (archive_data.get("promoted") or 0) == 0 and (
            archive_data.get("pending_before") or 0
        ) > 0:
            warnings.append(
                "池提升 0 篇（09-archive-reports promoted=0 但存在 pending，history 池无新增 Reference）"
            )

    # 4. data.tags 缺失
    checked = False
    all_missing = True
    for sd in subject_dirs:
        out = sd / "06-direct-scoring" / "output.json"
        if not out.exists():
            continue
        checked = True
        if _load(out).get("data", {}).get("tags"):
            all_missing = False
            break
    if checked and all_missing:
        warnings.append("评分标签缺失（06-direct-scoring 的 data.tags 为空，标签写回数据源不可用）")

    # 5. L3 技术特征覆盖率低（ADR 0015：冷启动期间 L3 稀疏，需显式暴露不静默退化）
    feat_out = intermediates / "pre" / "04-extract-features" / "output.json"
    if feat_out.exists():
        data = _load(feat_out).get("data", {})
        total = data.get("l3_total") or 0
        covered = data.get("l3_covered") or 0
        if total > 0 and covered / total < 0.5:
            warnings.append(
                f"L3 技术特征覆盖率低（{covered}/{total} 篇有 features，L3 检索大面积退化）"
            )

    # 6. rationale/tags 证据降级（基于 08-summarize 的 evidence 字段，字段级精确到维度）
    degraded = []
    for sd in subject_dirs:
        problems = _subject_evidence_problems(sd)
        if problems:
            degraded.append(f"{sd.name}（{'；'.join(problems)}）")
    if degraded:
        warnings.append(f"评分证据降级 {len(degraded)} 篇：" + "；".join(degraded))

    return warnings


def _subject_evidence_problems(subject_dir: Path) -> list[str]:
    """单篇的 08-summarize evidence 降级问题列表（缺证据:<维度> / tags缺失）。"""
    out = subject_dir / "08-summarize" / "output.json"
    if not out.exists():
        return []
    try:
        ev = json.loads(out.read_text(encoding="utf-8")).get("data", {}).get("evidence", {})
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(ev, dict):
        return []
    problems = []
    missing = ev.get("rationale_missing") or []
    if missing:
        problems.append("缺证据:" + "/".join(str(d) for d in missing))
    if ev.get("tags_missing"):
        problems.append("tags缺失")
    return problems


@dataclass
class FixReport:
    """一个已完成批次里「有问题的篇目」清单（逐篇 ERROR / WARN，供 --fix-warn）。"""

    error_subjects: list[str]  # Review 步骤产物 status=error 的篇目
    warn_subjects: dict[str, list[str]]  # 篇目 → 降级类型（缺证据:… / tags缺失）

    @property
    def all_subjects(self) -> list[str]:
        seen: list[str] = []
        for s in self.error_subjects:
            if s not in seen:
                seen.append(s)
        for s in self.warn_subjects:
            if s not in seen:
                seen.append(s)
        return seen

    @property
    def error_count(self) -> int:
        return len(self.error_subjects)

    @property
    def warn_count(self) -> int:
        return len(self.warn_subjects)

    @property
    def problem_count(self) -> int:
        return len(self.all_subjects)


def collect_fix_subjects(task_dir: Path) -> FixReport:
    """扫描已完成批次的逐篇问题（--fix-warn 数据源）。

    - ERROR：manifest.steps（Review 阶段步骤全集）产物 status=error 的篇目；
    - WARN：08-summarize evidence 字段级降级（缺证据/tags缺失）。
    """
    manifest = read_task_manifest(task_dir)
    steps = set(manifest.get("steps") or [])
    intermediates = task_dir / "intermediates"
    if not intermediates.is_dir():
        return FixReport(error_subjects=[], warn_subjects={})

    subject_dirs = [
        d for d in sorted(intermediates.iterdir()) if d.is_dir() and d.name not in ("pre", "post")
    ]

    error_subjects: list[str] = []
    warn_subjects: dict[str, list[str]] = {}

    for sd in subject_dirs:
        # ERROR：Review 步骤产物 status=error
        if steps:
            for step_dir in sorted(sd.iterdir()):
                if not step_dir.is_dir() or step_dir.name not in steps:
                    continue
                out = step_dir / "output.json"
                if not out.exists():
                    continue
                try:
                    data = json.loads(out.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if data.get("status") == "error":
                    error_subjects.append(sd.name)
                    break
        # WARN：08-summarize evidence
        problems = _subject_evidence_problems(sd)
        if problems:
            warn_subjects[sd.name] = problems

    return FixReport(error_subjects=error_subjects, warn_subjects=warn_subjects)


# ============================================================================
# Agent 观测（Agent Observation）——agent-stats 记录（ADR 0018）
# ============================================================================

# 降级哨兵 → 稳定短 key（与 _collect_degradation_warnings 的哨兵一一对应）
_DEGRADATION_KINDS = {
    "历史参考恒空": "history_empty",
    "技术特征恒空": "features_empty",
    "标签写回 0 篇": "tags_written_zero",
    "池提升失败": "promote_error",
    "池提升 0 篇": "promote_zero",
    "评分标签缺失": "score_tags_missing",
    "L3 技术特征覆盖率低": "l3_coverage_low",
    "评分证据降级": "evidence_degraded",
}


def _degradation_kind(warning: str) -> str:
    for prefix, key in _DEGRADATION_KINDS.items():
        if warning.startswith(prefix):
            return key
    return "degradation"


def _as_int(value: object, default: int = 0) -> int:
    """宽容转 int：None/非数字/损坏值回退默认（agent-stats 合并不因脏数据崩溃）。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _record_agent_stats(
    config: PipelineConfig,
    pipeline_dir: Path,
    all_step_results: list[StepResult],
    degradation_warnings: list[str],
    data_dir: str,
    record_degradation: bool = True,
) -> None:
    """把本次运行的 agent 步骤异常 + 降级记录进 agent-stats.json（ADR 0018）。

    - 分母 = .md 步骤执行数（跨所有 phase）；分子 = 其中 status=error 的。
    - 降级哨兵（缺证据等）作为额外异常计入（不增 total_steps）。
    - 按管线目录名分桶；agent 段指纹变化 → 该管线计数清零。
    - record_degradation=False 时跳过降级哨兵记录：续做（resume/fix-warn）
      复用产物后降级哨兵是重算的全量快照，重复计入会污染「异常占比」
      （.md 步骤已有 resumed 守卫，降级哨兵需同级别门控）。
    """
    md_steps: set[str] = set()
    for phase in config.phases:
        for sf in discover_steps(pipeline_dir / phase.directory):
            if sf.step_type == "md":
                md_steps.add(sf.stem)

    recorder = AgentStatsRecorder()
    for r in all_step_results:
        if r.step_name not in md_steps:
            continue
        if r.resumed:
            # 续做复用产物（未执行）→ 不计入统计，避免跨 resume 重复计数
            continue
        if r.attempt_history:
            # 按 attempt 逐次计数（ADR 0018 by_command）：升级链中间失败不再被
            # 「最后一次尝试」归因掩盖。
            for a in r.attempt_history:
                ok = bool(a["ok"])
                kind = "" if ok else classify_kind(a["error"] or "")
                recorder.record(ok=ok, kind=kind, command=a["command"])
        else:
            # 无 per-attempt 记录（老数据/非 _retry_step 路径）回退单次记录
            ok = r.status != "error"
            kind = "" if ok else classify_kind(r.error or "")
            recorder.record(ok=ok, kind=kind, command=r.command)

    if record_degradation:
        for w in degradation_warnings:
            recorder.record_anomaly(f"degradation:{_degradation_kind(w)}")

    pipeline_name = pipeline_dir.name
    fingerprint = compute_fingerprint(config.agent)
    stats_path = Path(data_dir) / "agent-stats.json"
    data = load_stats(stats_path)
    pipelines = data.setdefault("pipelines", {})
    slot = pipelines.get(pipeline_name)
    if not isinstance(slot, dict) or slot.get("fingerprint") != fingerprint:
        # 首次记录或 agent 段指纹变化 → 清零重来
        slot = default_stats(fingerprint)
    slot["total_steps"] = _as_int(slot.get("total_steps")) + recorder.total_steps
    slot["total_anomalies"] = _as_int(slot.get("total_anomalies")) + recorder.total_anomalies
    for k, v in recorder.by_kind.items():
        slot.setdefault("by_kind", {})[k] = slot.get("by_kind", {}).get(k, 0) + v
    for cmd, c in recorder.by_command.items():
        cmd_slot = slot.setdefault("by_command", {}).setdefault(cmd, {"steps": 0, "anomalies": 0})
        cmd_slot["steps"] += c["steps"]
        cmd_slot["anomalies"] += c["anomalies"]
    pipelines[pipeline_name] = slot
    save_stats(stats_path, data)


def _generate_report(
    report_path: Path,
    task_id: str,
    pipeline_name: str,
    all_phase_results: dict,
    all_step_results: list[StepResult],
    success: bool,
    phase_display: dict[str, str] | None = None,
    degradation_warnings: list[str] | None = None,
) -> str:
    """生成最终报告 markdown 文件，返回 CLI 可输出的结构化结论摘要。"""
    import datetime

    def _phase_label(phase_name: str) -> str:
        """阶段显示名：显式 display_name 优先，否则 name 首字母大写回退。"""
        return (phase_display or {}).get(phase_name, phase_name.capitalize())

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 统计 ──
    total = len(all_step_results)
    ok_count = sum(1 for r in all_step_results if r.status == "ok")
    skipped_count = sum(1 for r in all_step_results if r.status == "skipped")
    error_count = sum(1 for r in all_step_results if r.status == "error")

    lines = [
        "# 论文评审报告",
        "",
        f"- **Task ID**: `{task_id}`",
        f"- **Pipeline**: {pipeline_name}",
        f"- **时间**: {now}",
        f"- **状态**: {'✅ 通过' if success else '❌ 有错误'}",
        f"- **步骤统计**: {total} 步（✅ {ok_count} / ⚠️ {skipped_count} / ❌ {error_count}）",
    ]

    # ── warn 级降级项（结果为空信号，ADR 0014 哨兵）──
    if degradation_warnings:
        lines.append("")
        lines.append("## ⚠ 降级项（结果为空，可能影响评审质量）")
        lines.append("")
        for w in degradation_warnings:
            lines.append(f"- ⚠ {w}")
        lines.append("")

    # ── 结论部件（按 subject 聚合）──
    subject_summaries: dict[str, dict] = {}  # subject → {status, steps, errors}
    for phase_name, phase_results in all_phase_results.items():
        for subject_name, subj_results in phase_results.items():
            if subject_name == "_batch_":
                continue
            if subject_name not in subject_summaries:
                subject_summaries[subject_name] = {
                    "phase": phase_name,
                    "steps": [],
                    "ok": 0,
                    "error": 0,
                    "skipped": 0,
                }
            for sr in subj_results:
                subject_summaries[subject_name]["steps"].append(sr)
                if sr.status == "ok":
                    subject_summaries[subject_name]["ok"] += 1
                elif sr.status == "error":
                    subject_summaries[subject_name]["error"] += 1
                else:
                    subject_summaries[subject_name]["skipped"] += 1

    # ── 按阶段输出详情 ──
    for phase_name, phase_results in all_phase_results.items():
        lines.append(f"## {_phase_label(phase_name)} 阶段")
        lines.append("")
        for subject_name, subj_results in phase_results.items():
            is_batch = subject_name == "_batch_"
            if not is_batch:
                lines.append(f"### {subject_name}")
                lines.append("")
            for sr in subj_results:
                status_icon = "✅" if sr.status == "ok" else "⚠️" if sr.status == "skipped" else "❌"
                lines.append(f"- {status_icon} **{sr.step_name}**: {sr.status}")
                if sr.error:
                    lines.append(f"  - 错误: {sr.error}")
                if sr.data and any(v for v in sr.data.values() if v):
                    # 跳过 raw_output 避免报告臃肿（可到 intermediates 目录查看原文）
                    slim_data = {k: v for k, v in sr.data.items() if k != "raw_output" and v}
                    if slim_data:
                        lines.append(
                            f"  - 数据: {json.dumps(slim_data, ensure_ascii=False, indent=4)}"
                        )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"> 完整中间产物见: `{report_path.parent / 'intermediates'}`")

    report_dir = report_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ── CLI 结论：简洁的 per-subject 汇总 ──
    conclusion_lines = []
    subject_count = 0
    for subject_name, info in subject_summaries.items():
        subject_count += 1
        icon = "✅" if info["error"] == 0 else "❌"
        parts = [f"  {icon} {subject_name}"]
        detail_parts = []
        if info["ok"]:
            detail_parts.append(f"{info['ok']} ok")
        if info["skipped"]:
            detail_parts.append(f"{info['skipped']} skipped")
        if info["error"]:
            detail_parts.append(f"{info['error']} error")
        if detail_parts:
            parts.append(f"（{', '.join(detail_parts)}）")
        conclusion_lines.append("".join(parts))

    # 批量阶段
    for phase_name, phase_results in all_phase_results.items():
        if "_batch_" in phase_results:
            batch = phase_results["_batch_"]
            batch_ok = sum(1 for r in batch if r.status == "ok")
            batch_err = sum(1 for r in batch if r.status == "error")
            status_icon = "✅" if batch_err == 0 else "❌"
            conclusion_lines.insert(
                0, f"{_phase_label(phase_name)}: {status_icon} {batch_ok}/{len(batch)} 步通过"
            )

    summary = f"共 {len(all_step_results)} 步（✅ {ok_count} / ❌ {error_count}）"
    if subject_count:
        summary += f"，{subject_count} 篇论文"
    conclusion_lines.insert(0, summary)
    conclusion_lines.append(f"\n完整报告: {report_path}")

    return "\n".join(conclusion_lines)


# ============================================================================
# Task manifest（任务状态机）
# ============================================================================


def read_task_manifest(task_dir: Path) -> dict:
    """读取 task.json（容错：缺失/损坏返回空 dict，不抛异常）。"""
    manifest_path = Path(task_dir) / "task.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_task_manifest(task_dir: Path, **updates) -> None:
    """写/更新 task.json（Task Status 状态机）。

    读取已有内容合并更新后，临时文件 + rename 原子写回（避免中断时写坏）。
    status 取值：running / done / interrupted / abandoned。
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "task.json"
    manifest = read_task_manifest(task_dir)
    manifest.update(updates)
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(manifest_path)


def _infer_last_interrupted_step(task_dir: Path) -> str | None:
    """从 intermediates 推断中断位置：最后写入 output.json 的 subject/step。"""
    intermediates = Path(task_dir) / "intermediates"
    if not intermediates.is_dir():
        return None
    latest: tuple[float, str] | None = None
    for subject_dir in sorted(intermediates.iterdir()):
        if not subject_dir.is_dir():
            continue
        for step_dir in subject_dir.iterdir():
            out = step_dir / "output.json"
            if out.is_file():
                mtime = out.stat().st_mtime
                if latest is None or mtime > latest[0]:
                    latest = (mtime, f"{subject_dir.name}/{step_dir.name}")
    return latest[1] if latest else None


# ============================================================================
# 未完成任务检测（Resume 前置）
# ============================================================================

# 任务目录名格式：YYYYMMDD-HHMMSS-<hash>
_TASK_DIR_NAME_RE = re.compile(r"^\d{8}-\d{6}-")


def detect_unfinished_tasks(output_dir: Path) -> list[Path]:
    """扫描 result/ 下未完成（status ∈ {running, interrupted}）的任务目录，最近优先。

    - 无 task.json 的目录：视为未完成（中断发生在运行开始写 manifest 之前，或
      旧版本在完成时才写 task.json、中断未写入）——按目录名时间排序；
    - task.json 存在但无 status 字段：旧版本完成时写入的格式（当时无状态机，
      旧代码只在运行收尾写 task.json）——视为已完成，排除；
    - task.json 损坏/为空：无法判定状态，保守视为未完成（宁可多提示）；
    - status=done / abandoned：排除。
    """
    result_root = Path(output_dir) / "result"
    if not result_root.is_dir():
        return []

    unfinished: list[Path] = []
    for task_dir in sorted(result_root.iterdir()):
        if not task_dir.is_dir():
            continue
        # 仅接受任务目录命名（YYYYMMDD-HHMMSS-xxx）：result/ 下的杂物/备份目录
        # 无 task.json 会被误判为未完成并触发续做提示。老版本任务目录沿用同一
        # task_id 格式，不受影响。
        if not _TASK_DIR_NAME_RE.match(task_dir.name):
            continue
        manifest_path = task_dir / "task.json"
        if not manifest_path.exists():
            unfinished.append(task_dir)
            continue
        manifest = read_task_manifest(task_dir)
        if not manifest:
            # task.json 存在但损坏/为空：无法判定状态，保守视为未完成
            unfinished.append(task_dir)
            continue
        status = manifest.get("status")
        if status in ("running", "interrupted"):
            unfinished.append(task_dir)
        # status=None 且可解析 = 旧版本收尾写入的格式（无状态机）→ 已完成，排除

    # 最近优先：Task ID 前缀是 YYYYMMDD-HHMMSS，可按目录名倒序（缺失名时兜底 mtime）
    unfinished.sort(key=lambda d: (d.name, d.stat().st_mtime), reverse=True)
    return unfinished


def list_done_tasks(output_dir: Path) -> list[Path]:
    """扫描 result/ 下 status=done 的任务目录，最近优先（--fix-warn 数据源）。"""
    result_root = Path(output_dir) / "result"
    if not result_root.is_dir():
        return []
    done: list[Path] = []
    for task_dir in sorted(result_root.iterdir(), reverse=True):
        if not task_dir.is_dir() or not _TASK_DIR_NAME_RE.match(task_dir.name):
            continue
        manifest = read_task_manifest(task_dir)
        if manifest.get("status") == "done":
            done.append(task_dir)
    return done


# ============================================================================
# 主入口 — 薄编排层
# ============================================================================


def run_pipeline(
    pipeline_yaml: dict | str | Path,
    input_path: Path,
    pipeline_dir: Path | None = None,
    output_dir: Path | None = None,
    data_dir: str | None = None,
    target_phase: str | None = None,
    target_step: str | None = None,
    pool_progress: PoolProgress | None = None,
    resume_task_dir: Path | None = None,
    allow_degraded: bool = False,
    only_subjects: list[str] | None = None,
    recompute_review: bool = False,
    archive: bool = True,
) -> PipelineResult:
    """执行一条完整的 pipeline。

    Args:
        pipeline_yaml: pipeline 配置字典、YAML 文件路径或包含 pipeline.yaml 的目录。
        input_path: 输入 PDF 路径（单篇）或目录（多篇）。
        pipeline_dir: pipeline 定义目录的根。为 None 时从 pipeline_yaml 推断。
        output_dir: 覆盖配置中的 output_dir。
        data_dir: 数据目录（用于默认 output_dir 解析）。
        target_phase: 仅运行指定阶段（匹配 phases[].name）。
        target_step: 仅运行指定步骤名。
        pool_progress: 可选的 PoolProgress 实例。
        resume_task_dir: 续做（Resume）时复用前序 task 目录——原地续做：
            已完成 Steps（有 output.json）跳过，产物合并进原 task。
        allow_degraded: 显式降级开关（ADR 0014 哨兵）。默认 False——batch
            phase（pre/post）有 step 失败时中断管线，避免静默降级；传 True
            时失败后继续执行后续 phase（但 overall_success 仍为 False）。
        only_subjects: 只运行这些 subject（--fix-warn 用）。None = 全量。
            过滤只作用于 per_subject 阶段；manifest.subjects 与 resume 门控
            仍用全量列表（任务归属不变）。
        recompute_review: True 时续做不复用已完成 review 步骤产物（重跑）。
            --fix-warn 需重跑问题篇目，故置 True；Pre 仍按门控复用。
        archive: True（默认）时执行 Post 写回（标签库/history 池）；False 跳过。

    Returns:
        PipelineResult
    """
    # ── 配置解析 ──
    if isinstance(pipeline_yaml, (str, Path)):
        yaml_path = Path(pipeline_yaml)
        if yaml_path.is_dir():
            yaml_file = yaml_path / "pipeline.yaml"
            if yaml_file.exists():
                config = PipelineConfig.from_path(yaml_path)
                if pipeline_dir is None:
                    pipeline_dir = yaml_path
            else:
                config = PipelineConfig.from_dict(
                    {
                        "name": "default",
                        "phases": [
                            {"name": "review", "mode": "per_subject", "directory": str(yaml_path)}
                        ],
                    }
                )
        elif yaml_path.suffix in (".yaml", ".yml"):
            config = PipelineConfig.from_path(yaml_path)
            if pipeline_dir is None:
                pipeline_dir = yaml_path.parent
        else:
            config = PipelineConfig.from_dict(
                {
                    "name": "default",
                    "phases": [
                        {"name": "review", "mode": "per_subject", "directory": str(yaml_path)}
                    ],
                }
            )
    else:
        config = PipelineConfig.from_dict(pipeline_yaml)

    if output_dir:
        config.output_dir = output_dir
    elif data_dir:
        from paper_review.config import resolve_data_dir

        config.output_dir = resolve_data_dir(data_dir) / "output"
    if pipeline_dir is None:
        pipeline_dir = Path.cwd()

    # ── 任务指纹 ──
    import datetime
    import hashlib

    now = datetime.datetime.now()
    resume_mode = resume_task_dir is not None
    if resume_mode:
        # 续做：复用前序 task 目录与 ID（原地续做，产物合并进原 task）
        task_dir = Path(resume_task_dir)
        task_id = task_dir.name
        logger.info("Resuming task: %s → %s", task_id, task_dir)
    else:
        task_seed = f"{input_path.absolute()}:{now.isoformat()}"
        hash_suffix = hashlib.sha256(task_seed.encode()).hexdigest()[:8]
        task_id = now.strftime("%Y%m%d-%H%M%S") + "-" + hash_suffix
        task_dir = config.output_dir / "result" / task_id
        logger.info("Task ID: %s → %s", task_id, task_dir)

    resolved_data_dir = data_dir or os.environ.get("PAPER_REVIEW_DATA_DIR")
    if resolved_data_dir is None:
        from paper_review.config import resolve_data_dir

        resolved_data_dir = str(resolve_data_dir())

    base_env = {
        **os.environ,
        "PIPELINE_OUTPUT_DIR": str(config.output_dir.absolute()),
        "PIPELINE_PIPELINE_DIR": str(pipeline_dir.absolute()),
        "PIPELINE_DATA_DIR": resolved_data_dir,
        "PIPELINE_RESULT_DIR": str(task_dir),
        "PIPELINE_INPUT_PATH": str(input_path.absolute()),
    }

    # 全局 agent 配置注入（step 内调 Agent 的 type/provider/model/escalate）
    base_env[ENV_AGENT_TYPE] = config.agent.type
    if config.agent.provider:
        base_env[ENV_AGENT_PROVIDER] = config.agent.provider
    if config.agent.model:
        base_env[ENV_AGENT_MODEL] = config.agent.model
    if config.agent.escalate:
        base_env[ENV_AGENT_ESCALATE] = json.dumps(config.agent.escalate)

    # ── Index 配置注入 step 环境变量 ──
    from paper_review.auto_index import resolve_index_config

    raw_yaml: dict | None = None
    if isinstance(pipeline_yaml, (str, Path)):
        yaml_path = Path(pipeline_yaml)
        if yaml_path.is_dir():
            yaml_file = yaml_path / "pipeline.yaml"
        elif yaml_path.suffix in (".yaml", ".yml"):
            yaml_file = yaml_path
        else:
            yaml_file = None
        if yaml_file and yaml_file.exists():
            import yaml as _yaml

            raw_yaml = _yaml.safe_load(yaml_file.read_text())
    elif isinstance(pipeline_yaml, dict):
        raw_yaml = pipeline_yaml

    idx_cfg = resolve_index_config(raw_yaml, Path(resolved_data_dir))
    base_env.update(
        {
            "PIPELINE_INDEX_STORE_DIR": str(idx_cfg.store_dir),
            "PIPELINE_INDEX_REFERENCE_DIR": str(idx_cfg.reference_dir),
            "PIPELINE_INDEX_AUTO_INDEX": "1" if idx_cfg.auto_index else "0",
            "PIPELINE_INDEX_COPY_SUBJECTS": "1" if idx_cfg.copy_subjects else "0",
        }
    )

    # ── 生效的阶段 ──
    active_phases = (
        [p for p in config.phases if p.name == target_phase]
        if target_phase
        else [p for p in config.phases if p.directory]
    )
    # --fix-warn --fix-skip-archive：不写回 → 跳过 Post（最后一个 per_subject 之后的 batch 阶段）
    if not archive and not target_phase:
        last_ps = max(
            (i for i, p in enumerate(active_phases) if p.mode == "per_subject"), default=-1
        )
        active_phases = [p for i, p in enumerate(active_phases) if p.mode != "batch" or i < last_ps]

    all_phase_results: dict[str, dict[str, list[StepResult]]] = {}
    all_step_results: list[StepResult] = []
    overall_success = True

    if not active_phases:
        logger.warning("No active phases — nothing to run")
        _generate_report(
            task_dir / "report.md",
            task_id,
            config.name,
            all_phase_results,
            all_step_results,
            overall_success,
            {p.name: p.display_label for p in config.phases},
        )
        conclusion = _build_cli_tree(
            task_id, config.name, config, all_phase_results, pipeline_dir, task_dir
        )
        # 早退路径同样写 manifest（done）：否则 result/{task_id}/ 下只有 report.md
        # 无 task.json，detect_unfinished_tasks 会把它当未完成任务在下次 review 时提示续做。
        write_task_manifest(
            task_dir,
            task_id=task_id,
            status="done",
            created_at=now.isoformat(),
            pipeline=config.name,
            input=str(input_path.resolve()),
            subjects=[],
            steps=[],
            success=True,
            step_count=0,
            error_count=0,
        )
        return PipelineResult(
            subject="",
            success=overall_success,
            step_results=all_step_results,
            task_id=task_id,
            task_dir=task_dir,
            conclusion=conclusion,
        )

    # ── Subject 发现 ──
    subjects = discover_subjects(config, input_path, config.output_dir)
    # --fix-warn：只跑问题篇目。全量列表保留给 resume 门控与 manifest.subjects
    # （任务归属不变，仅本次执行的 per_subject 阶段被过滤）。
    all_subjects = subjects
    if only_subjects is not None:
        only_set = set(only_subjects)
        subjects = [s for s in subjects if s in only_set]
    primary_subject = subjects[0] if subjects else ""

    # ── Resume：Pre Phase 跳过判定 ──
    # 必须先读旧 manifest 再写 running——否则写入会覆盖旧 subjects，判定恒真。
    # 跳过条件（三者全满足）：
    #   1) 前序 Pre 产物确证：Pre 阶段最后一个 step 的 output.json 存在——只比对
    #      subjects 会在“中断发生在 Pre 阶段”时误跳过未完成的 Pre（manifest.subjects
    #      是 Pre 运行前写入的，resume 时 discover 对相同输入目录返回同一列表，
    #      等式恒真，无法区分 Pre 是否真正完成）；
    #   2) 前序 manifest subjects 与当前发现一致；
    #   3) 前序 input 与当前输入路径一致（同名不同目录时禁止复用 Pre 产物）。
    prev_manifest: dict = {}
    skip_pre_phase = False
    # Resume 续做时 review 步骤是否跳过已完成产物（有 output.json 且 ok/skipped）。
    # 与 Pre 跳过同一门控（input + subjects 一致才允许复用产物）：
    # 换输入目录后续做时禁止跳过——同名 subject（如不同目录下的 paper.pdf）
    # 的旧产物会被静默复用，跨批次混批。pipeline 名不参与（续做使用当前配置）。
    resume_skip_completed = False
    # T3: Resume 时 Pre 步骤级完成判定——产物存在且 status ok/skipped 的步骤名集合。
    # 部分完成的 Pre 不再全量重跑：已完成步骤跳过，未完成步骤带 SKIP_EXISTING 续做。
    pre_phase: PhaseConfig | None = None
    pre_steps: list[StepFile] = []
    pre_step_skip: set[str] = set()
    if resume_mode:
        prev_manifest = read_task_manifest(task_dir)
        # Pre 阶段 = 首个 per_subject 阶段之前的 batch 阶段。曾取 active_phases 中
        # 第一个 batch 阶段——对无 pre 的 [review, post] 管线会误判为 post，
        # 续做时跳过 Post（回归）；--phase post 时同样把用户显式要求的阶段跳过。
        per_subject_idx = next(
            (i for i, p in enumerate(active_phases) if p.mode == "per_subject"), None
        )
        pre_phase = (
            next((p for p in active_phases[:per_subject_idx] if p.mode == "batch"), None)
            if per_subject_idx is not None
            else None
        )
        if pre_phase is not None:
            pre_steps = discover_steps(pipeline_dir / pre_phase.directory)
            for sf in pre_steps:
                step_out = task_dir / "intermediates" / pre_phase.name / sf.stem / "output.json"
                # 与 review 步骤跳过同一原则：产物存在且 status 为 ok/skipped 才算完成。
                # 曾只判存在——status=error 的失败产物（如 index 构建失败）被静默跳过，
                # 续做时失败状态被永久固化（与 ADR 0005“失败产物续做时重跑”原则矛盾）。
                if step_out.exists():
                    try:
                        out = json.loads(step_out.read_text(encoding="utf-8"))
                        if out.get("status", "ok") in ("ok", "skipped"):
                            # 步骤级 ok 还不够：per-subject 产物存在 status=error 的篇
                            # （如 05 单篇检索失败）时不得跳过该步骤——失败篇需在续做时
                            # 重跑（ADR 0005），否则空引用被静默固化到评审。
                            if _per_subject_step_has_error(task_dir, sf.stem):
                                logger.info(
                                    "Resume: step '%s' has error per-subject products — will re-run",
                                    sf.stem,
                                )
                            else:
                                pre_step_skip.add(sf.stem)
                    except (json.JSONDecodeError, OSError):
                        logger.debug(
                            "pre step product unreadable: %s (treated as incomplete)", step_out
                        )
        prev_input = prev_manifest.get("input")
        input_matches = not prev_input or str(Path(prev_input).resolve()) == str(
            input_path.resolve()
        )
        if prev_input and not input_matches:
            logger.warning(
                "Resume: previous task input was %r, current input is %r — "
                "Pre will be re-run to avoid mixing batches",
                prev_input,
                str(input_path),
            )
        subjects_match = sorted(prev_manifest.get("subjects", [])) == sorted(all_subjects)
        resume_skip_completed = (
            resume_mode and input_matches and subjects_match and not recompute_review
        )
        # 整个 Pre 完成 = 所有步骤产物均 ok（逐步骤判定，非旧“判最后一步”）
        pre_complete = bool(pre_steps) and len(pre_step_skip) == len(pre_steps)
        skip_pre_phase = pre_complete and input_matches and subjects_match
        if resume_mode and not resume_skip_completed:
            reason = "fix-warn recompute" if recompute_review else "input or subjects mismatch"
            logger.warning(
                "Resume: review-step skip disabled (%s) — existing products will be re-run",
                reason,
            )
        if skip_pre_phase:
            logger.info(
                "Resume: Pre phase will be skipped (products verified + subjects/input match)"
            )
        elif pre_step_skip and input_matches and subjects_match:
            logger.info(
                "Resume: Pre step-level resume — %d/%d step(s) already done: %s",
                len(pre_step_skip),
                len(pre_steps),
                ", ".join(sorted(pre_step_skip)) or "none",
            )

    # ── Task manifest：标记运行开始（未完成任务的检测基础） ──
    # steps 记录 per_subject 步骤全集（不被 target_step 过滤）：Resume 摘要的完成
    # 进度以它为基准——磁盘并集在“中断于首步”时会缩小，导致部分完成被高估为完成。
    _review_steps_all: list[str] = []
    for _p in active_phases:
        if _p.mode == "per_subject":
            _review_steps_all = [sf.stem for sf in discover_steps(pipeline_dir / _p.directory)]
            break
    write_task_manifest(
        task_dir,
        task_id=task_id,
        status="running",
        # resume 保留原发起时间（created_at 属于任务发起时刻，不是续做时刻）
        created_at=prev_manifest.get("created_at") or now.isoformat(),
        pipeline=config.name,
        input=str(input_path.resolve()),
        subjects=all_subjects,
        steps=_review_steps_all,
    )

    # ── SIGINT 优雅中断：尽力写 interrupted 状态（Resume 检测依赖） ──
    # 仅主线程有效；kill -9/断电 等硬中断不经过此 handler——状态推断（running 无 done）兜底。
    _restore_sigint: Callable[[], None] | None = None
    # 进度卡引用（pp 创建后填充）：SIGINT 处理器内仅做轻量赋值标记中断。
    _pp_ref: list[PipelineProgress | None] = [None]
    if threading.current_thread() is threading.main_thread():
        import signal as _signal

        _prev_int = _signal.getsignal(_signal.SIGINT)

        def _sigint_mark_interrupted(signum, frame):
            # 信号处理器内做文件 I/O 属尽力而为：任何异常不得替换 KeyboardInterrupt。
            # 注意：此处禁止 logging——若 SIGINT 恰在主线程执行 logging 调用（持有
            # logging 模块内部锁）期间到达，handler 内再次 logging 会死锁而非抛异常。
            try:
                write_task_manifest(
                    task_dir,
                    status="interrupted",
                    interrupted_at_step=_infer_last_interrupted_step(task_dir),
                )
            except Exception as e:
                # 尽力而为：写失败依赖“running 无 done”状态推断兜底。
                # 禁止在此 logging——信号处理器内 logging 可能死锁（logging 锁
                # 非 async-signal-safe），故仅消费异常不记录。
                _ = e
            _signal.signal(_signal.SIGINT, _prev_int)
            # 标记中断（轻量赋值，无 I/O/锁，信号处理器内安全）。
            # 注意：CPython 的 Python 级信号处理器同样延迟到主线程 eval loop
            # （C 扩展返回后）才执行，无法在 ONNX/PyMuPDF 阻塞期间抢先停止
            # spinner——此处仅保证中断抛出后渲染的是“已中断”而非“进行中/完成”态。
            if _pp_ref[0] is not None:
                _pp_ref[0].mark_interrupted()
            raise KeyboardInterrupt

        _signal.signal(_signal.SIGINT, _sigint_mark_interrupted)

        def _restore_sigint_impl() -> None:
            _signal.signal(_signal.SIGINT, _prev_int)

        _restore_sigint = _restore_sigint_impl

    # ── Pool 环境变量覆盖 ──
    per_subject_phases = [p for p in active_phases if p.mode == "per_subject"]
    first_per_subject = per_subject_phases[0] if per_subject_phases else None
    if pool_progress is None and first_per_subject and first_per_subject.pool:
        for env_key, attr in [
            ("PAPER_REVIEW_POOL_WORKERS", "workers"),
            ("PAPER_REVIEW_POOL_TIMEOUT", "timeout"),
        ]:
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    setattr(first_per_subject.pool, attr, int(val))
                except ValueError:
                    pass

    logger.info(
        "Pipeline '%s' — %d phase(s), %d subject(s)", config.name, len(active_phases), len(subjects)
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Executor 构造 ──
    py_runner = PyStepRunner()
    md_executor = MdStepExecutor()

    # ── Progress ──
    phase_infos: list[PhaseProgressInfo] = []
    review_subjects = len(subjects) if subjects else 0
    for phase in active_phases:
        step_count = len(discover_steps(pipeline_dir / phase.directory))
        if phase.mode == "batch":
            phase_infos.append(
                PhaseProgressInfo(
                    name=phase.name,
                    display=phase.display_label,
                    kind="batch",
                    total=step_count,
                )
            )
        elif phase.mode == "per_subject":
            phase_infos.append(
                PhaseProgressInfo(
                    name=phase.name,
                    display=phase.display_label,
                    kind="per_subject",
                    total=step_count * review_subjects,
                    subjects=review_subjects,
                    steps_per=step_count,
                )
            )

    pp = PipelineProgress(phases=phase_infos)
    _pp_ref[0] = pp
    pp.start()

    # ── 阶段遍历 ──
    # 估算阶段超时：从 phase.step_timeout 或动态计算

    for phase in active_phases:
        phase_dir = pipeline_dir / phase.directory
        steps = discover_steps(phase_dir)
        if target_step and target_phase and phase.name == target_phase:
            steps = [s for s in steps if s.stem == target_step]
        if not steps:
            continue

        # 决定该阶段每 step 的超时
        phase_timeout = phase.step_timeout
        if phase_timeout == 0:
            # 动态估算：统计 .md 步骤的文本负载
            md_steps = [s for s in steps if s.step_type == "md"]
            if md_steps:
                # 从 PDF 文件大小估算实际字数（经验比例：1 字节 PDF ≈ 0.35 字符文本）
                # per_subject 模式用最大单篇，batch 模式用总和
                subject_chars_list = _estimate_subject_chars(subjects, config.output_dir)
                if not subject_chars_list:
                    # 无 subject 时用 0（estimate_step_timeout 返回 base 60s）
                    total_chars = 0
                elif phase.mode == "per_subject":
                    total_chars = max(subject_chars_list)
                else:
                    total_chars = sum(subject_chars_list)
                phase_timeout = estimate_step_timeout(
                    step_type="md",
                    total_chars=total_chars,
                    subject_count=len(subjects) if phase.mode == "batch" else 1,
                )
            else:
                phase_timeout = estimate_step_timeout(step_type="py")

        if phase.mode == "batch":
            if skip_pre_phase:
                # 续做：Pre 产物已在前序 task（格式转换/索引已完成），跳过首个 batch 阶段
                logger.info(
                    "Resume: skipping batch phase '%s' (Pre products already exist)", phase.name
                )
                phase_results = {}
                skip_pre_phase = False  # 只跳第一个 batch（Pre）
                all_phase_results[phase.name] = phase_results
                continue
            logger.info(
                "Phase [%s] batch — %d step(s) (timeout=%ds/step)",
                phase.name,
                len(steps),
                phase_timeout,
            )
            # T3: 仅 Pre 阶段参与步骤级跳过/断点续做（post 全量执行）；
            # 门控同 resume_skip_completed（input + subjects 一致才允许复用产物）
            is_pre_resume = (
                resume_skip_completed and pre_phase is not None and phase.name == pre_phase.name
            )
            phase_results = _execute_batch(
                phase=phase,
                steps=steps,
                output_dir=config.output_dir,
                base_env=base_env,
                executor=_make_executor(py_runner, md_executor),
                pp=pp,
                step_timeout=phase_timeout,
                skip_steps=pre_step_skip if is_pre_resume else None,
                resume_skip_existing=is_pre_resume,
            )
            # manifest_step 验证
            if phase.manifest_step:
                batch_results = phase_results.get("_batch_", [])
                manifest_ran = any(r.step_name == phase.manifest_step for r in batch_results)
                if manifest_ran:
                    mr = next(r for r in batch_results if r.step_name == phase.manifest_step)
                    if mr.status == "error":
                        logger.error(
                            "manifest_step '%s' failed in phase '%s'",
                            phase.manifest_step,
                            phase.name,
                        )
                    else:
                        # skipped = resume 时步骤级跳过（产物复用）——不是“成功执行”，
                        # 日志须区分，避免误导排障；manifest 文件来自上次运行仍有效，
                        # subject 重发现照常执行。
                        if mr.status == "skipped":
                            logger.info(
                                "Manifest step '%s' skipped (resume: products reused)",
                                phase.manifest_step,
                            )
                        else:
                            logger.info(
                                "Manifest step '%s' completed successfully", phase.manifest_step
                            )
                        # 刷新 Subject 列表：discover_subjects() 在循环开始前已调用过一次，
                        # 彼时 manifest 尚未生成，per_subject 阶段若以 manifest 为 subject_source
                        # 会静默 fallback 到 CLI 目录扫描（只认 .pdf，漏掉 docx/doc 转换产物）。
                        # manifest_step 跑完后用真实 manifest 重新发现一次，纠正后续阶段用到的列表。
                        full = discover_subjects(config, input_path, config.output_dir)
                        subjects = (
                            full
                            if only_subjects is None
                            else [s for s in full if s in set(only_subjects)]
                        )
                        primary_subject = subjects[0] if subjects else ""
                        # 同步更新进度条：subject 列表变化后总量需要重新计算
                        pp.set_subject_count(len(subjects))
                        # 同步重写 manifest.subjects：运行开始时写入的是 Pre 前的 CLI
                        # 扫描列表（docx/doc 未转换时缺项），续做的 subjects 比对依赖
                        # 真实列表，否则 manifest 来源管线续做被误判为不一致而全量重跑。
                        write_task_manifest(task_dir, subjects=full)
                else:
                    logger.warning(
                        "manifest_step '%s' not found in phase '%s'. Available: %s",
                        phase.manifest_step,
                        phase.name,
                        [s.stem for s in steps],
                    )

        elif phase.mode == "per_subject":
            logger.info(
                "Phase [%s] per_subject — %d step(s), %d subject(s) (timeout=%ds/step)",
                phase.name,
                len(steps),
                len(subjects),
                phase_timeout,
            )
            phase_results = _execute_per_subject(
                phase=phase,
                steps=steps,
                subjects=subjects,
                output_dir=config.output_dir,
                base_env=base_env,
                executor=_make_executor(py_runner, md_executor),
                pool_progress=pool_progress,
                pp=pp,
                step_timeout=phase_timeout,
                skip_completed=resume_skip_completed,
            )
        else:
            logger.warning("Unknown mode '%s' for phase '%s' — skipping", phase.mode, phase.name)
            continue

        # 哨兵：batch phase 有 step 失败且未显式降级 → 中断管线（ADR 0014）。
        # 本次事故根因是「静默降级」——05-batch-search 失败被 on_failure=skip
        # 吞掉，后续 review 照常跑。此处把默认行为改为中断，除非 allow_degraded。
        if (
            not allow_degraded
            and phase.mode == "batch"
            and any(r.status == "error" for r in phase_results.get("_batch_", []))
        ):
            failed = [r.step_name for r in phase_results.get("_batch_", []) if r.status == "error"]
            logger.error(
                "哨兵：phase '%s' 有 step 失败 %s —— 中断管线（传 --allow-degraded 可显式降级继续）",
                phase.name,
                failed,
            )
            overall_success = False
            all_phase_results[phase.name] = phase_results
            for subj_results in phase_results.values():
                all_step_results.extend(subj_results)
            break

        all_phase_results[phase.name] = phase_results
        for subj_results in phase_results.values():
            all_step_results.extend(subj_results)
            for r in subj_results:
                if r.status == "error":
                    overall_success = False

    # ── 完成 ──
    pp.finish()

    # warn 级哨兵：扫描中间产物收集结果空信号（报告 + 终端双呈现）
    degradation_warnings = _collect_degradation_warnings(task_dir)
    if degradation_warnings:
        for w in degradation_warnings:
            logger.warning("降级: %s", w)

    # ── Agent 观测：记录 .md 步骤异常 + 降级（ADR 0018）──
    # 观测是 best-effort 行为：其自身异常不得阻断报告生成与 manifest 收尾。
    try:
        _record_agent_stats(
            config=config,
            pipeline_dir=pipeline_dir,
            all_step_results=all_step_results,
            degradation_warnings=degradation_warnings,
            data_dir=resolved_data_dir,
            record_degradation=not resume_mode,
        )
    except Exception:
        logger.exception("agent-stats 记录失败（观测 best-effort，不阻断主流程）")

    _generate_report(
        task_dir / "report.md",
        task_id,
        config.name,
        all_phase_results,
        all_step_results,
        overall_success,
        {p.name: p.display_label for p in config.phases},
        degradation_warnings,
    )
    conclusion = _build_cli_tree(
        task_id, config.name, config, all_phase_results, pipeline_dir, task_dir
    )

    # ── Task manifest：标记完成 ──
    # 部分运行（--phase/--step）只执行了部分阶段/步骤：任务仍属“未完成”，保持
    # running 供下次 review 检测续做——写 done 会把未完成的阶段永久掩盖。
    partial_run = target_phase is not None or target_step is not None
    write_task_manifest(
        task_dir,
        status="running" if partial_run else "done",
        success=overall_success,
        step_count=len(all_step_results),
        error_count=sum(1 for r in all_step_results if r.status == "error"),
    )
    if _restore_sigint is not None:
        _restore_sigint()

    return PipelineResult(
        subject=primary_subject,
        success=overall_success,
        step_results=all_step_results,
        task_id=task_id,
        task_dir=task_dir,
        conclusion=conclusion,
        degradation_warnings=degradation_warnings,
    )


def _make_executor(py_runner: PyStepRunner, md_executor: MdStepExecutor) -> StepExecutor:
    """创建 StepExecutor adapter。

    将 PyStepRunner + MdStepExecutor 包装为 StepExecutor 协议，
    通过 _execute_step 薄分派实现。供 _retry_step 调用。
    """

    class _Adapter:
        def execute(self, step, step_dir, env, prior_results, subject_name, subject_text=""):
            return _execute_step(
                step,
                step_dir,
                env,
                prior_results,
                subject_name,
                py_runner,
                md_executor,
                subject_text=subject_text,
            )

    return _Adapter()
