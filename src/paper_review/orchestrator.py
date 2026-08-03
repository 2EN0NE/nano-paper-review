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
import os
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, wait
from pathlib import Path

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
from paper_review.progress import PipelineProgress
from paper_review.subject_discovery import discover_subjects

logger = get_logger("orchestrator")


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
) -> StepResult:
    """执行一个 Step，按 RetryConfig 重试。

    唯一的重试实现点——_execute_batch 和 _execute_per_subject 都调用此函数。
    """
    result: StepResult | None = None
    for attempt in range(1, retry_cfg.max_attempts + 1):
        logger.debug(
            "  [%s] step %s attempt %d/%d",
            subject_name,
            step.stem,
            attempt,
            retry_cfg.max_attempts,
        )

        try:
            result = executor.execute(step, step_dir, env, prior_results, subject_name)
            result.subject = subject_name
            result.attempt = attempt

            if result.status in ("ok", "skipped"):
                return result

            logger.warning(
                "  [%s] step %s attempt %d failed: %s",
                subject_name,
                step.stem,
                attempt,
                result.error,
            )
        except Exception as e:
            logger.error(
                "  [%s] step %s attempt %d raised: %s",
                subject_name,
                step.stem,
                attempt,
                e,
            )
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
    return result


# ============================================================================
# _execute_batch — 批量模式
# ============================================================================


def _execute_batch(
    phase: PhaseConfig,
    steps: list[StepFile],
    output_dir: Path,
    base_env: dict,
    executor: StepExecutor,
    pp: PipelineProgress | None = None,
) -> dict[str, list[StepResult]]:
    """批量模式：所有 Step 对 sentinel subject _batch_ 执行一次。

    用于 Pre / Post 阶段（mode=batch）。
    """
    results: list[StepResult] = []
    result_base = base_env.get("PIPELINE_RESULT_DIR", str(output_dir))

    for step in steps:
        step_dir = Path(result_base) / "intermediates" / phase.name / step.stem
        env = {
            **base_env,
            "PIPELINE_PHASE": phase.name,
            "PIPELINE_SUBJECT": "_batch_",
            "PIPELINE_STEP_NAME": step.stem,
            "PIPELINE_INTERMEDIATES": str(Path(result_base) / "intermediates"),
            "PIPELINE_DUPLICATE_POLICY": phase.duplicate_policy,
        }

        result = _retry_step(
            step=step,
            step_dir=step_dir,
            env=env,
            prior_results=results,
            subject_name="_batch_",
            retry_cfg=phase.retry,
            executor=executor,
        )
        results.append(result)

        # Progress — first batch phase is conventionally "pre", others "post"
        if pp:
            if phase.name == "pre":
                pp.pre_step_done()
            else:
                pp.post_step_done()

        if result.status == "error" and phase.retry.on_failure == "abort":
            logger.error("Aborting batch phase '%s' due to %s failure", phase.name, step.stem)
            break

    return {"_batch_": results}


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
) -> dict[str, list[StepResult]]:
    """逐 Subject 模式：每个 Subject 顺序执行所有 Step。

    当 pool.workers > 1 且 subjects > 1 时启用 ThreadPoolExecutor 并发。
    """
    pool_cfg = phase.pool
    use_pool = pool_cfg is not None and pool_cfg.workers > 1 and len(subjects) > 1

    if use_pool:
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
            pp.review_subject_running(subject)
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
) -> list[StepResult]:
    """对单个 Subject 执行全部 Step（顺序）。"""
    subject_results: list[StepResult] = []
    result_base = base_env.get("PIPELINE_RESULT_DIR", str(output_dir))

    for step in steps:
        logger.info("  [%s] step '%s' starting", subject, step.stem)
        step_dir = Path(result_base) / "intermediates" / subject / step.stem
        env = {
            **base_env,
            "PIPELINE_PHASE": phase.name,
            "PIPELINE_SUBJECT": subject,
            "PIPELINE_STEP_NAME": step.stem,
            "PIPELINE_INTERMEDIATES": str(Path(result_base) / "intermediates"),
            "PIPELINE_DUPLICATE_POLICY": phase.duplicate_policy,
        }

        result = _retry_step(
            step=step,
            step_dir=step_dir,
            env=env,
            prior_results=subject_results,
            subject_name=subject,
            retry_cfg=phase.retry,
            executor=executor,
        )
        subject_results.append(result)

        if pp:
            pp.review_step_done(subject)

        if result.status == "error" and phase.retry.on_failure == "abort":
            logger.error("Aborting pipeline for %s due to %s failure", subject, step.stem)
            break

    return subject_results


# ============================================================================
# 池化执行
# ============================================================================


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
) -> dict[str, list[StepResult]]:
    """Worker 池并发处理多个 Subject。"""
    actual_workers = min(pool_cfg.workers, len(subjects))
    per_subject_timeout = pool_cfg.timeout if pool_cfg.timeout > 0 else None

    logger.info(
        "Pool mode: %d worker(s) processing %d subject(s)",
        actual_workers,
        len(subjects),
    )

    all_results: dict[str, list[StepResult]] = {}
    errors: list[str] = []
    start_times: dict[str, float] = {}
    timed_out_futures: set[Future] = set()

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
            )
            future_map[fut] = s
            start_times[s] = time.monotonic()
            if pool_progress:
                pool_progress.on_subject_start(s)
            if pp:
                pp.review_subject_running(s)

        pending = set(future_map.keys())
        while pending:
            done, pending = wait(pending, timeout=1.0)

            for future in done:
                subject = future_map[future]
                try:
                    results = future.result()
                    all_results[subject] = results
                    elapsed = time.monotonic() - start_times[subject]
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
                if (
                    per_subject_timeout is not None
                    and (now - start_times[subject]) > per_subject_timeout
                ):
                    pending.discard(future)
                    timed_out_futures.add(future)
                    future.cancel()
                    errors.append(subject)
                    logger.error("  [%s] ✗ timed out after %ds", subject, per_subject_timeout)
                    if pool_progress:
                        pool_progress.on_subject_fail(
                            subject, "timeout", f"Timed out after {per_subject_timeout}s"
                        )
                    all_results[subject] = _make_error_results(
                        steps, subject, f"Timed out after {per_subject_timeout}s"
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
    if timed_out_futures:
        done_timed_out, _ = wait(timed_out_futures)
        for future in done_timed_out:
            subject = future_map[future]
            try:
                results = future.result()
                all_results[subject] = results
                elapsed = time.monotonic() - start_times[subject]
                logger.info("  [%s] ✓ completed after timeout in %.0fs", subject, elapsed)
            except (CancelledError, Exception) as e:
                logger.debug("  [%s] timed-out future resolved with %s", subject, type(e).__name__)

    if pool_cfg.ordered:
        all_results = {s: all_results[s] for s in subjects if s in all_results}

    if errors:
        logger.warning("Pool mode finished with %d failed subject(s): %s", len(errors), errors)

    return all_results


# ============================================================================
# 报告生成
# ============================================================================


def _generate_report(
    report_path: Path,
    task_id: str,
    pipeline_name: str,
    all_phase_results: dict,
    all_step_results: list[StepResult],
    success: bool,
) -> str:
    """生成最终报告 markdown 文件，返回 CLI 可输出的结论摘要。"""
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 论文评审报告",
        "",
        f"- **Task ID**: `{task_id}`",
        f"- **Pipeline**: {pipeline_name}",
        f"- **时间**: {now}",
        f"- **状态**: {'✅ 通过' if success else '❌ 有错误'}",
        "",
    ]

    conclusion_parts: list[str] = []

    for phase_name, phase_results in all_phase_results.items():
        lines.append(f"## {phase_name.upper()} 阶段")
        lines.append("")
        for subject_name, subj_results in phase_results.items():
            if subject_name != "_batch_":
                lines.append(f"### {subject_name}")
                lines.append("")
            for sr in subj_results:
                status_icon = "✅" if sr.status == "ok" else "⚠️" if sr.status == "skipped" else "❌"
                lines.append(f"- {status_icon} **{sr.step_name}**: {sr.status}")
                if sr.error:
                    lines.append(f"  - 错误: {sr.error}")
                raw = sr.data.get("raw_output", "")
                if raw and isinstance(raw, str) and len(raw) > 10:
                    lines.append("")
                    lines.append(raw)
                    lines.append("")
                    conclusion_parts.append(raw.strip())
                elif sr.data and any(v for v in sr.data.values() if v):
                    lines.append(f"  - 数据: {json.dumps(sr.data, ensure_ascii=False, indent=4)}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"> 完整中间产物见: `{report_path.parent / 'intermediates'}`")

    report_dir = report_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    conclusion = "\n".join(conclusion_parts).strip()
    if len(conclusion) > 500:
        conclusion = conclusion[:500] + "..."
    return conclusion


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

    # ── 生效的阶段 ──
    active_phases = (
        [p for p in config.phases if p.name == target_phase]
        if target_phase
        else [p for p in config.phases if p.directory]
    )

    all_phase_results: dict[str, dict[str, list[StepResult]]] = {}
    all_step_results: list[StepResult] = []
    overall_success = True

    if not active_phases:
        logger.warning("No active phases — nothing to run")
        conclusion = _generate_report(
            task_dir / "report.md",
            task_id,
            config.name,
            all_phase_results,
            all_step_results,
            overall_success,
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
    primary_subject = subjects[0] if subjects else ""

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
    pre_steps_count = 0
    review_subjects = len(subjects) if subjects else 0
    review_steps_per = 0
    post_steps_count = 0
    for phase in active_phases:
        step_count = len(discover_steps(pipeline_dir / phase.directory))
        if phase.mode == "batch":
            if not pre_steps_count:
                pre_steps_count = step_count
            else:
                post_steps_count = step_count
        elif phase.mode == "per_subject":
            review_steps_per = step_count

    pp = PipelineProgress(
        pre_steps=pre_steps_count,
        review_subjects=review_subjects,
        review_steps_per_subject=review_steps_per,
        post_steps=post_steps_count,
    )
    pp.start()

    # ── 阶段遍历 ──
    for phase in active_phases:
        phase_dir = pipeline_dir / phase.directory
        steps = discover_steps(phase_dir)
        if target_step and target_phase and phase.name == target_phase:
            steps = [s for s in steps if s.stem == target_step]
        if not steps:
            continue

        if phase.mode == "batch":
            logger.info("Phase [%s] batch — %d step(s)", phase.name, len(steps))
            phase_results = _execute_batch(
                phase=phase,
                steps=steps,
                output_dir=config.output_dir,
                base_env=base_env,
                executor=_make_executor(py_runner, md_executor),
                pp=pp,
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
                        logger.info(
                            "Manifest step '%s' completed successfully", phase.manifest_step
                        )
                else:
                    logger.warning(
                        "manifest_step '%s' not found in phase '%s'. Available: %s",
                        phase.manifest_step,
                        phase.name,
                        [s.stem for s in steps],
                    )

        elif phase.mode == "per_subject":
            logger.info(
                "Phase [%s] per_subject — %d step(s), %d subject(s)",
                phase.name,
                len(steps),
                len(subjects),
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
            )
        else:
            logger.warning("Unknown mode '%s' for phase '%s' — skipping", phase.mode, phase.name)
            continue

        all_phase_results[phase.name] = phase_results
        for subj_results in phase_results.values():
            all_step_results.extend(subj_results)
            for r in subj_results:
                if r.status == "error":
                    overall_success = False

    # ── 完成 ──
    pp.finish()

    conclusion = _generate_report(
        task_dir / "report.md",
        task_id,
        config.name,
        all_phase_results,
        all_step_results,
        overall_success,
    )

    task_meta = {
        "task_id": task_id,
        "pipeline": config.name,
        "input": str(input_path.absolute()),
        "subjects": subjects,
        "success": overall_success,
        "step_count": len(all_step_results),
        "error_count": sum(1 for r in all_step_results if r.status == "error"),
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps(task_meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return PipelineResult(
        subject=primary_subject,
        success=overall_success,
        step_results=all_step_results,
        task_id=task_id,
        task_dir=task_dir,
        conclusion=conclusion,
    )


def _make_executor(py_runner: PyStepRunner, md_executor: MdStepExecutor) -> StepExecutor:
    """创建 StepExecutor adapter。

    将 PyStepRunner + MdStepExecutor 包装为 StepExecutor 协议，
    通过 _execute_step 薄分派实现。供 _retry_step 调用。
    """

    class _Adapter:
        def execute(self, step, step_dir, env, prior_results, subject_name):
            return _execute_step(
                step, step_dir, env, prior_results, subject_name, py_runner, md_executor
            )

    return _Adapter()
