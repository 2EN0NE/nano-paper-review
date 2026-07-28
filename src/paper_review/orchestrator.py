"""
Orchestrator —— 评审流水线执行引擎

Phase 执行（顺序/池化）、报告生成、公共入口 run_pipeline()。

数据模型 → pipeline_models.py
Step 执行 → pipeline_steps.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from paper_review.logging_config import get_logger

# 从拆分后的子模块导入核心类型与逻辑
from paper_review.pipeline_models import (  # noqa: F401 — 向后兼容 re-export
    PhaseConfig,
    PipelineConfig,
    PipelineResult,
    PoolConfig,
    PoolProgress,
    PoolProgressEvent,
    RetryConfig,
    ReviewPhaseConfig,
    StepFile,
    StepResult,
    SubjectOrderConfig,
    SubjectOrderPriority,
    _load_yaml,
    _order_subjects,
    discover_steps,
    validate_step_output,
)
from paper_review.pipeline_steps import _run_step  # Step 分派

logger = get_logger("orchestrator")


# ============================================================================
# 阶段执行
# ============================================================================


def _process_single_subject(
    subject: str,
    steps: list[StepFile],
    phase_name: str,
    phase_config: PhaseConfig,
    output_dir: Path,
    base_env: dict,
    progress: PoolProgress | None = None,
) -> tuple[str, list[StepResult]]:
    """处理单个 Subject 的所有 Step（供顺序/池化模式共用）。

    每个 Subject 顺序执行全部 Steps，共享 prior_results 链。

    Returns:
        (subject_name, [StepResult, ...])
    """
    if progress:
        progress.on_subject_start(subject)

    subject_results: list[StepResult] = []

    for step in steps:
        # 构建 intermediates 路径（优先使用任务结果目录）
        result_base = base_env.get("PIPELINE_RESULT_DIR", str(output_dir))
        if phase_name == "review":
            step_dir = Path(result_base) / "intermediates" / subject / step.stem
        else:
            step_dir = Path(result_base) / "intermediates" / phase_name / step.stem

        env = {
            **base_env,
            "PIPELINE_PHASE": phase_name,
            "PIPELINE_SUBJECT": subject,
            "PIPELINE_STEP_NAME": step.stem,
            "PIPELINE_INTERMEDIATES": str(Path(result_base) / "intermediates"),
        }

        # 重试循环
        result: StepResult | None = None
        for attempt in range(1, phase_config.retry.max_attempts + 1):
            logger.debug(
                "  [%s] step %s attempt %d/%d",
                subject,
                step.stem,
                attempt,
                phase_config.retry.max_attempts,
            )

            try:
                result = _run_step(
                    step,
                    step_dir,
                    env,
                    prior_results=subject_results,
                    subject_name=subject,
                )
                result.subject = subject
                result.attempt = attempt

                if result.status in ("ok", "skipped"):
                    break  # 不需要重试

                logger.warning(
                    "  [%s] step %s attempt %d failed: %s",
                    subject,
                    step.stem,
                    attempt,
                    result.error,
                )

            except Exception as e:
                logger.error(
                    "  [%s] step %s attempt %d raised exception: %s",
                    subject,
                    step.stem,
                    attempt,
                    e,
                )
                result = StepResult(
                    step_name=step.stem,
                    status="error",
                    error=str(e),
                    subject=subject,
                    attempt=attempt,
                )

        if result is None:
            result = StepResult(
                step_name=step.stem,
                status="error",
                error="All attempts exhausted (no result)",
                subject=subject,
            )

        subject_results.append(result)

        if result.status == "error" and phase_config.retry.on_failure == "abort":
            logger.error("Aborting pipeline for %s due to %s failure", subject, step.stem)
            break

    if progress:
        progress.on_subject_complete(subject, subject_results)

    return subject, subject_results


def _run_subjects_pooled(
    steps: list[StepFile],
    subjects: list[str],
    phase_config: ReviewPhaseConfig,
    output_dir: Path,
    base_env: dict,
    progress: PoolProgress | None = None,
) -> dict[str, list[StepResult]]:
    """使用 Worker 池并发处理多个 Subject。

    每个 Worker 负责一个 Subject 的全部 Steps（顺序执行）。
    支持超时取消和进度回调。
    """
    pool_cfg = phase_config.pool
    actual_workers = min(pool_cfg.workers, len(subjects))

    logger.info(
        "Pool mode: %d worker(s) processing %d subject(s)",
        actual_workers,
        len(subjects),
    )

    all_results: dict[str, list[StepResult]] = {}
    errors: list[str] = []

    # 手动管理 executor——超时时用 shutdown(wait=False) 避免阻塞
    executor = ThreadPoolExecutor(max_workers=actual_workers)
    try:
        future_map = {}
        for s in subjects:
            fut = executor.submit(
                _process_single_subject,
                s,
                steps,
                "review",
                phase_config,
                output_dir,
                base_env,
                progress,
            )
            future_map[fut] = s

        # 使用 wait() 轮询——避免 as_completed 在超时后仍等待 running future
        pending = set(future_map.keys())
        while pending:
            poll_timeout = pool_cfg.timeout if pool_cfg.timeout > 0 else None
            done, pending = wait(pending, timeout=poll_timeout)

            for future in done:
                subject = future_map[future]
                try:
                    _, results = future.result()
                    all_results[subject] = results
                    logger.debug("  Subject '%s' completed (%d steps)", subject, len(results))
                except Exception as e:
                    logger.error("  Subject '%s' failed: %s", subject, e)
                    errors.append(subject)
                    if progress:
                        progress.on_subject_fail(subject, "error", str(e))
                    error_results: list[StepResult] = []
                    for step in steps:
                        error_results.append(
                            StepResult(
                                step_name=step.stem,
                                status="error",
                                error=f"Pool worker failed: {e}",
                                subject=subject,
                            )
                        )
                    all_results[subject] = error_results

            # 超时后有 pending future → 标记为超时失败
            if pending:
                logger.error("  %d subject(s) timed out after %ds", len(pending), pool_cfg.timeout)
                for future in pending:
                    subject = future_map[future]
                    errors.append(subject)
                    future.cancel()
                    if progress:
                        progress.on_subject_fail(
                            subject, "timeout", f"Timed out after {pool_cfg.timeout}s"
                        )
                    error_results = []
                    for step in steps:
                        error_results.append(
                            StepResult(
                                step_name=step.stem,
                                status="error",
                                error=f"Timed out after {pool_cfg.timeout}s",
                                subject=subject,
                            )
                        )
                    all_results[subject] = error_results
                break  # 不再继续轮询
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # 按原始顺序返回（pool_cfg.ordered）
    if pool_cfg.ordered:
        all_results = {s: all_results[s] for s in subjects if s in all_results}

    if errors:
        logger.warning("Pool mode finished with %d failed subject(s): %s", len(errors), errors)

    return all_results


def _run_phase_steps(
    steps: list[StepFile],
    phase_name: str,
    subjects: list[str],
    phase_config: PhaseConfig,
    output_dir: Path,
    base_env: dict,
    progress: PoolProgress | None = None,
) -> dict[str, list[StepResult]]:
    """运行阶段内的所有步骤。

    Review Phase 支持 Worker 池并发（当 subjects > 1 且 pool.workers > 1 时），
    Pre/Post Phase 保持顺序逐 Step 执行。

    Args:
        steps: 本阶段的 Step 列表。
        phase_name: 'pre' | 'review' | 'post'。
        subjects: Subject 名称列表（review 阶段逐篇，pre/post 批量）。
        phase_config: 阶段配置（含 retry + pool）。
        output_dir: 输出根目录。
        base_env: 基础环境变量。
        progress: 可选的 PoolProgress 回调（仅 review phase 有效）。

    Returns:
        {subject_name: [StepResult, ...]}
    """
    # Review Phase 且多 Subject 时启用池化
    use_pool = (
        phase_name == "review"
        and isinstance(phase_config, ReviewPhaseConfig)
        and phase_config.pool.workers > 1
        and len(subjects) > 1
    )

    if use_pool:
        # isinstance guard above guarantees phase_config is ReviewPhaseConfig here
        review_cfg: ReviewPhaseConfig = phase_config  # type: ignore[assignment]
        return _run_subjects_pooled(
            steps=steps,
            subjects=subjects,
            phase_config=review_cfg,
            output_dir=output_dir,
            base_env=base_env,
            progress=progress,
        )

    # 顺序执行模式（Pre/Post 单 Subject，或 Review 单 Subject/workers=1）
    all_results: dict[str, list[StepResult]] = {s: [] for s in subjects}
    result_base = base_env.get("PIPELINE_RESULT_DIR", str(output_dir))

    for step in steps:
        for subject in subjects:
            step_dir = (
                Path(result_base) / "intermediates" / subject / step.stem
                if phase_name == "review"
                else Path(result_base) / "intermediates" / phase_name / step.stem
            )
            env = {
                **base_env,
                "PIPELINE_PHASE": phase_name,
                "PIPELINE_SUBJECT": subject,
                "PIPELINE_STEP_NAME": step.stem,
                "PIPELINE_INTERMEDIATES": str(Path(result_base) / "intermediates"),
            }

            result: StepResult | None = None
            for attempt in range(1, phase_config.retry.max_attempts + 1):
                try:
                    result = _run_step(
                        step,
                        step_dir,
                        env,
                        prior_results=all_results[subject],
                        subject_name=subject,
                    )
                    result.subject = subject
                    result.attempt = attempt
                    if result.status in ("ok", "skipped"):
                        break
                    logger.warning("  [%s] attempt %d failed: %s", subject, attempt, result.error)
                except Exception as e:
                    logger.error("  [%s] attempt %d raised exception: %s", subject, attempt, e)
                    result = StepResult(
                        step_name=step.stem,
                        status="error",
                        error=str(e),
                        subject=subject,
                        attempt=attempt,
                    )

            if result is None:
                result = StepResult(
                    step_name=step.stem,
                    status="error",
                    error="All attempts exhausted (no result)",
                    subject=subject,
                )

            all_results[subject].append(result)

            if result.status == "error" and phase_config.retry.on_failure == "abort":
                logger.error("Aborting pipeline due to %s failure on %s", step.stem, subject)
                return all_results

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
                # 提取 data.raw_output 作为结论
                raw = sr.data.get("raw_output", "")
                if raw and isinstance(raw, str) and len(raw) > 10:
                    lines.append("")
                    lines.append(raw)
                    lines.append("")
                    conclusion_parts.append(raw.strip())
                elif sr.data and any(v for v in sr.data.values() if v):
                    lines.append(f"  - 数据: {json.dumps(sr.data, ensure_ascii=False, indent=4)}")
            lines.append("")

    # 底部路径提示
    lines.append("---")
    lines.append("")
    lines.append(f"> 完整中间产物见: `{report_path.parent / 'intermediates'}`")

    report_dir = report_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # 结论摘要：取最后的非空内容，截断到 500 字
    conclusion = "\n".join(conclusion_parts).strip()
    if len(conclusion) > 500:
        conclusion = conclusion[:500] + "..."
    return conclusion


# ============================================================================
# 主入口
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
    """执行一条完整的 pipeline（三段式：Pre → Review → Post）。

    Args:
        pipeline_yaml: pipeline 配置字典、YAML 文件路径或包含 pipeline.yaml 的目录。
        input_path: 输入 PDF 路径（单篇）或目录（多篇）。
        pipeline_dir: pipeline 定义目录的根。为 None 时从 pipeline_yaml 推断。
        output_dir: 覆盖配置中的 output_dir。
        data_dir: 数据目录（用于默认 output_dir 解析）。
        target_phase: 仅运行指定阶段（'pre' / 'review' / 'post'）。
        target_step: 仅运行指定步骤名（需已有中间产物）。
        pool_progress: 可选的 PoolProgress 实例，接收 Worker 池进度事件。

    Returns:
        PipelineResult
    """
    # 解析配置
    if isinstance(pipeline_yaml, (str, Path)):
        yaml_path = Path(pipeline_yaml)
        if yaml_path.is_dir():
            yaml_file = yaml_path / "pipeline.yaml"
            if yaml_file.exists():
                raw = _load_yaml(yaml_file)
                if pipeline_dir is None:
                    pipeline_dir = yaml_path
            else:
                raw = {
                    "name": "default",
                    "output_dir": "./output",
                    "review": {"directory": str(yaml_path)},
                }
        elif yaml_path.suffix in (".yaml", ".yml"):
            raw = _load_yaml(yaml_path)
            if pipeline_dir is None:
                pipeline_dir = yaml_path.parent
        else:
            raw = {
                "name": "default",
                "output_dir": "./output",
                "review": {"directory": str(yaml_path)},
            }
    else:
        raw = pipeline_yaml

    config = PipelineConfig.from_dict(raw)
    if output_dir:
        config.output_dir = output_dir
    elif data_dir:
        # 从 data_dir 推导默认 output_dir（仅程序化调用走此分支；CLI 永远传 output_dir=...）
        from paper_review.config import resolve_data_dir

        dd = resolve_data_dir(data_dir)
        config.output_dir = dd / "output"
    if pipeline_dir is None:
        pipeline_dir = Path.cwd()

    # 确定 subject
    if input_path.is_dir():
        raw_subjects = sorted(
            f.stem for f in input_path.iterdir() if f.is_file() and f.suffix == ".pdf"
        )
    else:
        raw_subjects = [input_path.stem]

    if not raw_subjects:
        logger.warning("No PDF files found at %s", input_path)
        return PipelineResult(subject="", success=True)

    # 应用 Subject 排序
    subjects = _order_subjects(raw_subjects, config.review.subject_order)
    primary_subject = subjects[0]

    # 从全局配置（config.yaml / env var）覆盖 pool 默认值
    if pool_progress is None:
        # 从环境变量读取 pool 默认覆盖
        env_workers = os.environ.get("PAPER_REVIEW_POOL_WORKERS")
        env_timeout = os.environ.get("PAPER_REVIEW_POOL_TIMEOUT")
        if env_workers is not None:
            try:
                config.review.pool.workers = int(env_workers)
            except ValueError:
                pass
        if env_timeout is not None:
            try:
                config.review.pool.timeout = int(env_timeout)
            except ValueError:
                pass

    logger.info("Pipeline '%s' starting — %d subject(s)", config.name, len(subjects))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 生成任务指纹（在任何阶段执行前，确保 intermediates 路径可用）
    import datetime
    import hashlib

    task_seed = f"{input_path.absolute()}:{datetime.datetime.now().isoformat()}"
    task_id = hashlib.sha256(task_seed.encode()).hexdigest()[:8]
    task_dir = config.output_dir / "result" / task_id
    logger.info("Task ID: %s → %s", task_id, task_dir)

    # 解析 data_dir（从参数或环境变量）
    resolved_data_dir = data_dir
    if resolved_data_dir is None:
        resolved_data_dir = os.environ.get("PAPER_REVIEW_DATA_DIR")
    if resolved_data_dir is None:
        # fallback: 从 cwd 或 home 自动发现
        from paper_review.config import resolve_data_dir

        resolved_data_dir = str(resolve_data_dir())

    base_env = {
        **os.environ,
        "PIPELINE_OUTPUT_DIR": str(config.output_dir.absolute()),
        "PIPELINE_PIPELINE_DIR": str(pipeline_dir.absolute()),
        "PIPELINE_DATA_DIR": resolved_data_dir,
        "PIPELINE_RESULT_DIR": str(task_dir),
    }

    # 定义各阶段
    phase_defs: list[tuple[str, PhaseConfig, str]] = [
        ("pre", config.pre, config.pre.directory),
        ("review", config.review, config.review.directory),
        ("post", config.post, config.post.directory),
    ]

    if target_phase:
        phase_defs = [p for p in phase_defs if p[0] == target_phase]

    all_phase_results: dict[str, dict[str, list[StepResult]]] = {}
    all_step_results: list[StepResult] = []
    overall_success = True

    # 顺序执行各阶段
    for phase_name, phase_cfg, phase_dir_str in phase_defs:
        if not phase_dir_str:
            logger.debug("Phase '%s' has no directory, skipping", phase_name)
            continue

        phase_dir = pipeline_dir / phase_dir_str
        steps = discover_steps(phase_dir)

        if not steps:
            logger.debug("Phase '%s' has no steps, skipping", phase_name)
            continue

        # 过滤单步骤
        if target_step:
            steps = [s for s in steps if s.stem == target_step]
            if not steps:
                logger.warning("Step '%s' not found in phase '%s'", target_step, phase_name)
                continue

        # 确定 subjects（pre/post 为批量模式：一个虚拟 subject）
        if phase_name == "review":
            phase_subjects = subjects
        else:
            phase_subjects = ["_batch_"]

        logger.info(
            "Phase [%s] — %d step(s), %d subject(s)",
            phase_name,
            len(steps),
            len(phase_subjects),
        )

        phase_results = _run_phase_steps(
            steps=steps,
            phase_name=phase_name,
            subjects=phase_subjects,
            phase_config=phase_cfg,
            output_dir=config.output_dir,
            base_env=base_env,
            progress=pool_progress,
        )
        all_phase_results[phase_name] = phase_results

        # 汇总
        for subj, subj_results in phase_results.items():
            all_step_results.extend(subj_results)
            for r in subj_results:
                if r.status == "error":
                    overall_success = False

    # 生成 report.md
    report_path = task_dir / "report.md"
    conclusion = _generate_report(
        report_path,
        task_id,
        config.name,
        all_phase_results,
        all_step_results,
        overall_success,
    )

    # 写入元数据
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
