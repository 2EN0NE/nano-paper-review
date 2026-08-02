"""
Orchestrator —— 评审流水线执行引擎

Phase 执行（顺序/池化）、报告生成、公共入口 run_pipeline()。

数据模型 → pipeline_models.py
Step 执行 → pipeline_steps.py
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, wait
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
from paper_review.progress import PipelineProgress

logger = get_logger("orchestrator")

# 解析时模板变量 pattern：{{ variable }}
_PARSE_TIME_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


# ============================================================================
# Manifest & Subject 发现
# ============================================================================


def _load_manifest(manifest_path: Path) -> dict | None:
    """加载 subject-manifest.json，返回解析后的 dict。

    文件不存在时返回 None（调用方可回退到 CLI 发现）；
    文件存在但不可解析时记录 error 后仍返回 None，
    调用方应据此判断 Pre 阶段可能未正确完成。
    """
    if not manifest_path.exists():
        logger.info("Manifest not found at %s — will fall back to CLI discovery", manifest_path)
        return None
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        logger.info("Loaded manifest: %d subjects", len(data.get("subjects", [])))
        return data
    except json.JSONDecodeError as e:
        logger.error(
            "Manifest file exists at %s but JSON is corrupt: %s. "
            "Pre phase may have failed to produce a valid manifest.",
            manifest_path,
            e,
        )
        return None
    except OSError as e:
        logger.error("Failed to read manifest at %s: %s", manifest_path, e)
        return None


def _resolve_parse_time_vars(value: str, var_map: dict[str, str]) -> str:
    """替换字符串中的 {{ var_name }} 模板变量。对未匹配变量发出 warning。"""

    def _replacer(m: re.Match) -> str:
        var_name = m.group(1)
        if var_name in var_map:
            return var_map[var_name]
        logger.warning(
            "Unknown template variable '{{ %s }}' in '%s' — left as-is",
            var_name,
            value,
        )
        return m.group(0)

    return _PARSE_TIME_VAR.sub(_replacer, value)


def _resolve_config_vars(value: str, output_dir: Path) -> str:
    """解析配置中的 {{ output_dir }} 等变量。"""
    var_map = {
        "output_dir": str(output_dir.absolute()),
    }
    return _resolve_parse_time_vars(value, var_map)


def _discover_subjects(
    input_path: Path,
) -> list[str]:
    """从 CLI 输入路径发现 Subject 列表（纯 CLI 扫描）。

    manifest 模式由 run_pipeline 内联处理，此函数仅做 CLI 回退扫描。
    """
    if input_path.is_dir():
        return sorted(f.stem for f in input_path.iterdir() if f.is_file() and f.suffix == ".pdf")
    else:
        return [input_path.stem]


def _apply_duplicate_policy(
    subjects: list[str],
    policy: str,
) -> list[str]:
    """按 duplicate_policy 处理同名 Subject。

    policy:
      skip   — 以先出现的为准，后出现的跳过（默认）
      rename — 自动在重复项后加 -1, -2 等后缀
      error  — 抛出 ValueError
    """
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

    # skip（默认）: 保留先出现的
    if policy != "skip":
        logger.warning(
            "Unknown duplicate_policy '%s' — falling back to 'skip'. Valid values: skip, rename, error",
            policy,
        )

    result: list[str] = []
    seen: set[str] = set()
    skipped_count = 0
    for s in subjects:
        if s in seen:
            skipped_count += 1
            logger.info("Duplicate subject '%s' skipped (duplicate_policy=skip)", s)
            continue
        seen.add(s)
        result.append(s)
    if skipped_count:
        logger.info("Skipped %d duplicate subject(s)", skipped_count)
    return result


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
    pp: PipelineProgress | None = None,
) -> tuple[str, list[StepResult]]:
    """处理单个 Subject 的所有 Step（供顺序/池化模式共用）。

    每个 Subject 顺序执行全部 Steps，共享 prior_results 链。

    Returns:
        (subject_name, [StepResult, ...])
    """
    if progress:
        progress.on_subject_start(subject)
    if pp:
        pp.review_subject_running(subject)

    subject_results: list[StepResult] = []

    for step in steps:
        logger.info(
            "  [%s] step '%s' starting",
            subject,
            step.stem,
        )
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
            "PIPELINE_DUPLICATE_POLICY": phase_config.duplicate_policy,
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

        if pp:
            pp.review_step_done(subject)

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
    pp: PipelineProgress | None = None,
) -> dict[str, list[StepResult]]:
    """使用 Worker 池并发处理多个 Subject。

    每个 Worker 负责一个 Subject 的全部 Steps（顺序执行）。
    超时按单个 Subject 计时（非整个池），每秒轮询进度。
    """
    pool_cfg = phase_config.pool
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
    timed_out_futures: set[Future] = set()  # 超时但因 running 无法 cancel 的 future

    executor = ThreadPoolExecutor(max_workers=actual_workers)
    try:
        future_map: dict[Future, str] = {}
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
                pp,
            )
            future_map[fut] = s
            start_times[s] = time.monotonic()

        pending = set(future_map.keys())
        while pending:
            done, pending = wait(pending, timeout=1.0)  # 每秒 poll

            for future in done:
                subject = future_map[future]
                try:
                    _, results = future.result()
                    all_results[subject] = results
                    elapsed = time.monotonic() - start_times[subject]
                    logger.info(
                        "  [%s] ✓ completed in %.0fs",
                        subject,
                        elapsed,
                    )
                except Exception as e:
                    logger.error("  [%s] ✗ failed: %s", subject, e)
                    errors.append(subject)
                    if progress:
                        progress.on_subject_fail(subject, "error", str(e))
                    all_results[subject] = _make_error_results(steps, subject, str(e))

            # 检查每个 pending subject 是否超时
            now = time.monotonic()
            timed_out: list[Future] = []
            for future in list(pending):
                subject = future_map[future]
                if (
                    per_subject_timeout is not None
                    and (now - start_times[subject]) > per_subject_timeout
                ):
                    timed_out.append(future)

            for future in timed_out:
                pending.discard(future)
                timed_out_futures.add(future)
                subject = future_map[future]
                future.cancel()
                errors.append(subject)
                logger.error(
                    "  [%s] ✗ timed out after %ds",
                    subject,
                    per_subject_timeout,
                )
                if progress:
                    progress.on_subject_fail(
                        subject,
                        "timeout",
                        f"Timed out after {per_subject_timeout}s",
                    )
                all_results[subject] = _make_error_results(
                    steps,
                    subject,
                    f"Timed out after {per_subject_timeout}s",
                )

            # 实时进度 — PipelineProgress 由 _process_single_subject 内的回调更新，
            # spinner 线程负责刷新。这里只需回退到 logger 报告（无 pp 时）。
            if not pp:
                done_count = len(all_results)
                running_count = len(pending)
                logger.info(
                    "  Progress: %d/%d done, %d running",
                    done_count,
                    len(subjects),
                    running_count,
                )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # ── 等待超时但 worker 线程仍在运行的 future 实质完成 ──
    # Python ThreadPoolExecutor 无法 cancel 正在运行的线程（cancel() 返回 False），
    # 因此超时 Subject 的 worker 可能仍在跑。必须在 return 前等它们结束，
    # 否则 Post 阶段会在 Review 未完成时拿到不完整的 intermediates。
    if timed_out_futures:
        done_timed_out, _ = wait(timed_out_futures)
        for future in done_timed_out:
            subject = future_map[future]
            try:
                _, results = future.result()
                all_results[subject] = results
                elapsed = time.monotonic() - start_times[subject]
                logger.info(
                    "  [%s] ✓ completed after timeout in %.0fs",
                    subject,
                    elapsed,
                )
            except (CancelledError, Exception) as e:
                # worker 已被 cancel 或已将异常记入 all_results
                logger.debug("  [%s] timed-out future resolved with %s", subject, type(e).__name__)

    if pool_cfg.ordered:
        all_results = {s: all_results[s] for s in subjects if s in all_results}

    if errors:
        logger.warning("Pool mode finished with %d failed subject(s): %s", len(errors), errors)

    return all_results


def _make_error_results(
    steps: list[StepFile],
    subject: str,
    error_msg: str,
) -> list[StepResult]:
    """为超时/失败的 subject 生成统一的 error StepResult 列表。"""
    return [
        StepResult(
            step_name=step.stem,
            status="error",
            error=error_msg,
            subject=subject,
        )
        for step in steps
    ]


def _run_phase_steps(
    steps: list[StepFile],
    phase_name: str,
    subjects: list[str],
    phase_config: PhaseConfig,
    output_dir: Path,
    base_env: dict,
    progress: PoolProgress | None = None,
    pp: PipelineProgress | None = None,
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
            pp=pp,
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
                "PIPELINE_DUPLICATE_POLICY": phase_config.duplicate_policy,
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

            # Progress display update
            if pp:
                if phase_name == "pre":
                    pp.pre_step_done()
                elif phase_name == "post":
                    pp.post_step_done()
                elif phase_name == "review":
                    pp.review_step_done(subject)

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

    all_phase_results: dict[str, dict[str, list[StepResult]]] = {}
    all_step_results: list[StepResult] = []
    overall_success = True
    subjects: list[str] = []
    primary_subject = ""

    # ── 确定需要跑哪些阶段 ──
    has_pre = bool(config.pre.directory)
    has_review = bool(config.review.directory)
    has_post = bool(config.post.directory)

    if target_phase:
        has_pre = target_phase == "pre"
        has_review = target_phase == "review"
        has_post = target_phase == "post"

    # ── Pre Phase（在 Subject 发现之前运行）──
    pre_steps_count = 0
    if has_pre:
        pre_dir = pipeline_dir / config.pre.directory
        pre_steps = discover_steps(pre_dir)
        if target_step and target_phase == "pre":
            pre_steps = [s for s in pre_steps if s.stem == target_step]
        if pre_steps:
            logger.info("Phase [pre] — %d step(s)", len(pre_steps))
            pre_steps_count = len(pre_steps)
            pre_results = _run_phase_steps(
                steps=pre_steps,
                phase_name="pre",
                subjects=["_batch_"],
                phase_config=config.pre,
                output_dir=config.output_dir,
                base_env=base_env,
            )
            all_phase_results["pre"] = pre_results
            for subj, subj_results in pre_results.items():
                all_step_results.extend(subj_results)
                for r in subj_results:
                    if r.status == "error":
                        overall_success = False

            # 验证 manifest_step 是否实际运行
            declared_step = config.pre.manifest_step
            if declared_step:
                pre_batch_results = pre_results.get("_batch_", [])
                manifest_step_ran = any(r.step_name == declared_step for r in pre_batch_results)
                if manifest_step_ran:
                    manifest_step_result = next(
                        r for r in pre_batch_results if r.step_name == declared_step
                    )
                    if manifest_step_result.status == "error":
                        logger.error(
                            "Declared manifest_step '%s' failed in Pre phase — "
                            "manifest may be missing or incomplete",
                            declared_step,
                        )
                    else:
                        logger.info(
                            "Manifest step '%s' completed successfully",
                            declared_step,
                        )
                else:
                    logger.warning(
                        "Declared manifest_step '%s' was not found in Pre phase steps. "
                        "Available steps: %s",
                        declared_step,
                        [s.stem for s in pre_steps],
                    )

    # ── Subject 发现（Pre 阶段可能已产生 manifest）──
    use_manifest = config.review.subject_source.type == "manifest"
    if use_manifest:
        manifest_path_str = _resolve_config_vars(
            config.review.subject_source.path, config.output_dir
        )
        manifest_data = _load_manifest(Path(manifest_path_str))
        if manifest_data:
            subjects_data = manifest_data.get("subjects", [])
            if subjects_data:
                raw_subjects = [s["name"] for s in subjects_data]
            else:
                logger.warning("Manifest contains 0 subjects — falling back to CLI discovery")
                raw_subjects = _discover_subjects(input_path)
        else:
            # manifest 不存在或损坏 → CLI 扫描（_load_manifest 已记录具体原因）
            logger.info(
                "Falling back to CLI subject discovery from %s",
                input_path,
            )
            raw_subjects = _discover_subjects(input_path)
    else:
        raw_subjects = _discover_subjects(input_path)

    raw_subjects = _apply_duplicate_policy(raw_subjects, config.review.duplicate_policy)

    if not raw_subjects:
        logger.warning("No subjects found for review — skipping Review/Post phases")
        # 如果已经跑过 Pre，直接返回结果
        report_path = task_dir / "report.md"
        conclusion = _generate_report(
            report_path,
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

    subjects = _order_subjects(raw_subjects, config.review.subject_order)
    primary_subject = subjects[0]

    # 从全局配置（config.yaml / env var）覆盖 pool 默认值
    if pool_progress is None:
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

    logger.info("Pipeline '%s' — %d subject(s)", config.name, len(subjects))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Progress display ──
    # Discover step counts for review/post phases
    _review_steps_for_count = (
        discover_steps(pipeline_dir / config.review.directory) if has_review else []
    )
    _post_steps_for_count = discover_steps(pipeline_dir / config.post.directory) if has_post else []
    pp = PipelineProgress(
        pre_steps=pre_steps_count,
        review_subjects=len(subjects),
        review_steps_per_subject=len(_review_steps_for_count),
        post_steps=len(_post_steps_for_count),
    )
    # Pre phase already completed
    for _ in range(pre_steps_count):
        pp.pre_step_done()
    pp.start()

    # ── Review Phase ──
    if has_review:
        review_dir = pipeline_dir / config.review.directory
        review_steps = discover_steps(review_dir)
        if target_step and target_phase == "review":
            review_steps = [s for s in review_steps if s.stem == target_step]
        if review_steps:
            logger.info(
                "Phase [review] — %d step(s), %d subject(s)",
                len(review_steps),
                len(subjects),
            )
            review_results = _run_phase_steps(
                steps=review_steps,
                phase_name="review",
                subjects=subjects,
                phase_config=config.review,
                output_dir=config.output_dir,
                base_env=base_env,
                progress=pool_progress,
                pp=pp,
            )
            all_phase_results["review"] = review_results
            for subj, subj_results in review_results.items():
                all_step_results.extend(subj_results)
                for r in subj_results:
                    if r.status == "error":
                        overall_success = False

    # ── Post Phase ──
    if has_post:
        post_dir = pipeline_dir / config.post.directory
        post_steps = discover_steps(post_dir)
        if target_step and target_phase == "post":
            post_steps = [s for s in post_steps if s.stem == target_step]
        if post_steps:
            logger.info("Phase [post] — %d step(s)", len(post_steps))
            post_results = _run_phase_steps(
                steps=post_steps,
                phase_name="post",
                subjects=["_batch_"],
                phase_config=config.post,
                output_dir=config.output_dir,
                base_env=base_env,
                pp=pp,
            )
            all_phase_results["post"] = post_results
            for subj, subj_results in post_results.items():
                all_step_results.extend(subj_results)
                for r in subj_results:
                    if r.status == "error":
                        overall_success = False

    # 完成进度条
    pp.finish()

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
