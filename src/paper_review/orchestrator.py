"""
Orchestrator —— 评审流水线执行引擎

薄编排层：配置解析 → Subject 发现 → 遍历 phases → 报告生成。
模式函数：_execute_batch（批量） / _execute_per_subject（逐 Subject，支持池化）。

数据模型 → pipeline_models.py
Step 执行 → pipeline_steps.py
Subject 发现 → subject_discovery.py
"""
from __future__ import annotations


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
from paper_review.timeout_estimator import estimate_step_timeout

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
    step_timeout: int = 0,
) -> StepResult:
    """执行一个 Step，按 RetryConfig 重试。

    唯一的重试实现点——_execute_batch 和 _execute_per_subject 都调用此函数。
    step_timeout 通过 env['PIPELINE_STEP_TIMEOUT'] 传递给 executor。
    """
    timed_env = {**env, "PIPELINE_STEP_TIMEOUT": str(step_timeout)}
    result: StepResult | None = None
    t0 = time.monotonic()

    for attempt in range(1, retry_cfg.max_attempts + 1):
        logger.info(
            "  [%s] ▶ step '%s' attempt %d/%d (timeout=%ds)",
            subject_name,
            step.stem,
            attempt,
            retry_cfg.max_attempts,
            step_timeout,
        )

        try:
            result = executor.execute(step, step_dir, timed_env, prior_results, subject_name)
            elapsed = time.monotonic() - t0
            result.subject = subject_name
            result.attempt = attempt

            if result.status in ("ok", "skipped"):
                logger.info(
                    "  [%s] ✓ step '%s' %s in %.1fs (attempt %d)",
                    subject_name,
                    step.stem,
                    result.status,
                    elapsed,
                    attempt,
                )
                return result

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
    step_timeout: int = 0,
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
            step_timeout=step_timeout,
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
    step_timeout: int = 0,
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
            step_timeout=step_timeout,
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
            step_timeout=step_timeout,
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
) -> list[StepResult]:
    """对单个 Subject 执行全部 Step（顺序）。"""
    subject_results: list[StepResult] = []
    result_base = base_env.get("PIPELINE_RESULT_DIR", str(output_dir))
    t0 = time.monotonic()

    logger.info(
        "  [%s] ▶ starting %d step(s) (timeout=%ds/step)", subject, len(steps), step_timeout
    )

    for step in steps:
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
            step_timeout=step_timeout,
        )
        subject_results.append(result)

        if pp:
            pp.review_step_done(subject)

        if result.status == "error" and phase.retry.on_failure == "abort":
            logger.error("Aborting pipeline for %s due to %s failure", subject, step.stem)
            break

    elapsed = time.monotonic() - t0
    logger.info("  [%s] ✓ all %d step(s) done (%.1fs total)", subject, len(steps), elapsed)
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
    step_timeout: int = 0,
) -> dict[str, list[StepResult]]:
    """Worker 池并发处理多个 Subject。"""
    actual_workers = min(pool_cfg.workers, len(subjects))
    per_subject_timeout = pool_cfg.timeout if pool_cfg.timeout > 0 else None

    logger.info(
        "Pool mode: %d worker(s) processing %d subject(s) (step_timeout=%ds)",
        actual_workers,
        len(subjects),
        step_timeout,
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
                step_timeout,
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
                # 恢复完成后同步 pool_progress 和 errors 列表
                if pool_progress:
                    pool_progress.on_subject_complete(subject, results)
                if subject in errors:
                    errors.remove(subject)
            except (CancelledError, Exception) as e:
                logger.debug("  [%s] timed-out future resolved with %s", subject, type(e).__name__)

    if pool_cfg.ordered:
        all_results = {s: all_results[s] for s in subjects if s in all_results}

    if errors:
        logger.warning("Pool mode finished with %d failed subject(s): %s", len(errors), errors)

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
        if phase.mode == "batch":
            batch = phase_results.get("_batch_", [])
            b_ok = sum(1 for r in batch if r.status == "ok")
            b_err = sum(1 for r in batch if r.status == "error")
            icon = "✅" if b_err == 0 else "❌"
            lines.append(f"{phase_prefix} {phase.name.upper()} (batch) {icon} {b_ok}/{len(batch)}")
        else:
            subjects_in_phase = [s for s in phase_results if s != "_batch_"]
            lines.append(
                f"{phase_prefix} {phase.name.upper()} (per_subject) "
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
    """生成最终报告 markdown 文件，返回 CLI 可输出的结构化结论摘要。"""
    import datetime

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
        "",
    ]

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
        lines.append(f"## {phase_name.upper()} 阶段")
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
                0, f"{phase_name.upper()}: {status_icon} {batch_ok}/{len(batch)} 步通过"
            )

    summary = f"共 {len(all_step_results)} 步（✅ {ok_count} / ❌ {error_count}）"
    if subject_count:
        summary += f"，{subject_count} 篇论文"
    conclusion_lines.insert(0, summary)
    conclusion_lines.append(f"\n完整报告: {report_path}")

    return "\n".join(conclusion_lines)


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
        _generate_report(
            task_dir / "report.md",
            task_id,
            config.name,
            all_phase_results,
            all_step_results,
            overall_success,
        )
        conclusion = _build_cli_tree(
            task_id, config.name, config, all_phase_results, pipeline_dir, task_dir
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
                # batch 模式处理全部 subject，per_subject 模式每步仅一个 subject
                estimated_chars = len(subjects) * 5000 if phase.mode == "batch" else 5000
                phase_timeout = estimate_step_timeout(
                    step_type="md",
                    total_chars=estimated_chars,
                    subject_count=len(subjects),
                )
            else:
                phase_timeout = estimate_step_timeout(step_type="py")

        if phase.mode == "batch":
            logger.info(
                "Phase [%s] batch — %d step(s) (timeout=%ds/step)",
                phase.name,
                len(steps),
                phase_timeout,
            )
            phase_results = _execute_batch(
                phase=phase,
                steps=steps,
                output_dir=config.output_dir,
                base_env=base_env,
                executor=_make_executor(py_runner, md_executor),
                pp=pp,
                step_timeout=phase_timeout,
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

    _generate_report(
        task_dir / "report.md",
        task_id,
        config.name,
        all_phase_results,
        all_step_results,
        overall_success,
    )
    conclusion = _build_cli_tree(
        task_id, config.name, config, all_phase_results, pipeline_dir, task_dir
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
