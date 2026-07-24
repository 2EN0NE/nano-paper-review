"""
Orchestrator —— 评审流水线执行引擎

处理 pipeline.yaml 解析、Step 发现/排序/顺序执行、中间产物管理。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from paper_rag.logging_config import get_logger

logger = get_logger("orchestrator")


# ============================================================================
# 数据模型
# ============================================================================


@dataclass
class RetryConfig:
    max_attempts: int = 1
    on_failure: str = "skip"  # 'skip' | 'abort'


@dataclass
class SubjectOrderPriority:
    first: list[str] = field(default_factory=list)
    last: list[str] = field(default_factory=list)


@dataclass
class SubjectOrderConfig:
    sort_by: str = "name"  # 'name' | 'regex'
    direction: str = "asc"  # 'asc' | 'desc'
    priority: SubjectOrderPriority | None = None

    def __post_init__(self):
        if isinstance(self.priority, dict):
            self.priority = SubjectOrderPriority(**self.priority)


@dataclass
class PhaseConfig:
    directory: str = ""
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class ReviewPhaseConfig(PhaseConfig):
    subject_order: SubjectOrderConfig = field(default_factory=SubjectOrderConfig)


@dataclass
class PipelineConfig:
    name: str = "unnamed"
    version: str = "1.0"
    output_dir: Path = Path("./output")
    pre: PhaseConfig = field(default_factory=PhaseConfig)
    review: ReviewPhaseConfig = field(default_factory=ReviewPhaseConfig)
    post: PhaseConfig = field(default_factory=PhaseConfig)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        name = data.get("name", "unnamed")
        version = data.get("version", "1.0")
        output_dir = Path(data.get("output_dir", "./output"))

        pre_data = data.get("pre", {})
        pre = PhaseConfig(
            directory=pre_data.get("directory", ""),
            retry=RetryConfig(
                max_attempts=pre_data.get("retry", {}).get("max_attempts", 1),
                on_failure=pre_data.get("retry", {}).get("on_failure", "skip"),
            ),
        )

        review_data = data.get("review", {"directory": ""})
        priority_data = review_data.get("subject_order", {}).get("priority")
        review = ReviewPhaseConfig(
            directory=review_data.get("directory", ""),
            retry=RetryConfig(
                max_attempts=review_data.get("retry", {}).get("max_attempts", 1),
                on_failure=review_data.get("retry", {}).get("on_failure", "skip"),
            ),
            subject_order=SubjectOrderConfig(
                sort_by=review_data.get("subject_order", {}).get("sort_by", "name"),
                direction=review_data.get("subject_order", {}).get("direction", "asc"),
                priority=SubjectOrderPriority(**priority_data) if priority_data else None,
            ),
        )

        post_data = data.get("post", {})
        post = PhaseConfig(
            directory=post_data.get("directory", ""),
            retry=RetryConfig(
                max_attempts=post_data.get("retry", {}).get("max_attempts", 1),
                on_failure=post_data.get("retry", {}).get("on_failure", "skip"),
            ),
        )

        return cls(
            name=name,
            version=version,
            output_dir=output_dir,
            pre=pre,
            review=review,
            post=post,
        )


@dataclass
class StepFile:
    """发现的步骤文件信息。"""

    path: Path
    step_type: str  # 'py' | 'md'
    stem: str  # 无扩展名的文件名
    order: int = 0  # 排序优先级


@dataclass
class StepResult:
    step_name: str
    status: str  # 'ok' | 'error' | 'skipped'
    error: str | None = None
    subject: str = ""
    data: dict = field(default_factory=dict)
    attempt: int = 1


@dataclass
class PipelineResult:
    subject: str = ""
    success: bool = True
    step_results: list[StepResult] = field(default_factory=list)
    report_dir: Path | None = None


# ============================================================================
# Step 发现与排序
# ============================================================================

_VALID_EXTENSIONS = {".py", ".md"}


def discover_steps(phase_dir: Path) -> list[StepFile]:
    """扫描阶段目录，发现所有 .py / .md 文件，按规则排序。

    排序优先级：pipeline.yaml 显式声明 > 文件名前缀（01-, 02-） > 文件名字典序。
    这里通过文件名前缀实现——文件级排序，pipeline.yaml 的覆盖在 pipeline config 中处理。
    """
    if not phase_dir.exists():
        return []

    files = [f for f in phase_dir.iterdir() if f.is_file() and f.suffix in _VALID_EXTENSIONS]

    def sort_key(f: Path) -> tuple:
        stem = f.stem
        # 提取前缀数字（如果有）
        prefix = 0
        rest = stem
        if stem.count("-") >= 1:
            parts = stem.split("-", 1)
            try:
                prefix = int(parts[0])
                rest = parts[1]
            except ValueError:
                pass

        # 有前缀（prefix > 0）排在没有前缀（prefix=0）之前
        has_prefix = 0 if prefix > 0 else 1
        return (has_prefix, prefix, rest)

    files.sort(key=sort_key)

    steps = []
    for f in files:
        stem = f.stem
        step_type = "py" if f.suffix == ".py" else "md"
        # 提取排序依据
        order = 0
        if stem.count("-") >= 1:
            parts = stem.split("-", 1)
            try:
                order = int(parts[0])
            except ValueError:
                pass
        steps.append(StepFile(path=f, step_type=step_type, stem=stem, order=order))

    return steps


# ============================================================================
# Step 执行器
# ============================================================================


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
    4. 拼接前缀 + 已替换的 prompt
    5. subprocess.run(["pi", "-m", final_prompt])
    6. 将 stdout 写入 output.json
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

    from paper_rag.template_engine import TemplateContext, resolve_variables

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
    from paper_rag.template_engine import build_agent_prefix

    prefix = build_agent_prefix(
        step_name=step.stem,
        step_dir=str(step_dir),
        prior_results=prior_results,
    )

    # 4. 拼接 final prompt
    final_prompt = prefix + resolved_md

    # 5. 调用 pi
    pi_binary = env.get("PIPELINE_PI_BINARY", "pi")
    try:
        proc = subprocess.run(
            [pi_binary, "-m", final_prompt],
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

    # 6. 处理 stdout → output.json
    output_text = proc.stdout.strip()

    # 尝试解析 stdout 作为 JSON
    output_file = step_dir / "output.json"
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


# ============================================================================
# 阶段执行
# ============================================================================


def _run_phase_steps(
    steps: list[StepFile],
    phase_name: str,
    subjects: list[str],
    phase_config: PhaseConfig,
    output_dir: Path,
    base_env: dict,
) -> dict[str, list[StepResult]]:
    """运行阶段内的所有步骤。

    Args:
        steps: 本阶段的 Step 列表。
        phase_name: 'pre' | 'review' | 'post'。
        subjects: Subject 名称列表（review 阶段逐篇，pre/post 批量）。
        phase_config: 阶段配置（含 retry）。
        output_dir: 输出根目录。
        base_env: 基础环境变量。

    Returns:
        {subject_name: [StepResult, ...]}
    """
    all_results: dict[str, list[StepResult]] = {s: [] for s in subjects}

    for step in steps:
        logger.info(
            "Phase %s / Step %s — processing %d subject(s)",
            phase_name,
            step.stem,
            len(subjects),
        )

        for subject in subjects:
            # 构建 intermediates 路径
            if phase_name == "review":
                step_dir = output_dir / "intermediates" / subject / step.stem
            else:
                step_dir = output_dir / "intermediates" / phase_name / step.stem

            env = {
                **base_env,
                "PIPELINE_PHASE": phase_name,
                "PIPELINE_SUBJECT": subject,
                "PIPELINE_STEP_NAME": step.stem,
                "PIPELINE_INTERMEDIATES": str(output_dir / "intermediates"),
            }

            # 重试循环
            result: StepResult | None = None
            for attempt in range(1, phase_config.retry.max_attempts + 1):
                logger.debug(
                    "  [%s] attempt %d/%d",
                    subject,
                    attempt,
                    phase_config.retry.max_attempts,
                )

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
                        break  # 不需要重试

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
# YAML 辅助
# ============================================================================


def _load_yaml(path: Path) -> dict:
    """加载 YAML 文件，返回 dict。"""
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ============================================================================
# Subject 排序
# ============================================================================


def _order_subjects(subjects: list[str], config: SubjectOrderConfig) -> list[str]:
    """按配置排序 Subject 列表。"""
    if not subjects:
        return []

    if config.sort_by == "regex" and config.priority:
        # 优先级排序：first 组的排在最前，last 组的排在最后
        first_patterns = config.priority.first or []
        last_patterns = config.priority.last or []

        def _group_key(name: str) -> tuple:
            for i, pat in enumerate(first_patterns):
                if re.search(pat, name):
                    return (0, i, name)
            for pat in last_patterns:
                if re.search(pat, name):
                    return (2, 0, name)
            return (1, 0, name)

        result = sorted(subjects, key=_group_key)
    else:
        result = sorted(subjects)

    direction = config.direction or "asc"
    if direction == "desc":
        result.reverse()

    return result


# ============================================================================
# output.json 校验
# ============================================================================


def validate_step_output(data: dict, step_name: str) -> dict:
    """校验并修复 step 输出，确保符合最小 schema。

    Args:
        data: 从 output.json 解析的字典。
        step_name: 步骤名（用于 fallback）。

    Returns:
        格式化/修复后的输出。
    """
    output = {
        "step": data.get("step", step_name),
        "status": data.get("status", "ok"),
        "error": data.get("error"),
        "data": data.get("data", {}),
    }
    # 强制 status 为合法值
    if output["status"] not in ("ok", "error", "skipped"):
        output["status"] = "ok"
    return output


# ============================================================================
# 主入口
# ============================================================================


def run_pipeline(
    pipeline_yaml: dict | str | Path,
    input_path: Path,
    pipeline_dir: Path | None = None,
    output_dir: Path | None = None,
    target_phase: str | None = None,
    target_step: str | None = None,
) -> PipelineResult:
    """执行一条完整的 pipeline（三段式：Pre → Review → Post）。

    Args:
        pipeline_yaml: pipeline 配置字典、YAML 文件路径或包含 pipeline.yaml 的目录。
        input_path: 输入 PDF 路径（单篇）或目录（多篇）。
        pipeline_dir: pipeline 定义目录的根。为 None 时从 pipeline_yaml 推断。
        output_dir: 覆盖配置中的 output_dir。
        target_phase: 仅运行指定阶段（'pre' / 'review' / 'post'）。
        target_step: 仅运行指定步骤名（需已有中间产物）。

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

    logger.info("Pipeline '%s' starting — %d subject(s)", config.name, len(subjects))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    base_env = {
        **os.environ,
        "PIPELINE_OUTPUT_DIR": str(config.output_dir.absolute()),
        "PIPELINE_PIPELINE_DIR": str(pipeline_dir.absolute()),
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
        )
        all_phase_results[phase_name] = phase_results

        # 汇总
        for subj, subj_results in phase_results.items():
            all_step_results.extend(subj_results)
            for r in subj_results:
                if r.status == "error":
                    overall_success = False

    # 构建结果
    report_dir = config.output_dir / "reports" / primary_subject

    return PipelineResult(
        subject=primary_subject,
        success=overall_success,
        step_results=all_step_results,
        report_dir=report_dir,
    )
