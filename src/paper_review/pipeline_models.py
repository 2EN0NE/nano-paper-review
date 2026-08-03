"""
管线数据模型

定义 Pipeline 的配置结构、运行时数据类型、Step 发现与排序。
从 orchestrator.py 拆出，供 CLI / 测试 / orchestrator 共用。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from paper_review.logging_config import get_logger

logger = get_logger("orchestrator")

# Pool 配置的硬上限
_POOL_WORKERS_MAX = 64


# ============================================================================
# 进度跟踪
# ============================================================================


@dataclass
class PoolProgressEvent:
    """单个进度事件。"""

    event_type: str  # 'subject_start' | 'subject_complete' | 'subject_fail'
    subject: str = ""
    timestamp: str = ""
    step_count: int = 0
    error: str = ""


class PoolProgress:
    """Worker 池进度跟踪器。

    收集 Subject 级别的事件（开始/完成/失败），
    供 CLI 实时展示或测试验证。
    """

    def __init__(self):
        self.events: list[PoolProgressEvent] = []

    def on_subject_start(self, subject: str) -> None:
        self.events.append(
            PoolProgressEvent(
                event_type="subject_start",
                subject=subject,
                timestamp=datetime.now().isoformat(timespec="seconds"),
            )
        )

    def on_subject_complete(self, subject: str, step_results: list) -> None:
        self.events.append(
            PoolProgressEvent(
                event_type="subject_complete",
                subject=subject,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                step_count=len(step_results),
            )
        )

    def on_subject_fail(self, subject: str, status: str, error: str) -> None:
        self.events.append(
            PoolProgressEvent(
                event_type="subject_fail",
                subject=subject,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                error=error,
            )
        )

    @property
    def total(self) -> int:
        return len([e for e in self.events if e.event_type == "subject_start"])

    @property
    def completed(self) -> int:
        return len([e for e in self.events if e.event_type == "subject_complete"])

    @property
    def failed(self) -> int:
        return len([e for e in self.events if e.event_type == "subject_fail"])

    @property
    def pending(self) -> int:
        return self.total - self.completed - self.failed

    def summary(self) -> str:
        return f"{self.total} total, {self.completed} \u2713, {self.failed} \u2717, {self.pending} pending"


# ============================================================================
# Pipeline 配置
# ============================================================================


@dataclass
class PoolConfig:
    """Worker 池化配置——Review Phase 中多 Subject 并行处理。

    Attributes:
        workers: 最大并发 Worker 数。
                 设为 0 自动根据 CPU 核数推导（上限 64）。
                 设为 1 退化为顺序执行。
        timeout: 单个 Subject 超时秒数（0 = 无超时）。
        ordered: 是否按 Subject 原始顺序返回结果（默认 True）。
    """

    workers: int = 5
    timeout: int = 0
    ordered: bool = True

    def __post_init__(self):
        # 自动推导：workers=0 时根据 CPU 核数
        if self.workers == 0:
            cpus = os.cpu_count() or 1
            self.workers = min(cpus, _POOL_WORKERS_MAX)
            logger.info("Auto-detected pool.workers=%d (from %d CPU(s))", self.workers, cpus)

        # 下限
        if self.workers < 1:
            logger.warning("pool.workers=%d is too low, clamping to 1", self.workers)
            self.workers = 1

        # 上限
        if self.workers > _POOL_WORKERS_MAX:
            logger.warning(
                "pool.workers=%d exceeds max %d, clamping to %d",
                self.workers,
                _POOL_WORKERS_MAX,
                _POOL_WORKERS_MAX,
            )
            self.workers = _POOL_WORKERS_MAX


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
    """单个管线阶段的配置。

    mode='batch' 的阶段对所有 Subject 批量执行一次；
    mode='per_subject' 的阶段对每个 Subject 逐个执行，支持 Worker 池并发。
    """

    name: str = ""
    mode: str = "batch"  # 'batch' | 'per_subject'
    directory: str = ""
    retry: RetryConfig = field(default_factory=RetryConfig)
    duplicate_policy: str = "skip"  # 'skip' | 'rename' | 'error'

    # batch-only
    manifest_step: str = ""

    # per_subject-only
    subject_source: SubjectSourceConfig | None = None
    subject_order: SubjectOrderConfig | None = None
    pool: PoolConfig | None = None


@dataclass
class SubjectSourceConfig:
    """Subject 来源配置。"""

    type: str = "cli"  # 'cli' | 'manifest'
    path: str = ""  # manifest 文件路径（type=manifest 时有效）


def _parse_phase(data: dict) -> PhaseConfig:
    """从字典解析单个 PhaseConfig。"""
    mode = data.get("mode", "batch")
    subject_source_data = data.get("subject_source")
    priority_data = data.get("subject_order", {}).get("priority")
    pool_data = data.get("pool")

    return PhaseConfig(
        name=data.get("name", ""),
        mode=mode,
        directory=data.get("directory", ""),
        retry=RetryConfig(
            max_attempts=data.get("retry", {}).get("max_attempts", 1),
            on_failure=data.get("retry", {}).get("on_failure", "skip"),
        ),
        duplicate_policy=data.get("duplicate_policy", "skip"),
        manifest_step=data.get("manifest_step", ""),
        subject_source=(
            SubjectSourceConfig(
                type=subject_source_data.get("type", "cli"),
                path=subject_source_data.get("path", ""),
            )
            if subject_source_data
            else None
        ),
        subject_order=SubjectOrderConfig(
            sort_by=data.get("subject_order", {}).get("sort_by", "name"),
            direction=data.get("subject_order", {}).get("direction", "asc"),
            priority=SubjectOrderPriority(**priority_data) if priority_data else None,
        ),
        pool=(
            PoolConfig(
                workers=pool_data.get("workers", 5),
                timeout=pool_data.get("timeout", 0),
                ordered=pool_data.get("ordered", True),
            )
            if pool_data
            else None
        ),
    )


@dataclass
class PipelineConfig:
    name: str = "unnamed"
    version: str = "1.0"
    output_dir: Path = Path("./output")
    phases: list[PhaseConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        name = data.get("name", "unnamed")
        version = data.get("version", "1.0")
        output_dir = Path(data.get("output_dir", "./output"))

        phases_data = data.get("phases", [])
        phases: list[PhaseConfig] = []
        for pd in phases_data:
            phase = _parse_phase(pd)
            phases.append(phase)

        return cls(
            name=name,
            version=version,
            output_dir=output_dir,
            phases=phases,
        )

    @classmethod
    def from_path(cls, path: Path) -> PipelineConfig:
        """从 YAML 文件或目录加载 PipelineConfig。

        path 可以是 pipeline.yaml 文件路径，或包含 pipeline.yaml 的目录。
        """
        if path.is_dir():
            yaml_file = path / "pipeline.yaml"
            if yaml_file.exists():
                raw = _load_yaml(yaml_file)
            else:
                raw = {"name": "default", "output_dir": "./output"}
        elif path.suffix in (".yaml", ".yml"):
            raw = _load_yaml(path)
        else:
            raw = {"name": "default", "output_dir": "./output"}
        return cls.from_dict(raw)


# ============================================================================
# 运行时模型
# ============================================================================


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
    task_id: str = ""
    task_dir: Path | None = None
    conclusion: str = ""


# ============================================================================
# Step 发现与排序
# ============================================================================

_VALID_EXTENSIONS = {".py", ".md"}


def discover_steps(phase_dir: Path) -> list[StepFile]:
    """扫描阶段目录，发现所有 .py / .md 文件，按规则排序。

    排序优先级（由高到低）：
    1. pipeline.yaml 中显式声明的 steps 顺序
    2. 文件名前缀数字（如 01-search.py > 02-*.py）
    3. OS 原生排序（稳定兜底）
    """
    if not phase_dir.is_dir():
        return []
    steps: list[StepFile] = []
    for f in sorted(phase_dir.iterdir()):
        if f.suffix.lower() in _VALID_EXTENSIONS and not f.name.startswith("."):
            step_type = f.suffix.lower()[1:]  # '.py' → 'py', '.md' → 'md'
            steps.append(StepFile(path=f, step_type=step_type, stem=f.stem))

    # 解析文件名前缀数字作为 order（无前缀=999）
    prefix_re = re.compile(r"^(\d+)")
    for s in steps:
        m = prefix_re.match(s.stem)
        try:
            s.order = int(m.group(1)) if m else 999
        except ValueError:
            s.order = 999

    steps.sort(key=lambda s: (s.order, s.stem))
    return steps


# ============================================================================
# 工具函数
# ============================================================================


def _load_yaml(path: Path) -> dict:
    """加载 YAML 文件，返回 dict。文件不存在或解析失败时返回空 dict。"""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("Failed to load YAML from %s: %s", path, e)
        return {}


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
