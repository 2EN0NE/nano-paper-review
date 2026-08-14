"""
from __future__ import annotations

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
class IndexConfig:
    """Auto-Index 配置——pipeline.yaml 的 index 段。

    Attributes:
        store_dir: 搜索引擎数据目录（SQLite + FAISS），留空 → {data_dir}/index/。
        reference_dir: 参考论文 PDF 归档目录，留空 → {data_dir}/origin/pdf/。
        auto_index: 首次运行时自动对 reference_dir 做批量索引。
        copy_subjects: 将 review 的 subjects PDF 复制到 reference_dir。
    """

    store_dir: Path | None = None
    reference_dir: Path | None = None
    auto_index: bool = True
    copy_subjects: bool = True


@dataclass
class PoolConfig:
    """Worker 池化配置——Review Phase 中多 Subject 并行处理。

    Attributes:
        workers: 初始并发 Worker 数（fixed 模式的固定值，dynamic 模式的起始值）。
                 设为 0 自动根据 CPU 核数推导（上限 64）。
                 设为 1 退化为顺序执行。
        timeout: 单个 Subject 超时秒数（0 = 无超时）。
        ordered: 是否按 Subject 原始顺序返回结果（默认 True）。
        profile: 并发策略。'fixed' 固定 workers 数，'dynamic' 根据 API 限流/错误自适应调整。
        workers_min: dynamic 模式下的最小 worker 数（默认 1）。
        workers_max: dynamic 模式下的最大 worker 数（默认取 workers 值，上限 64）。
    """

    workers: int = 5
    timeout: int = 0
    ordered: bool = True
    profile: str = "fixed"  # 'fixed' | 'dynamic'
    workers_min: int = 1
    workers_max: int = 0  # 0 = 和 workers 相同
    granularity: str = (
        "subject"  # 'subject'（worker=一个 Subject 跑完全部 Steps）| 'step'（按 Step 分波次）
    )

    def __post_init__(self):
        # 粒度合法性校验：非法值回退 subject（保持现有行为）
        if self.granularity not in ("subject", "step"):
            logger.warning(
                "pool.granularity=%r is invalid — falling back to 'subject'", self.granularity
            )
            self.granularity = "subject"
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

        # workers_max 默认和 workers 相同
        if self.workers_max < 1:
            self.workers_max = self.workers

        # 约束 workers_min / workers_max
        self.workers_min = max(1, self.workers_min)
        self.workers_max = max(self.workers_min, min(self.workers_max, _POOL_WORKERS_MAX))


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
    display_name: str = ""  # 进度卡/报告显示名（空 → name.capitalize() 回退）
    retry: RetryConfig = field(default_factory=RetryConfig)
    duplicate_policy: str = "skip"  # 'skip' | 'rename' | 'error'
    step_timeout: int = 0  # 单 Step 超时秒数（0=无超时，或从 pool.timeout 继承）

    # batch-only
    manifest_step: str = ""

    # per_subject-only
    subject_source: SubjectSourceConfig | None = None
    subject_order: SubjectOrderConfig | None = None
    pool: PoolConfig | None = None

    @property
    def display_label(self) -> str:
        """阶段显示名：显式 display_name 优先，否则 name 首字母大写回退。"""
        return self.display_name or self.name.capitalize()


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
        display_name=data.get("display_name", ""),
        retry=RetryConfig(
            max_attempts=data.get("retry", {}).get("max_attempts", 1),
            on_failure=data.get("retry", {}).get("on_failure", "skip"),
        ),
        duplicate_policy=data.get("duplicate_policy", "skip"),
        manifest_step=data.get("manifest_step", ""),
        step_timeout=data.get("step_timeout", 0),
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
                profile=pool_data.get("profile", "fixed"),
                workers_min=pool_data.get("workers_min", 1),
                workers_max=pool_data.get("workers_max", 0),
                granularity=pool_data.get("granularity", "subject"),
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

    @classmethod
    def discover_all(cls, pipelines_dir: Path) -> list[tuple[str, str]]:
        """扫描 pipelines/ 子目录，返回 [(目录名, 管线名), ...]。

        Args:
            pipelines_dir: pipelines/ 根目录路径。

        Returns:
            (目录名, 管线名) 列表，按目录名排序。目录不存在或无子目录时返回空列表。
        """
        if not pipelines_dir.is_dir():
            return []

        results: list[tuple[str, str]] = []
        for entry in sorted(pipelines_dir.iterdir()):
            if not entry.is_dir():
                continue
            yaml_file = entry / "pipeline.yaml"
            if not yaml_file.is_file():
                continue
            # 从 pipeline.yaml 读取 name 字段作为显示名，fallback 目录名
            # _load_yaml 内部已捕获 (OSError, yaml.YAMLError)，无额外异常需处理
            raw = _load_yaml(yaml_file)
            display_name = raw.get("name", entry.name)
            results.append((entry.name, display_name))

        return results


def resolve_pipeline_dir(
    data_dir: Path,
    pipeline_name: str | None = None,
) -> Path | None:
    """从 pipelines/ 解析管线目录路径。

    优先项目级 ./.paper-review/，回退用户级 ~/.paper-review/。

    Args:
        data_dir: 数据目录路径（如 ~/.paper-review）。
        pipeline_name: 管线名称。为 None 时：
            - 仅一个管线时自动返回
            - 多个管线时返回 None（调用方应做交互式选择）

    Returns:
        管线目录路径，或 None（需要交互选择或无管线）。
    """
    pipelines_dir = data_dir / "pipelines"
    discovered = PipelineConfig.discover_all(pipelines_dir)

    if not discovered:
        return None

    # 指定名称：精确匹配
    if pipeline_name is not None:
        for dir_name, _ in discovered:
            if dir_name == pipeline_name:
                return pipelines_dir / dir_name
        return None

    # 不指定名称：只有唯一一个时自动选择
    if len(discovered) == 1:
        return pipelines_dir / discovered[0][0]

    return None  # 多个管线，需交互选择


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
    task_dir: Path = field(
        default_factory=lambda: Path("")
    )  # run_pipeline 恒赋值；空值仅 dataclass 默认占位
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
