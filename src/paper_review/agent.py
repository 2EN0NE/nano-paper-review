"""Agent 抽象 —— LLM Agent（pi / opencode）的命令构建与模型选择。

统一 .md Agent 步骤（``pipeline_steps.AgentRunner``）与 .py 步骤
（``04-extract-features`` 等模板）对嵌套 Agent 的调用，避免命令拼装漂移。

模型选择策略（留空兜底，降低对特定 provider/model 硬编码的弱依赖）：

- **provider/model 留空 → 不传对应 CLI flag**，由 Agent 继承自身默认
  （pi 继承 ``PI_PROVIDER``/``PI_MODEL`` 环境；opencode 用自身 config 默认）。

升级链（Agent Escalation Chain，ADR 0017）：

- ``agent.escalate`` 定义「第 N 次尝试用哪条命令」的完整命令行序列（str 或
  argv 列表），框架解析后按 ``_retry_step`` 的 attempt 单调推进、顶部饱和。
- ``build_command`` 仅用于 ``escalate`` 缺省时的向后兼容单命令路径。

当前仅实现 pi；opencode 预留（入参形状不同：opencode 用 ``--model
provider/model`` 合并串，pi 用 ``--provider``/``--model`` 两个分离 flag）。
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field

# 已实现的 Agent 类型（opencode 预留，暂未实现）
SUPPORTED_AGENT_TYPES = ("pi",)

# 环境变量名（orchestrator 注入 → AgentRunner / .py 步骤读取）
ENV_AGENT_TYPE = "PIPELINE_AGENT_TYPE"
ENV_AGENT_PROVIDER = "PIPELINE_AGENT_PROVIDER"
ENV_AGENT_MODEL = "PIPELINE_AGENT_MODEL"
# 升级链（命令列表，JSON 序列化后注入；.md 由 _retry_step 消费，.py 由脚本消费）
ENV_AGENT_ESCALATE = "PIPELINE_AGENT_ESCALATE"
# _retry_step 为 .md 步骤按 attempt 解析出的单条命令（argv 的 JSON）
ENV_AGENT_COMMAND = "PIPELINE_AGENT_COMMAND"
# 升级链的每步骤总预算（retry.max_attempts）；调 pi 的 .py 步骤据此封顶内部迭代
ENV_AGENT_MAX_ATTEMPTS = "PIPELINE_AGENT_MAX_ATTEMPTS"


@dataclass
class AgentConfig:
    """Agent 启动配置 —— pipeline.yaml 的 ``agent`` 段（全局或 phase 级）。

    Attributes:
        type: Agent 类型（``"pi"``；``"opencode"`` 预留未实现）。
        provider: Agent 的 provider（空 = 不传 flag，继承 Agent 默认）。
        model: Agent 的 model（空 = 不传 flag，继承 Agent 默认）。
        escalate: 升级链命令列表（每条是完整命令行 str 或 argv list[str]；
            空 = 不启用升级链，回退 provider/model 单命令路径）。
    """

    type: str = "pi"
    provider: str = ""
    model: str = ""
    escalate: list = field(default_factory=list)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> AgentConfig:
        """从 step 环境变量解析 Agent 配置（缺省 type=pi、provider/model/escalate 空）。"""
        escalate: list = []
        raw = env.get(ENV_AGENT_ESCALATE, "") or ""
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                # 大声失败：升级链由框架 json.dumps 注入，解析失败说明配置/环境损坏，
                # 静默回退 [] 会悄悄禁用升级链并掩盖损坏信号（快速失败原则）。
                raise ValueError(f"{ENV_AGENT_ESCALATE} 不是合法 JSON: {raw!r}") from e
            if not isinstance(parsed, list):
                raise ValueError(
                    f"{ENV_AGENT_ESCALATE} 必须是 JSON 数组，实际为 {type(parsed).__name__}: {raw!r}"
                )
            escalate = parsed
        return cls(
            type=env.get(ENV_AGENT_TYPE, "pi") or "pi",
            provider=env.get(ENV_AGENT_PROVIDER, "") or "",
            model=env.get(ENV_AGENT_MODEL, "") or "",
            escalate=escalate,
        )

    def has_explicit_model(self) -> bool:
        """是否显式配置了 provider/model（phase 级覆盖判定用）。"""
        return bool(self.provider or self.model)


def model_args(cfg: AgentConfig) -> list[str]:
    """按 Agent 类型生成 provider/model 相关 CLI args（留空不传）。

    pi: ``--provider X --model Y``（两个分离 flag）。
    opencode: ``--model provider/model``（合并串）——预留，未实现。
    """
    if cfg.type == "pi":
        args: list[str] = []
        if cfg.provider:
            args += ["--provider", cfg.provider]
        if cfg.model:
            args += ["--model", cfg.model]
        return args
    # pi-lens-ignore: no-raise-not-implemented — opencode 预留，未实现时明确失败而非静默降级
    raise NotImplementedError(
        f"agent type {cfg.type!r} 未实现（当前仅支持 {SUPPORTED_AGENT_TYPES}）"
    )


def build_command(
    cfg: AgentConfig,
    binary: str,
    prompt_file: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    """构建完整 Agent 命令行（含 provider/model flag）。

    pi: ``[binary, *model_args, *extra_args, --no-session, -p @prompt_file]``
    """
    if cfg.type == "pi":
        return [
            binary,
            *model_args(cfg),
            *(extra_args or []),
            "--no-session",
            "-p",
            f"@{prompt_file}",
        ]
    # pi-lens-ignore: no-raise-not-implemented — opencode 预留，未实现时明确失败而非静默降级
    raise NotImplementedError(
        f"agent type {cfg.type!r} 未实现（当前仅支持 {SUPPORTED_AGENT_TYPES}）"
    )


def parse_escalation_chain(escalate: list) -> list[list[str]]:
    """将 ``agent.escalate`` 解析为 argv 列表的列表。

    每条可以是完整命令行字符串（``shlex.split`` 解析，支持引号）或 argv 列表
    （原样转 str）。空条目跳过。返回的 argv 是「命令前缀」——不含 prompt 注入
    （``--no-session -p @prompt.md`` 由框架追加，见 append_prompt_args）。
    """
    chain: list[list[str]] = []
    for entry in escalate:
        if isinstance(entry, str):
            parts = shlex.split(entry)
        elif isinstance(entry, (list, tuple)):
            parts = [str(x) for x in entry]
        else:
            continue
        if parts:
            chain.append(parts)
    return chain


def resolve_command_for_attempt(chain: list[list[str]], attempt: int) -> list[str] | None:
    """返回第 attempt 次尝试（1-based）应使用的命令 argv。

    链空返回 None（调用方回退 build_command 单命令路径）；attempt 超出链长时
    饱和到末条（顶部饱和语义，ADR 0017）。
    """
    if not chain:
        return None
    idx = min(attempt - 1, len(chain) - 1)
    return chain[idx]


def append_prompt_args(argv: list[str], prompt_file: str) -> list[str]:
    """框架兜底：在命令前缀后追加 ``--no-session -p @prompt_file``。

    prompt 文件路径运行时生成、``--no-session`` 是批处理不污染会话的必需项，
    两者由框架统一追加，升级链条目只负责「二进制 + 任意 flag + 模型」。
    """
    return [*argv, "--no-session", "-p", f"@{prompt_file}"]
