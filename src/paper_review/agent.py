"""Agent 抽象 —— LLM Agent（pi / opencode）的命令构建与模型选择。

统一 .md Agent 步骤（``pipeline_steps.AgentRunner``）与 .py 步骤
（``04-extract-features`` 等模板）对嵌套 Agent 的调用，避免命令拼装漂移。

模型选择策略（留空兜底，降低对特定 provider/model 硬编码的弱依赖）：

- **provider/model 留空 → 不传对应 CLI flag**，由 Agent 继承自身默认
  （pi 继承 ``PI_PROVIDER``/``PI_MODEL`` 环境；opencode 用自身 config 默认）。
- **显式传了 provider/model 但非零退出报错 → 回退为不传（Agent 默认）**，
  由调用方记录 warning。

当前仅实现 pi；opencode 预留（入参形状不同：opencode 用 ``--model
provider/model`` 合并串，pi 用 ``--provider``/``--model`` 两个分离 flag）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# 已实现的 Agent 类型（opencode 预留，暂未实现）
SUPPORTED_AGENT_TYPES = ("pi",)

# 环境变量名（orchestrator 注入 → AgentRunner / .py 步骤读取）
ENV_AGENT_TYPE = "PIPELINE_AGENT_TYPE"
ENV_AGENT_PROVIDER = "PIPELINE_AGENT_PROVIDER"
ENV_AGENT_MODEL = "PIPELINE_AGENT_MODEL"


@dataclass
class AgentConfig:
    """Agent 启动配置 —— pipeline.yaml 的 ``agent`` 段（全局或 phase 级）。

    Attributes:
        type: Agent 类型（``"pi"``；``"opencode"`` 预留未实现）。
        provider: Agent 的 provider（空 = 不传 flag，继承 Agent 默认）。
        model: Agent 的 model（空 = 不传 flag，继承 Agent 默认）。
    """

    type: str = "pi"
    provider: str = ""
    model: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> AgentConfig:
        """从 step 环境变量解析 Agent 配置（缺省 type=pi、provider/model 空）。"""
        return cls(
            type=env.get(ENV_AGENT_TYPE, "pi") or "pi",
            provider=env.get(ENV_AGENT_PROVIDER, "") or "",
            model=env.get(ENV_AGENT_MODEL, "") or "",
        )

    def has_explicit_model(self) -> bool:
        """是否显式配置了 provider/model（决定报错时是否回退重试）。"""
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


def build_command_without_model(
    cfg: AgentConfig,
    binary: str,
    prompt_file: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    """构建不带 provider/model flag 的 Agent 命令行（报错回退用）。"""
    return build_command(AgentConfig(type=cfg.type), binary, prompt_file, extra_args)


# 模型配置错误特征（stderr 命中才触发「回退为不传 model flag」重试）。
# 仅这类错误才回退：显式 provider/model 无效/欠费/不存在。其他非零退出
# （prompt 崩溃、网络抖动、限流等）应直接传播为 error，避免静默改用默认
# 模型掩盖根因、造成同批 review 内模型不一致。
_MODEL_CONFIG_ERROR_PATTERNS = (
    "402",
    "insufficient balance",
    "insufficient quota",
    "model not found",
    "model not exist",
    "unknown model",
    "invalid model",
    "model not supported",
    "no such model",
)


def is_model_config_error(stderr: str) -> bool:
    """判断 stderr 是否命中「显式 provider/model 配置无效」特征。

    只有命中才触发回退为「不传 model flag」（Agent 默认）重试。
    依据错误码（402）或稳定标识符（insufficient balance / model not found 等），
    不做宽泛子串匹配。
    """
    if not stderr:
        return False
    lowered = stderr.lower()
    return any(pattern in lowered for pattern in _MODEL_CONFIG_ERROR_PATTERNS)
