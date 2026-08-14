"""
超时估算器 —— 基于论文数量和文本规模的动态超时控制。

估算策略：
- .py 步骤：固定 180s（本地计算，但 03-batch-search 含 embedding/reranker 真推理：
  50 候选逐条精排在 2C/4G 机器上需 10-40s，大库加载索引另计）
- .md 步骤：基于 subject 文本长度 + subject 数量估算
  base_timeout = 60 + (total_chars / 1000) * 15
  多 subject 时乘以缓冲因子 1.2
  上限钳制到 900s，下限 60s

常量标定：
- _CHARS_PER_SEC_FACTOR = 15：每千字符额外给 15 秒，基于中等长度（~5000字）
  技术文章在本地 2C/4G 机器上 LLM 处理的实测估算。实际值是保守的——
  pi 子进程内包含 LLM API 往返+本地后处理，远高于纯推理时间。
- _MULTI_SUBJECT_FACTOR = 1.2：多 subject 缓冲因子，覆盖 Agent 上下文切换
  和中间产物 I/O 开销。
- 以上均为经验初始值，待长期运维数据标定。可考虑移至 config.yaml。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 常量
_MIN_TIMEOUT = 60  # 最小单步超时（秒）
_MAX_TIMEOUT = 900  # 最大单步超时（秒），对应 15 分钟
# .py 步骤固定超时：03-batch-search 加载 embedding + reranker 真推理（50 候选逐条
# 精排 + 大库索引加载），2C/4G 机器实测可能超过原 60s 基准，故提升至 180s
_PY_STEP_TIMEOUT = 180
_CHARS_PER_SEC_FACTOR = 45  # 每千字符额外超时秒数（提高以覆盖 API 延迟波动）
_MULTI_SUBJECT_FACTOR = 1.2  # 多 subject 缓冲因子


def estimate_step_timeout(
    step_type: str = "md",
    total_chars: int = 0,
    subject_count: int = 1,
) -> int:
    """估算单个 Step 的合理超时。

    Args:
        step_type: 'py' | 'md'
        total_chars: 所有 subject 的文本总字符数
        subject_count: subject 总数

    Returns:
        超时秒数，范围 [_MIN_TIMEOUT, _MAX_TIMEOUT]
    """
    if step_type == "py":
        return _PY_STEP_TIMEOUT

    # .md Agent 步骤：基于文本量估算 LLM 处理时间
    base = 60
    try:
        extra = int((total_chars / 1000) * _CHARS_PER_SEC_FACTOR)
    except (TypeError, ValueError):
        extra = 0
    timeout = base + extra

    # 多 subject 缓冲
    if subject_count > 1:
        try:
            timeout = int(timeout * _MULTI_SUBJECT_FACTOR)
        except (TypeError, ValueError):
            pass

    timeout = max(_MIN_TIMEOUT, min(_MAX_TIMEOUT, timeout))

    logger.debug(
        "Estimated step_timeout=%ds (step_type=%s, total_chars=%d, subjects=%d)",
        timeout,
        step_type,
        total_chars,
        subject_count,
    )
    return timeout
