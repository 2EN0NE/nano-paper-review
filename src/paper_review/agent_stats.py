"""Agent 观测（Agent Observation）——计数器存储 + 异常分类（ADR 0018）。

agent-stats.json 按管线分桶存储聚合计数（total_steps / total_anomalies /
by_kind / by_command + agent 段指纹）。原始细节留在 paper-review.log，不重复落盘。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from paper_review.pipeline_steps import AGENT_OUTPUT_NOT_JSON_ERROR

logger = logging.getLogger(__name__)


def classify_kind(error: str) -> str:
    """把失败原因归一化为开放式 kind key（兜底 exception:<Name>）。

    覆盖现有 emit 点（AgentRunner / _retry_step / 降级哨兵）的失败形态；
    未知形态回退 exception:<首 token>，保证新异常自动落新 key 而无需改枚举。
    """
    if not error:
        return "error"
    e = error.strip()
    low = e.lower()

    if "timed out" in low or "never started" in low:
        return "timeout"
    if AGENT_OUTPUT_NOT_JSON_ERROR in e:
        return "json_format"
    if "auth" in low and "unavailable" in low:
        return "auth_unavailable"
    if "429" in e or "rate limit" in low or "too many requests" in low:
        return "rate_limited_429"
    if "503" in e:
        return "server_error_503"
    if "binary" in low and "not found" in low:
        return "binary_missing"

    m = re.search(r"exited with code (\d+)", low)
    if m:
        return f"exit:{m.group(1)}"

    first = e.split()[0] if e.split() else "error"
    return f"exception:{first.rstrip(':,.')}"


class AgentStatsRecorder:
    """线程安全的内存计数器（运行期累加，结束单次落盘）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_steps = 0
        self.total_anomalies = 0
        self.by_kind: dict[str, int] = {}
        self.by_command: dict[str, dict[str, int]] = {}

    def record(self, ok: bool, kind: str = "", command: str = "") -> None:
        with self._lock:
            self.total_steps += 1
            if command:
                slot = self.by_command.setdefault(command, {"steps": 0, "anomalies": 0})
                slot["steps"] += 1
            if not ok:
                self.total_anomalies += 1
                self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
                if command:
                    self.by_command[command]["anomalies"] += 1

    def record_anomaly(self, kind: str) -> None:
        """记录一个非步骤执行的异常（如降级哨兵）：只增异常数，不增 total_steps。"""
        with self._lock:
            self.total_anomalies += 1
            self.by_kind[kind] = self.by_kind.get(kind, 0) + 1


def compute_fingerprint(agent: Any) -> str:
    """agent 段指纹（escalate + type，ADR 0018）。变化 → 该管线统计清零。"""
    payload = {"type": getattr(agent, "type", "pi"), "escalate": getattr(agent, "escalate", [])}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def default_stats(fingerprint: str) -> dict:
    return {
        "fingerprint": fingerprint,
        "total_steps": 0,
        "total_anomalies": 0,
        "by_kind": {},
        "by_command": {},
    }


def load_stats(path: Path) -> dict:
    """加载 agent-stats.json；缺失/损坏返回 {"pipelines": {}}。"""
    if not path.exists():
        return {"pipelines": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("pipelines"), dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"pipelines": {}}


def save_stats(path: Path, data: dict) -> None:
    """原子写回 agent-stats.json（tmp + rename）。

    写失败（磁盘满/权限不足）记录告警而非静默吞掉——agent-stats 是观测数据，
    写失败不阻断 review 主流程（边界安全降级），但必须留下失败信号（ADR 0018）。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        logger.warning("agent-stats 写盘失败（%s）: %s", path, e)
