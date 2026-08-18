"""
04-extract-features.py — 从 Subject 首段提取技术关键词（辅助信号）

关键词从 Subject 正文首段（约 500 字）提取，作为给评审 Agent 的辅助参考
信号，不再是检索输入（ADR 0008）。

匹配词表来自自更新标签库（papers.tags 聚合）；标签库为空时回退冷启动
种子词表（_SEED_KEYWORDS）。

结果写入 per-subject intermediates（intermediates/{subject}/04-extract-features/
output.json），供 Review Phase 评分步骤通过模板变量读取。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

from paper_review.extractor import extract_pdf_with_timeout
from paper_review.search.search_types import QUERY_FIRST_PARA_CHARS

logger = logging.getLogger("paper_review.pre")

# 单篇 extract_pdf 软超时（秒）：PyMuPDF 同步调用无超时，卡死篇经 daemon 线程软超时跳过
_EXTRACT_PDF_TIMEOUT = 60
# 卡死篇阈值：累计超时达此值后不再启动新线程（直接返回空串走词表兑底），
# 防止批内多个卡死 PDF 持续累积 daemon 线程与打开的文档句柄直到进程退出
_MAX_STUCK_PDFS = 5
_stuck_count = 0

# 冷启动种子词表（标签库为空时兑底；随评审积累的标签库会替代它）
_SEED_KEYWORDS = [
    "深度学习",
    "神经网络",
    "卷积",
    "注意力",
    "Transformer",
    "图神经网络",
    "强化学习",
    "迁移学习",
    "联邦学习",
    "元学习",
    "BERT",
    "GPT",
    "LSTM",
    "GRU",
    "CNN",
    "RNN",
    "GAN",
    "知识图谱",
    "推荐系统",
    "计算机视觉",
    "自然语言处理",
    "聚类",
    "分类",
    "回归",
    "降维",
    "特征提取",
    "模型压缩",
    "量化",
    "剪枝",
    "知识蒸馏",
    "对比学习",
    "自监督",
    "预训练",
    "微调",
]


def _load_tag_library(store_dir: str) -> list[str]:
    """从 papers.tags 聚合历史标签库（去重保序）。无库/无标签返回空列表。"""
    db_path = Path(store_dir) / "index.sqlite"
    if not db_path.exists():
        return []
    tags: list[str] = []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT tags FROM papers WHERE tags IS NOT NULL AND tags != '[]' ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    for (raw,) in rows:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, list):
            for t in parsed:
                if isinstance(t, str) and t.strip() and t not in tags:
                    tags.append(t.strip())
    return tags


def extract_keywords(text: str, tag_library: list[str]) -> list[str]:
    """从文本中提取技术关键词（词表子串匹配，去重保序）。"""
    found = [kw for kw in tag_library if kw in text]
    return list(dict.fromkeys(found))


def merge_feature_sets(llm_features: list[str], keyword_features: list[str]) -> list[str]:
    """并集汇总 LLM 抽取与词表兜底的技术特征，去重保序（LLM 优先）。

    ADR 0015：词表匹配是 LLM 抽取的确定性兜底——LLM 不稳定可能漏掉标签库
    已有的词，词表保证已知词永不丢。并集天然去重，无冲突。
    """
    return list(dict.fromkeys([*llm_features, *keyword_features]))


def _build_feature_prompt(text: str, tag_library: list[str] | None = None) -> str:
    """构建 LLM 抽取技术方法关键词的 prompt（要求只输出 JSON 数组）。

    粒度约束（ADR 0015）：消歧与规范化前置到抽取端——LLM 结合文章上下文
    消歧（CMS → 并发标记清除）、统一中文全称（GC → 垃圾回收），并优先对齐
    参考词表（标签库）。检索端只做简单匹配，不做跨语言/歧义消解。
    """
    lib_block = ""
    if tag_library:
        # 截断前 100 词防 prompt 爆炸（标签库随评审积累增长）
        lib_block = (
            "\n参考规范词表（这些词已规范化，优先从中选择；词表没有的按上述规则规范化）：\n"
            + "、".join(tag_library[:100])
            + "\n"
        )
    return (
        "你是一名技术论文分析助手。请从下面的论文首段中抽取「技术方法」关键词。\n\n"
        "要求：\n"
        "1. 结合文章上下文理解术语含义，消歧后输出规范化的技术名词：\n"
        "   - 有歧义的缩写必须根据上下文判断并展开为全称"
        "（如 GC 语境的「CMS」→「并发标记清除」）。\n"
        "   - 无歧义的官方名称保留原名（如「ZGC」「G1」「ParNew」）。\n"
        "   - 同一概念统一用中文全称（如「垃圾回收」而非「GC」「Garbage Collection」）。\n"
        "2. 抽取 3~8 个，按重要性降序。\n"
        "3. 只输出一个 JSON 数组，不要任何说明、注释或 Markdown 代码块标记。\n"
        f"{lib_block}"
        f"\n论文首段：\n{text}\n\n"
        "输出（JSON 数组）："
    )


def _parse_feature_json(output: str) -> list[str]:
    """从 pi 输出提取技术特征 JSON 数组（容忍 Markdown 包裹）。

    pi 输出可能是纯 JSON 数组，也可能被 ```json ... ``` 代码块包裹，
    或夹杂前后说明文字。失败返回空列表（由词表兜底）。
    """
    import re

    if not output.strip():
        return []
    # 优先提取 fenced code block（容错 Markdown 包裹，ADR 0014 Bug C）
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", output)
    if m:
        candidate = m.group(1)
    else:
        # 退化：查找第一个 [ ... ] 片段
        m2 = re.search(r"\[[\s\S]*?\]", output)
        candidate = m2.group(0) if m2 else output
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _write_json(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  ✗ 写入 {path} 失败: {e}")


def _extract_pdf_with_timeout(pdf_path: str, timeout: int | float = _EXTRACT_PDF_TIMEOUT) -> str:
    """提取 PDF 文本，超时/异常返回空串（卡死篇不阻塞批次）。

    extract_pdf_with_timeout 的本地包装：累计卡死篇数达到 _MAX_STUCK_PDFS 后
    不再启动新线程（直接返回空串），避免多个卡死 PDF 的 daemon 线程与文档句柄
    持续累积到进程结束。失败原因记录到 paper-review.log（文件日志）。
    """
    global _stuck_count
    if _stuck_count >= _MAX_STUCK_PDFS:
        logger.warning(
            "04: 已达卡死阈值（%d 篇），跳过提取（词表兑底）: %s", _MAX_STUCK_PDFS, pdf_path
        )
        return ""
    text, timed_out = extract_pdf_with_timeout(pdf_path, timeout=timeout)
    if timed_out:
        _stuck_count += 1
    return text


def _process_subject(
    name: str,
    pdf_path: Path,
    subject_paper_ids: dict[str, str],
    tag_library: list[str],
    store,
    intermediates_dir: str,
    pi_binary: str,
    extract_timeout: int = _EXTRACT_PDF_TIMEOUT,
) -> tuple[int, int, int]:
    """处理单篇：提取 → 词表 → LLM → 写回 features → 写 per-subject 产物。

    单篇失败隔离：extract_pdf 失败/超时返回空文本、LLM 失败由词表兜底——
    均不中断批次；耗时与失败原因记入 paper-review.log。

    Returns: (features_written_delta, llm_feature_count, feature_count)
    """
    t0 = time.monotonic()
    raw_text = _extract_pdf_with_timeout(str(pdf_path), timeout=extract_timeout)
    extract_t = time.monotonic() - t0

    first_para = raw_text[:QUERY_FIRST_PARA_CHARS]
    text_with_name = first_para + " " + name.replace("-", " ")
    # 词表兜底（确定性，保证已知词永不丢）
    keyword_features = extract_keywords(text_with_name, tag_library)
    # LLM 主线（发现新词 + 已知词）
    t1 = time.monotonic()
    llm_features = _llm_extract_features(
        text_with_name, pi_binary=pi_binary, tag_library=tag_library
    )
    llm_t = time.monotonic() - t1
    # 并集汇总
    features = merge_feature_sets(llm_features, keyword_features)

    # 写回 papers.features（立即写，使同批后续 Subject 可检索到）
    features_written = 0
    paper_id = subject_paper_ids.get(name)
    if store is not None and paper_id and features:
        try:
            if store.update_features(paper_id, features):
                features_written = 1
        except Exception as e:  # noqa: BLE001 — 写回失败不中断批次
            logger.warning("04: %s features 写回失败: %s", name, e)

    _write_json(
        Path(intermediates_dir) / name / "04-extract-features" / "output.json",
        {
            "step": "04-extract-features",
            "status": "ok",
            "error": None,
            "data": {
                # 兼容 06-direct-scoring 的 {intermediates.04-extract-features.data.keywords}
                "keywords": keyword_features,
                "keyword_count": len(keyword_features),
                # L3 精排数据源（ADR 0015）
                "features": features,
                "feature_count": len(features),
                "llm_features": llm_features,
            },
        },
    )

    logger.info(
        "04 %s: extract=%.1fs llm=%.1fs features=%d (llm=%d, kw=%d, written=%d)",
        name,
        extract_t,
        llm_t,
        len(features),
        len(llm_features),
        len(keyword_features),
        features_written,
    )
    return features_written, len(llm_features), len(features)


def _load_subject_paper_ids(intermediates_dir: str) -> dict[str, str]:
    """读 02-auto-index 产出的 subject name → paper_id 映射。

    用于写回 papers.features（需要 paper_id 定位索引中的 Subject）。
    """
    path = Path(intermediates_dir) / "pre" / "02-auto-index" / "output.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8")).get("data", {})
        ids = data.get("subject_paper_ids", {})
        if isinstance(ids, dict):
            return ids
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _llm_extract_features(
    text: str,
    pi_binary: str = "pi",
    timeout: int = 300,
    tag_library: list[str] | None = None,
) -> list[str]:
    """调 pi（LLM）从文本抽取技术方法关键词（LLM 主线，ADR 0015）。

    失败（超时/非零退出/无法解析）返回空列表，由词表兜底。
    """
    import subprocess
    import tempfile

    prompt = _build_feature_prompt(text, tag_library=tag_library)
    try:
        fd, prompt_file = tempfile.mkstemp(suffix=".md", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(prompt)
    except OSError:
        return []
    try:
        # Agent 配置（type/provider/model）——留空不传 flag（继承 Agent 默认），
        # 显式配置无效（402/模型不存在）时回退为不传。复用 agent.py 与 AgentRunner
        # 同一实现，防两处漂移。
        from paper_review.agent import (
            AgentConfig,
            build_command,
            build_command_without_model,
            is_model_config_error,
        )

        agent_cfg = AgentConfig.from_env(os.environ)
        pi_cmd = build_command(agent_cfg, pi_binary, prompt_file)
        proc = subprocess.run(  # noqa: S603 — pi_binary is user-configurable (same as AgentRunner)
            pi_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if (
            proc.returncode != 0
            and agent_cfg.has_explicit_model()
            and is_model_config_error(proc.stderr or "")
        ):
            print(
                f"  ⚠ 显式 provider/model 配置无效（exit {proc.returncode}）——回退为 Agent 默认重试"
            )
            pi_cmd = build_command_without_model(agent_cfg, pi_binary, prompt_file)
            proc = subprocess.run(  # noqa: S603
                pi_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        if proc.returncode != 0:
            # 失败原因记入 paper-review.log（词表兜底继续，但可观测定位）
            stderr_tail = (proc.stderr or "").strip()[-200:] or f"exit {proc.returncode}"
            logger.warning("04: LLM 抽取失败（%s），词表兜底", stderr_tail)
            return []
        return _parse_feature_json(proc.stdout or "")
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("04: LLM 抽取超时/异常（%s），词表兜底", e)
        return []
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass


def main():
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")
    output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")
    intermediates_dir = os.environ.get("PIPELINE_INTERMEDIATES", ".")

    manifest_path = Path(output_dir) / "subject-manifest.json"
    subjects: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            subjects = manifest.get("subjects", [])
        except (json.JSONDecodeError, OSError):
            print(f"  ⚠ 无法读取 manifest: {manifest_path}")

    from paper_review.progress import load_existing_step_products, report_batch_progress
    from paper_review.search.store import Store

    # 匹配词表：自更新标签库优先，冷启动回退种子词表
    store_dir = os.environ.get("PIPELINE_INDEX_STORE_DIR", "./index")
    tag_library = _load_tag_library(store_dir)
    using_seed = not tag_library
    if using_seed:
        tag_library = _SEED_KEYWORDS
    print(f"04-extract-features: 词表={'种子' if using_seed else '标签库'}({len(tag_library)} 词)")

    # subject name → paper_id（02-auto-index 产物，用于写回 papers.features）
    subject_paper_ids = _load_subject_paper_ids(intermediates_dir)
    pi_binary = os.environ.get("PIPELINE_PI_BINARY", "pi")

    # 打开 Store（写回 papers.features，立即写使同批后续 Subject 可检索到）
    db_path = Path(store_dir) / "index.sqlite"
    store = None
    if db_path.exists():
        try:
            store = Store(str(db_path))
            store.load_for_search()
        except Exception as e:
            print(f"  ⚠ 打开索引失败（{e}），跳过 features 写回")
            store = None

    subject_count = 0
    features_written = 0
    # T4: Resume 断点续做——跳过已有 per-subject 产物的篇（不重复调 LLM）
    resume_skip = os.environ.get("PIPELINE_RESUME_SKIP_EXISTING") == "1"
    existing: set[str] = set()
    if resume_skip:
        existing = set(
            load_existing_step_products(subjects, intermediates_dir, "04-extract-features")
        )
        if existing:
            print(f"04-extract-features: 续做复用 {len(existing)} 篇已提取产物")
    reused = 0
    total = len(subjects)
    for i, subj in enumerate(subjects, 1):
        name = subj["name"]
        if name in existing:
            reused += 1
            report_batch_progress(i, total, name, reused=reused)
            continue
        pdf_path = Path(subj["pdf_path"])

        # 单篇处理（提取→词表→LLM→写回→产物），失败隔离 + 耗时记入 paper-review.log
        written, llm_count, feature_count = _process_subject(
            name=name,
            pdf_path=pdf_path,
            subject_paper_ids=subject_paper_ids,
            tag_library=tag_library,
            store=store,
            intermediates_dir=intermediates_dir,
            pi_binary=pi_binary,
        )
        features_written += written
        subject_count += 1
        logger.info(
            "04 [%d/%d] %s: 完成 (llm=%d, features=%d)", i, total, name, llm_count, feature_count
        )
        report_batch_progress(i, total, name, reused=reused)

    # 统计 L3 覆盖率（索引中 features 非空的 paper 比例，ADR 0015 哨兵信号）
    l3_total = 0
    l3_covered = 0
    if store is not None:
        try:
            rows = store.db.execute("SELECT features FROM papers").fetchall()
            l3_total = len(rows)
            l3_covered = sum(1 for (f,) in rows if f and f != "[]")
        except Exception as e:
            print(f"  ⚠ 统计 L3 覆盖率失败: {e}")
        store.close()

    l3_coverage = l3_covered / l3_total if l3_total else 0.0
    output = {
        "step": "04-extract-features",
        "status": "ok",
        "error": None,
        "data": {
            "subject_count": total,  # 总篇数（含复用的 skipped 篇，与 05 语义一致）
            "reused_count": reused,
            "features_written": features_written,
            "l3_coverage": l3_coverage,
            "l3_covered": l3_covered,
            "l3_total": l3_total,
        },
    }
    _write_json(Path(step_dir) / "output.json", output)

    print(f"04-extract-features: {subject_count} subject(s), {features_written} features written")


if __name__ == "__main__":
    main()
