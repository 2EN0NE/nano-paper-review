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
import os
import sqlite3
from pathlib import Path

from paper_review.search.search_types import QUERY_FIRST_PARA_CHARS

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
            "SELECT tags FROM papers WHERE tags IS NOT NULL AND tags != '[]'"
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


def _build_feature_prompt(text: str) -> str:
    """构建 LLM 抽取技术方法关键词的 prompt（要求只输出 JSON 数组）。

    粒度约束（ADR 0015）：只抽具体技术方法，不抽宽泛领域词；粒度控制
    前置到抽取端而非打分端加权。
    """
    return (
        "你是一名技术论文分析助手。请从下面的论文首段中抽取「技术方法」关键词。\n\n"
        "要求：\n"
        "1. 只抽取具体的技术方法/技术手段（如：向量化执行、MPP、列式存储、倒排索引），\n"
        "   不要抽取宽泛的领域词（如：数据库、分布式、机器学习）。\n"
        "2. 抽取 3~8 个，按重要性降序。\n"
        "3. 只输出一个 JSON 数组，不要任何其他文字。\n\n"
        f"论文首段：\n{text}\n\n"
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


def _llm_extract_features(text: str, pi_binary: str = "pi", timeout: int = 300) -> list[str]:
    """调 pi（LLM）从文本抽取技术方法关键词（LLM 主线，ADR 0015）。

    失败（超时/非零退出/无法解析）返回空列表，由词表兜底。
    """
    import subprocess
    import tempfile

    prompt = _build_feature_prompt(text)
    try:
        fd, prompt_file = tempfile.mkstemp(suffix=".md", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(prompt)
    except OSError:
        return []
    try:
        proc = subprocess.run(  # noqa: S603 — pi_binary is user-configurable (same as AgentRunner)
            [pi_binary, "--no-session", "-p", f"@{prompt_file}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return []
        return _parse_feature_json(proc.stdout or "")
    except (subprocess.TimeoutExpired, OSError):
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

    from paper_review.extractor import extract_pdf
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
    for subj in subjects:
        name = subj["name"]
        pdf_path = Path(subj["pdf_path"])

        try:
            raw_text = extract_pdf(str(pdf_path))
        except Exception:
            raw_text = ""

        first_para = raw_text[:QUERY_FIRST_PARA_CHARS]
        text_with_name = first_para + " " + name.replace("-", " ")
        # 词表兜底（确定性，保证已知词永不丢）
        keyword_features = extract_keywords(text_with_name, tag_library)
        # LLM 主线（发现新词 + 已知词）
        llm_features = _llm_extract_features(text_with_name, pi_binary=pi_binary)
        # 并集汇总
        features = merge_feature_sets(llm_features, keyword_features)

        # 写回 papers.features（立即写）
        paper_id = subject_paper_ids.get(name)
        if store is not None and paper_id and features:
            try:
                if store.update_features(paper_id, features):
                    features_written += 1
            except Exception as e:
                print(f"  ⚠ {name} features 写回失败: {e}")

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
        subject_count += 1

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
            "subject_count": subject_count,
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
