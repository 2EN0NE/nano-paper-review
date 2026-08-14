"""
04-extract-keywords.py — 从 Subject 首段提取技术关键词（辅助信号）

关键词从 Subject 正文首段（约 500 字）提取，作为给评审 Agent 的辅助参考
信号，不再是检索输入（ADR 0008）。

结果写入 per-subject intermediates（intermediates/{subject}/04-extract-keywords/
output.json），供 Review Phase 评分步骤通过模板变量读取。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from paper_review.search.search_types import QUERY_FIRST_PARA_CHARS

# 简单中文技术关键词列表（免加载模型）
_TECH_KEYWORDS = [
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


def extract_keywords(text: str) -> list[str]:
    """从文本中提取技术关键词（子串匹配，去重保序）。"""
    found = [kw for kw in _TECH_KEYWORDS if kw in text]
    return list(dict.fromkeys(found))


def _write_json(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"  ✗ 写入 {path} 失败: {e}")


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

    subject_count = 0
    for subj in subjects:
        name = subj["name"]
        pdf_path = Path(subj["pdf_path"])

        try:
            raw_text = extract_pdf(str(pdf_path))
        except Exception:
            raw_text = ""

        first_para = raw_text[:QUERY_FIRST_PARA_CHARS]
        # 从正文首段提取，并补充 Subject 名称中的关键词
        keywords = extract_keywords(first_para + " " + name.replace("-", " "))

        _write_json(
            Path(intermediates_dir) / name / "04-extract-keywords" / "output.json",
            {
                "step": "04-extract-keywords",
                "status": "ok",
                "error": None,
                "data": {"keywords": keywords, "keyword_count": len(keywords)},
            },
        )
        subject_count += 1

    output = {
        "step": "04-extract-keywords",
        "status": "ok",
        "error": None,
        "data": {"subject_count": subject_count},
    }
    _write_json(Path(step_dir) / "output.json", output)

    print(f"04-extract-keywords: {subject_count} subject(s) processed")


if __name__ == "__main__":
    main()
