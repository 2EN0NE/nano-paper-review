"""
02-extract-keywords.py — 提取技术关键词
使用 jieba 分词和 TF-IDF 提取 Subject 的技术关键词
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.environ.get("PIPELINE_PIPELINE_DIR", "."))


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
    """从文本中提取技术关键词。"""
    found = [kw for kw in _TECH_KEYWORDS if kw in text]
    return list(dict.fromkeys(found))  # 去重保序


def main():
    subject = os.environ.get("PIPELINE_SUBJECT", "")
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")
    intermediates_dir = os.environ.get("PIPELINE_INTERMEDIATES", ".")

    # 读取前序步骤（search）的输出
    search_output = os.path.join(intermediates_dir, subject, "01-search", "output.json")
    references = []

    if os.path.exists(search_output):
        with open(search_output) as f:
            search_data = json.load(f)
        references = search_data.get("data", {}).get("references", [])

    # 从 references 提取关键词（模拟场景）
    all_titles = [r.get("title", "") for r in references]
    combined = " ".join(all_titles)
    keywords = extract_keywords(combined)

    # 补充 Subject 名称中的关键词
    subject_kws = extract_keywords(subject.replace("-", " "))
    keywords = list(dict.fromkeys(subject_kws + keywords))

    output = {
        "step": "02-extract-keywords",
        "status": "ok",
        "error": None,
        "data": {
            "keywords": keywords,
            "keyword_count": len(keywords),
            "references_count": len(references),
        },
    }

    os.makedirs(step_dir, exist_ok=True)
    with open(os.path.join(step_dir, "output.json"), "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
