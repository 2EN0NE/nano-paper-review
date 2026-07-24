你正在对一篇论文进行创新性评审。请仔细阅读 Subject 全文和检索到的参考文献，从以下几个方面评分：

## 评审维度

### 1. 方法创新性 (0-10)

- 是否提出了新的算法、模型或理论框架？
- 与参考文献中已有方法相比，核心差异是什么？
- 创新点是否具体、可验证？

### 2. 技术可行性 (0-10)

- 方法是否有明确的实现路径？
- 依赖假设是否合理？
- 是否有理论分析或证明？

### 3. 实验完整性 (0-10)

- 实验设计是否覆盖了关键对比？
- 数据集是否具有代表性？
- 结果分析是否深入？

## 输出格式

请严格按照以下 JSON 格式输出：

```json
{
  "step": "03-novelty",
  "status": "ok",
  "error": null,
  "data": {
    "scores": {
      "novelty": 7.5,
      "feasibility": 6.0,
      "experiments": 8.0
    },
    "summary": "简短总结（2-3句）",
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["不足1", "不足2"],
    "overall_score": 7.2
  }
}
```

## 上下文信息

- 论文名称: `{subject.name}`
- 检索到的参考文献: {intermediates.02-extract-keywords.data.keywords}
- 参考文献详情: {intermediates.01-search.data.references}
