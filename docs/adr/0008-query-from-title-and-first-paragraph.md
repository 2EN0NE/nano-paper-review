# Query 由「标题 + 正文首段」生成，不引入 jieba 关键词提取

**Context**: 检索的 query 原由 Subject 文件名生成（几乎无信息）；`02-extract-keywords.py` 的 docstring 声称使用 jieba TF-IDF，实际是硬编码 33 词表 + 子串匹配，且从检索结果反推关键词（顺序颠倒）。

**Decision**: query 改为「title_hint + 正文首段（截断约 500 字）」，同一个 query 同时喂给 BM25 与向量两条腿；不引入 jieba 依赖。`02-extract-keywords` 前移为「从 Subject 首段提取关键词」，仅作为给评审 Agent 的辅助参考信号，不再是检索输入。

**Why**: bge 语义 embedding 对自然语言 query 效果好，标题 + 首段语义密度远高于文件名；BM25 侧 CJK 已用「汉字间插空格」分词，长 query 可直接匹配；引入 jieba 会新增依赖，且硬编码词表对非 AI 论文几乎无命中。

## Considered Options

- **title_hint + jieba TF-IDF 全文关键词**: 需引入 jieba 依赖，且对非 AI 领域的关键词抽取质量无保证。放弃。
- **仅 title_hint**: 依赖文件名解析质量，对 arXiv 等弱命名文件退化为文件名。放弃。
