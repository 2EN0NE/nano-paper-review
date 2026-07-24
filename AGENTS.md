# nano-paper-review — Agent 开发指南

> **维护原则**：本文档仅保留项目入口必须知晓的要点。每完成一个重要任务，若产生新的设计知识，按此分级沉淀——
>
> - **一级**（这里）：架构全景、关键设计决策摘要、规范约定
> - **二级**：`CONTEXT.md`（领域词汇）、`SPEC.md`（需求规格）
> - **三级**：`docs/*.md`（模块细节、API 参考、部署步骤、ADR）

## 首先阅读

在开始任何开发工作前，按顺序阅读：

1. **[CONTEXT.md](./CONTEXT.md)** — 领域词汇表。**Subject**、**Reference**、**Review Phase**、**Intermediates** 等核心术语的唯一定义来源。涉及管线概念时必须对齐此文件。
2. **[SPEC.md](./SPEC.md)** — 需求与设计决策规格。检索系统的完整用户故事、技术选型理由（CJK 分词策略、加权 Mean Pooling、后过滤 vs 前过滤等）、接口契约。
3. **`docs/SPEC-PIPELINE.md`** — 管线编制的需求规格。管线的用户故事、Phase 执行模型、Step 形态、模板变量系统、重试策略。

## 项目目的

离线技术论文评审工具。核心是**评审流水线**（批量格式归一化 → 逐篇 Agent 评审 → 批量化持久化），依赖本地混合检索引擎（BM25 + FAISS + Cross-Encoder 精排）驱动相似文章匹配。

## 架构速览

```
src/paper_rag/
├── orchestrator.py     # 评审管线执行引擎 ← 项目核心
├── template_engine.py  # 模板变量替换 + Agent 前缀生成
├── store.py            # SQLite（FTS5 BM25）+ FAISS 持久化
├── extractor.py        # PDF 提取（PyMuPDF）+ 文件名元数据解析
├── chunker.py          # 512 字分块，overlap 128，参考文献截断
├── indexer.py          # build_index：分块 → embedding → Mean Pooling → FAISS
├── models.py           # bge-small-zh-v1.5 embedding 模型管理
├── retriever.py        # BM25 + Vector → RRF → (可选) Cross-Encoder 精排
├── reranker.py         # bge-reranker-v2-m3 精排封装
├── server.py           # Flask HTTP API
├── config.py           # Pydantic 配置加载
└── cli.py              # paper-review CLI（Typer）

pipeline/
├── pipeline.yaml       # 管线编排定义
├── pre-review/         # .py / .md 批量执行
├── review-pipeline/    # .py / .md 逐篇 Agent 步骤
└── post-review/        # .py / .md 批量执行
```

测试 mirrors src —— `tests/test_store.py`、`tests/test_orchestrator.py` 等。

## 规范约定

- **`Store` 是唯一持久化入口**：所有索引操作（add/remove/rebuild）通过 Store，不直接操作 SQLite。
- **配置读取**：`config.py` 的 Pydantic 模型；默认值在 `store.py` 顶层常量。
- **向量序列化**：`struct.pack("f" * dim, *vec)` 写入 BLOB。
- **CLI**：Typer 框架，`paper-review` 统一入口。新增子命令时，docstring 即为 `--help` 文案，必须写清用法和选项含义。

## 关键设计决策

详细讨论见 `SPEC.md`，此处仅列要点：

- **CJK 分词**：索引/查询时在汉字间插入空格，FTS5 unicode61 按空格分 token。
- **双 FAISS 索引**：`papers.index`（文档级）+ `chunks.index`（chunk 级），IndexFlatIP + L2 归一化 = 余弦相似度。
- **文档向量**：加权 Mean Pooling，按位置三段加权（head=5.0 / body=2.0 / tail=4.0，比例可配置）。
- **检索后过滤**：全库搜索 → RRF 融合 → 按 pool 过滤结果，保证不遗漏跨池匹配。
- **内容去重**：SHA-256 哈希 → content_dedup 表。
- **Embedding 指纹**：写入 `embed_fingerprint`，加载时对比，不一致则 warn + `rebuild_doc_vectors()`。
- **Agent 步骤**：通过 `subprocess.run(["pi", "-m", prompt])` 调用 pi。理由见 `docs/adr/0001-subprocess-pi-agent-steps.md`。
- **管线执行模型**：Pre/Post 批量模式，Review 逐篇模式。Step 排序优先级：pipeline.yaml 显式声明 > 文件名前缀 > OS 排序。

## 检索管道

```
query → BM25(FTS5, chunk级) → max聚合到论文分
      → FAISS(文档级向量)     → cosine similarity
      → RRF融合(k=60)        → Top-30候选
      → Cross-Encoder精排    → Top-5结果
```

`pool_filter` 在 RRF 后作用（后过滤）。

## 评审流水线

```
Pre Phase (batch) → Review Phase (per subject) → Post Phase (batch)
```

- **Step 形态**：`.py`（Python 直接执行）或 `.md`（pi agent 调用）
- **中间产物**：`{output_dir}/intermediates/{subject}/{step}/output.json`
- **模板变量**：`.md` 中 `{subject.name}`, `{intermediates.XX.data.YY}` 等，提交 Agent 前替换
- **Agent 前缀**：框架自动拼接前序步骤汇总 + 输出约束
- **错误策略**：pipeline.yaml 定义重试 + skip/abort
- **CLI**：`paper-review review <path>` 统一入口

详见 `CONTEXT.md`（术语）、`docs/PIPELINE.md`（设计）、`docs/SPEC-PIPELINE.md`（需求规格）。

## 测试策略

- **Seam**：`Store(":memory:")` 纯内存 SQLite；管线测试用临时目录 + mock `subprocess.run`
- **测试数据**：确定性纯文本模拟 PDF 内容
- **不测试**：FAISS 和 sentence-transformers 的第三方行为；HTTP 路由单独集成测试

前置条件：`PYTHONPATH=src pip install -e .`
