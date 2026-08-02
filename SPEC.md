# 规范：论文评审流水线（paper-review）

## Problem Statement

用户在一台 **2019 年 Debian Linux** 机器上（2CPU/4GB RAM，无 GPU，离线环境）需要建立一个**本地论文评审工具**，支持：

1. 从 PDF 目录自动解析、索引中文技术论文
2. 通过 HTTP API 和 CLI 两种接口检索与待评议论文相似的已入库论文
3. 检索结果支持文档级（论文→论文）和片段级（文本→相关片段）两种粒度
4. 检索作为上层「论文自动评审流水线」的关键子模块，上层通过 LLM（内网 API）预处理待评议论文（生成摘要/关键词），检索系统负责本地技术索引与混合检索

用户已通过 grilling 会话确定了完整的技术决策树，原型已验证核心管道可运行。

## Solution

一个纯 Python 的离线论文检索系统，关键技术栈：

- **存储**：SQLite（标准库自带，零额外依赖），FTS5 做 BM25 全文检索
- **向量索引**：FAISS CPU（IndexFlatIP，内积相似度），两套独立索引（文档级 + chunk 级）
- **Embedding 模型**：`BAAI/bge-small-zh-v1.5`（~100MB，512 维）
- **Reranker 模型**：`BAAI/bge-reranker-v2-m3`（~1.1GB fp16 加载）
- **PDF 提取**：PyMuPDF（fitz），针对中文单栏论文，简化的过滤策略
- **分层架构**：Extractor → Chunker → Indexer (BM25 + FAISS) → Retriever (RRF + Reranker) → CLI/HTTP

## User Stories

1. 作为一名评审人员，我希望能把历史论文 PDF 放入一个文件夹并运行一条命令，使得这些论文被自动解析、分块、建立索引，供后续检索使用

2. 作为一名评审人员，我希望能通过 CLI 输入一段文本（或一篇待评论文的摘要），快速找到历史库中最相似的 N 篇论文及其相关信息

3. 作为一名评审人员，我希望能通过 HTTP API（localhost 端口）发送检索请求，使得外部服务（如评审流水线的 agent 步骤）能调用检索功能

4. 作为一名评审人员，我希望能看到检索结果的完整信息，包括相似度分数、论文标题、作者、年份、匹配内容片段、来源池（history/pending），以便直接用于论文比对

5. 作为一名评审人员，我希望能按池（history 池、pending 池）过滤检索结果，以便对不同批次论文做定向分析

6. 作为一名评审人员，我希望能支持 chunk 级检索（输入一段话，匹配相似片段），以便在精读阶段精确定位相关论述

7. 作为一名评审人员，我希望能查看当前索引的状态（论文数、chunk 数、各池分布、向量/BM25 条目数），以便了解当前知识库的覆盖情况

8. 作为一名运维人员，我希望能将系统一键打包并离线部署到无网的老旧 Linux 机器上（Python 3.12 自编译），使得运行过程中不需要任何网络访问

9. 作为一名运维人员，我希望能通过配置文件（YAML）控制检索行为（分块参数、加权策略、Top-K 数量），使得在不改代码的情况下调整检索策略

10. 作为一名评审人员，当模型或加权配置变更后，我希望能收到「向量与当前配置不兼容」的警告，并能一键重新嵌入文档向量

11. 作为一名评审人员，我希望能将同一篇论文的多个副本（文件名不同、内容相同）去重，避免索引膨胀和检索结果重复

12. 作为一名评审人员，我希望能从文件名中自动解析出论文标题、作者、年份等元数据，对格式规整的文件名（如 `01.提案表-基于深度学习-张三.pdf`）能精准提取

13. 作为一名开发者，我希望能使用纯 Python 标准库 + pip 可安装的依赖实现整个系统，使得依赖冲突最小化

## CLI 命令设计原则

paper-review 的命令遵循以下设计原则：

### 命令职责边界

| 命令 | 定位 | 职责 |
|------|------|------|
| `init` | 开箱引导 | 一键生成完整可用的项目脚手架：config.yaml + pipeline.yaml + 所有默认管线步骤文件（pre-review/、review-pipeline/、post-review/）。生成后用户可直接运行 `paper-review review ./paper.pdf` 体验完整流程。引导完成后提示 `config` 命令和配置文件路径。 |
| `config` | 完整设置 | 模型选择（本地发现 + 在线 3 档推荐）、配置编辑。`init` 之后的下一步。 |
| `review` | 核心工作 | 执行评审流水线。单 PDF 或目录批量。 |
| `index` | 检索子系统 | 建历史论文索引。 |
| `search` | 检索子系统 | 混合检索。 |
| `status` | 检索子系统 | 查看索引状态。 |
| `serve` | 检索子系统 | 启动 HTTP API。 |

### 设计红线

1. **不新增与现有命令职责重叠的命令**。优先扩展现有命令的参数而非新建。
2. **`init` 做开箱即用的引导**。`init` 应生成完整的项目脚手架（含所有默认步骤文件），确保用户按 README 步骤即可直接体验评审管线。`init` 结束时应提示 `config` 命令和配置文件路径。
3. **模型选择逻辑统一在 `model_discovery` 模块**，供 `config` 命令和 `install.sh` 共用。
4. **install.sh 的模型流程与 `config` 一致**：先扫描本地 → 有则列出让选，没有则 3 档推荐。

## Implementation Decisions

### 1. 包结构与模块边界

```
paper-review/
├── pyproject.toml
├── src/paper_review/
│   ├── cli.py              # CLI 入口（Typer）
│   ├── server.py           # HTTP API 入口（Flask）
│   ├── config.py           # 配置加载（Pydantic）
│   ├── extractor.py        # PDF 文本提取（PyMuPDF）
│   ├── logging_config.py   # 日志系统
│   │
│   ├── orchestrator.py     # 流水线执行引擎
│   ├── pipeline_models.py  # 管线数据模型 + Step 发现
│   ├── pipeline_steps.py   # .py / .md Step 执行器
│   ├── template_engine.py  # 模板变量替换 + Agent 前缀
│   │
│   ├── store.py            # SQLite 持久化层（含 schema、FTS5）
│   ├── retriever.py        # 检索管道（BM25 + FAISS + RRF + Reranker）
│   ├── reranker.py         # Cross-Encoder 精排封装
│   ├── chunker.py          # 分块（段落边界优先 + 滑动窗口）
│   ├── indexer.py          # 索引构建与增量管理
│   ├── embedder.py         # ONNX Runtime 嵌入引擎
│   └── models.py           # embedding / reranker 模型管理
├── tests/
└── data/
```

### 2. 存储层（store.py）

使用 Python 标准库 `sqlite3` 模块，单文件数据库，schema：

```sql
-- 论文元数据
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    filepath TEXT NOT NULL,
    filename TEXT NOT NULL,
    title_hint TEXT,
    year INTEGER,
    author_hint TEXT,
    arxiv_id TEXT,
    tags TEXT,            -- JSON 数组
    pool TEXT,            -- 'history' | 'pending'
    raw_text TEXT,
    pages INTEGER
);

-- 内容去重
CREATE TABLE content_dedup (
    sha256 TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL
);

-- Chunk 元数据
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    text TEXT NOT NULL,
    page_num INTEGER,
    seq INTEGER,
    position_weight REAL DEFAULT 1.0
);

-- FTS5 BM25 全文索引
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    paper_id UNINDEXED,
    text,
    tokenize='unicode61'
);

-- Chunk 向量（float32 BLOB）
CREATE TABLE chunk_vectors (
    chunk_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    dim INTEGER DEFAULT 512
);

-- 文档级 Mean Pooling 向量
CREATE TABLE doc_vectors (
    paper_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    dim INTEGER DEFAULT 512,
    weight_config TEXT
);

-- Embedding 模型指纹
CREATE TABLE embed_fingerprint (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

**CJK 分词策略**：FTS5 的 `unicode61` tokenizer 不能自动切分中文。采用与 QMD 相同的策略——在索引和查询时，对 CJK 字符之间插入空格，使每个汉字成为独立 token。查询时需要同规范化。BM25 分数不跨查询归一化（RRF 融合只取排名，不关心绝对值）。

### 3. 分块策略（chunker.py）

- Chunk 大小：512 字符（对应 bge-small-zh-v1.5 的 512 token 窗口）
- Overlap：128 字符
- 切分位置：优先在段落边界（`\n\n`）断开
- 参考文献过滤：检测中文「参考文献」标题后截断
- 位置权重：三段式加权（configurable）
  - Head（前 15% chunk）：`weight=5.0`
  - Body（中间 chunk）：`weight=2.0`
  - Tail（后 10% chunk）：`weight=4.0`

原型验证了加权 Mean Pooling 的行为：每个 chunk 编码后按位置权重加权平均得到论文级向量，可通过 `rebuild_doc_vectors()` 重新计算。

### 4. 文档级检索管道（retriever.py）

```
查询文本（或论文全文本）
    │
    ▼
┌──────────────────────────────┐
│ 1. BM25 (FTS5)              │  chunk 级检索
│    返回 chunk_id + BM25 分   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 2. BM25 max 聚合到论文      │  chunk_id → paper_id
│    paper_score = max(chunk)  │  取最高分 chunk
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 3. FAISS 文档级检索         │  查询向量 vs 论文级 Mean Pooling 向量
│    返回 paper_id + cos sim   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 4. RRF 融合                 │  k=60
│    score = Σ 1/(k+rank+1)    │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ 5. Cross-Encoder 精排       │  bge-reranker-v2-m3
│    对 Top-50 候选重排序      │  返回 Top-5
└──────────┬───────────────────┘
           │
           ▼
          返回结果（含元数据、匹配片段、分数）
```

**检索范围策略**：全库搜索后按 pool 过滤（B 方案）。即 BM25 + FAISS 在所有池中搜索，RRF 融合后，只返回指定池的结果。这样可以避免遗漏跨池潜在匹配。

### 5. Embedding 模型与部署

- **Embedding**：`BAAI/bge-small-zh-v1.5`（~100MB，512 维），使用 `sentence-transformers` 加载
- **Reranker**：`BAAI/bge-reranker-v2-m3`（~560M 参数），fp16 加载约 1.1GB，fp32 约 2.27GB
- **总内存预算**：embedding ~100MB + reranker ~1.1GB + FAISS + Python ≈ 2-2.5GB，在 4GB 限制内
- **离线部署方案**：有网环境下使用 `pip download --platform manylinux2014_x86_64` 打包 whl，用 `wget`/`huggingface-cli download` 下载模型权重，打包为 tarball，离线解压后用 `pip install --no-index` 安装
- **Python 版本**：3.12.13（用户自编译）

### 6. FAISS 索引

- 两套独立 IndexFlatIP（内积相似度，结合 L2 归一化等价于余弦相似度）
  - `papers.index`：文档级向量（N 条，N = 论文数）
  - `chunks.index`：chunk 级向量（~20N 条，N = 论文数× 平均 chunk 数）
- 论文量 < 1 万时，IndexFlatIP 够用；> 1 万时切换 IndexIVFFlat 或 HNSWFlat
- ID 映射：独立 `id_map.json` 文件记录 FAISS 索引位置 ↔ chunk_id/paper_id 的映射

### 7. 增量索引

- BM25（SQLite FTS5）支持原生增量 INSERT，无全量重建
- FAISS `add_with_ids` 支持增量写入
- 每次 `add_paper` 操作在事务中执行：元数据 + chunks + FTS + chunk向量 + 文档向量 + 内容哈希，全部成功才提交
- `remove_paper` 级联删除所有关联数据（外键 + 手动 FTS 删除）
- 内容去重通过 SHA-256 哈希检测：同内容论文仅存储元数据，共享向量

### 8. Embedding 指纹

每次存储文档向量时同时存储 `embed_fingerprint`，格式为 `"bge-small-zh-v1.5/dim=512/head=5.0_body=2.0_tail=4.0"`。启动加载时对比当前配置，不一致时在日志中输出警告，并提供 `rebuild_doc_vectors()` 方法触重新计算。

### 9. 文件名元数据提取

采用三段正则解析（优先级递减）：

1. `{序号}.{文档类型}-{标题}-{作者}.pdf` — 用户实际命名格式（支持的格式：`01.提案表-XXX-张三.pdf`）
2. `{年份}_{作者}_{标题}.pdf` — 通用格式
3. 纯正则兜底

解析是尽力而为的，不保证所有字段都能提取。未提取的字段留空，用户可通过 LLM 标签系统补充。

### 10. 检索结果格式

```json
{
  "results": [
    {
      "paper_id": "abc123def456",
      "filename": "01.提案表-基于深度学习-张三.pdf",
      "pool": "history",
      "score": 0.8934,
      "title_hint": "基于深度学习的方法研究",
      "year": 2023,
      "author_hint": "张三",
      "arxiv_id": "",
      "pages": 8,
      "match_chunk_snippet": "本文提出了一种新的深度学习融合方法...",
      "tags": []
    }
  ],
  "meta": {
    "query": "深度学习信用评估",
    "total_results": 5,
    "pool_filter": null,
    "took_ms": 1247
  }
}
```

### 11. 接口契约

**CLI**：

```bash
# 建索引
python -m paper_review.cli index --pdf-dir ./pdfs

# 搜索
python -m paper_review.cli search "深度学习信用评估"
python -m paper_review.cli search "深度学习信用评估" --pool history
python -m paper_review.cli search "图神经网络" --chunk-level   # chunk 级

# 服务模式
python -m paper_review.cli serve --port 8765

# 状态
python -m paper_review.cli status
```

**HTTP API**：

```
POST /search
{
  "query": "深度学习信用评估",
  "limit": 5,
  "pool_filter": "history",        # 可选
  "chunk_level": false,            # chunk 级检索
  "with_rerank": true,
  "intent": ""                      # 可选的 LLM 意图消歧
}
→ { "results": [...], "meta": {...} }

GET /status → { "papers": N, "chunks": N, ... }
```

## Testing Decisions

- **测试策略**：纯逻辑单元测试为主，Seam 放在 `Store` 和 `build_index` / `search` 函数
- **测试什么**：分块逻辑（段落边界、参考文献截断、重叠窗口）、文件名元数据提取、RRF 融合的排名正确性、加权 Mean Pooling 的数学正确性、FTS5 CJK 分词的命中率
- **不测试什么**：FAISS 和 embedding 模型的行为（这些是第三方库，假设正确）；HTTP 路由（单独集成测试）
- **测试数据**：使用原型中已有的确定性模拟数据（`_make_fake_content`），不需要真实 PDF
- **Seam 位置**：`Store.__init__(:memory:)` 提供纯内存数据库用于测试，无需真实文件系统

## Out of Scope

- LLM 摘要/关键词生成不属于本项目范围（上层评审流水线通过内网 API 完成）
- PDF OCR 后备方案（用户确认无双栏，表格可简化处理，原型阶段不做过度兼容）
- 多用户并发与鉴权（单机 localhost 服务，用户独占）
- 集群部署与分布式索引
- Web UI（只提供 HTTP API 和 CLI）
- 论文间的精细比对功能（检索系统只负责「找到相似论文」，比对是上层评审流水线的职责）

## Further Notes

- 原型项目位于 `prototype/` 目录，包含：
  - `prototype/logic.py` — 纯逻辑模块（含 Store、分块、Embedding 模拟、检索管道）
  - `prototype/tui.py` — 交互式 TUI，可通过按键驱动索引/搜索/状态查看
  - 运行方式：`python -m prototype.tui`
  - 原型已经验证了以下决策：SQLite FTS5 CJK 分词、文档+chunk 双 FAISS 索引、加权 Mean Pooling 重嵌入、池过滤、内容去重、Embedding 指纹检测
- 参考项目 QMD（`@tobilu/qmd`）的架构为：SQLite FTS5 + sqlite-vec + GGUF 模型 + RRF 融合 + 位置感知混合。本方案的 CJK 分词、事务化增量索引、内容哈希去重的设计灵感来自 QMD。
