# nano-paper-review — Agent 开发指南

> **维护原则**：本文档仅保留项目入口必须知晓的要点。每完成一个重要任务，若产生新的设计知识，按此分级沉淀——
>
> - **一级**（这里）：架构全景、关键设计决策摘要、规范约定
> - **二级**：`CONTEXT.md`（领域词汇）、`SPEC.md`（需求规格）
> - **三级**：`docs/*.md`（模块细节、API 参考、部署步骤、ADR）

## 首先阅读

在开始任何开发工作前，按顺序阅读：

1. **[CONTEXT.md](./CONTEXT.md)** — 领域词汇表。**Subject**、**Reference**、**Review Phase**、**Intermediates** 等核心术语的唯一定义来源。涉及管线概念时必须对齐此文件。
2. **[SPEC.md](./SPEC.md)** — 需求与设计决策规格。检索系统的完整用户故事、技术选型理由（CJK 分词策略、后过滤 vs 前过滤等）、接口契约。
3. **`docs/SPEC-PIPELINE.md`** — 管线编制的需求规格。管线的用户故事、Phase 执行模型、Step 形态、模板变量系统、重试策略。

## 项目目的

离线技术论文评审工具。核心是**评审流水线**（批量格式归一化 → 逐篇 Agent 评审 → 批量化持久化），依赖本地混合检索引擎（BM25 + FAISS + Cross-Encoder 精排）驱动相似文章匹配。

## 数据目录（.paper-review/）

所有持久化数据集中在 data directory 中，解析优先级：

```
1. --data-dir <path>  显式指定（优先级最高）
2. ./.paper-review/   当前目录下**已初始化**（含 pipelines/，即运行过 init）时自动使用；
   存在但未初始化的空目录（如仅 logs/ 或 install.sh --offline 只写入的 config.yaml）
   视为残留，回退用户级
3. ~/.paper-review/   兜底（自动创建）
```

### 目录结构

```
.paper-review/                  # 或 ~/.paper-review/
├── config.yaml                 # CLI 配置文件（自动搜索）
├── .scaffold-manifest          # 脚手架版本 + 文件清单（版本检测/孤儿清理）
├── index/
│   ├── index.sqlite            # SQLite 数据库（FTS5 BM25 + 元数据）
│   ├── chunks.index            # Chunk 级 FAISS 向量索引
│   └── chunks_id_map.json      # FAISS ID → chunk_id 映射
├── pdfs/                       # PDF 源文件
├── output/                     # 管线输出
│   ├── intermediates/          #  中间产物
│   │   ├── {subject}/{step}/output.json
│   │   ├── pre/{step}/output.json
│   │   └── post/{step}/output.json
│   └── reports/{subject}/      #  最终报告
└── logs/
    └── paper-review.log           # 运行时日志
```

### 关键路径对照

| 内容 | 位置 |
|---|---|
| SQLite + FAISS 索引 | `{data_dir}/index/` |
| PDF 源文件 | `{data_dir}/pdfs/` |
| 管线中间产物 | `{data_dir}/output/intermediates/` |
| 最终报告 | `{data_dir}/output/reports/` |
| 运行时日志 | `{data_dir}/logs/` |
| ONNX 模型缓存 | `~/.cache/paper-review/models/`（不受 data_dir 影响） |
| CLI 命令行 | `--data-dir` 或 `PAPER_REVIEW_DATA_DIR` 环境变量 |

> 模型缓存独立于 data_dir 的原因：每个 ONNX 模型约 400MB-1GB，跨项目共享免重复下载。

### config.yaml

配置文件搜索顺序：`{data_dir}/config.yaml` > `cwd/config.yaml`。
所有路径字段 (`index_dir`, `pdf_dir`) 留空即自动从 `{data_dir}` 推导。
显式设置的值优先于自动推导。

```bash
# 手动创建项目级数据目录
mkdir .paper-review

# 显式指定
paper-review --data-dir /custom/data status

# 环境变量
PAPER_REVIEW_DATA_DIR=/custom/data paper-review index --pdf-dir ~/papers
```

## 架构速览

```
src/paper_review/
├── orchestrator.py     # 评审管线执行引擎 ← 项目核心
├── template_engine.py  # 模板变量替换 + Agent 前缀生成
├── store.py            # SQLite（FTS5 BM25）+ FAISS 持久化
├── extractor.py        # PDF 提取（PyMuPDF）+ 文件名元数据解析
├── chunker.py          # 512 字分块，overlap 128，参考文献截断
├── indexer.py          # build_index：分块 → embedding → chunk 向量
├── models.py           # bge-small-zh-v1.5 embedding 模型管理
├── retriever.py        # BM25 + Vector → RRF → (可选) Cross-Encoder 精排
├── embedder.py          # ONNX Runtime 嵌入引擎（CPU-only）
├── reranker.py         # ONNX Runtime 精排（CPU-only）
├── server.py           # Flask HTTP API
├── config.py           # Pydantic 配置加载
├── scaffold.py         # Scaffold 版本检测 + manifest + 孤儿清理
└── cli.py              # paper-review CLI（Typer）

src/paper_review/templates/   # Scaffold Template —— init 生成脚本的唯一权威内容源
├── config.yaml           # 默认 config.yaml
├── pipeline.yaml         # 默认管线编排定义
├── pre-review/           # .py 批量执行
├── review-pipeline/      # .py / .md 逐篇 Agent 步骤
└── post-review/          # .py 批量执行
```

测试 mirrors src —— `tests/test_store.py`、`tests/test_orchestrator.py` 等。

## 规范约定

- **`Store` 是唯一持久化入口**：所有索引操作（add/remove/rebuild）通过 Store，不直接操作 SQLite。
- **配置读取**：`config.py` 的 Pydantic 模型；默认值在 `store.py` 顶层常量。
- **向量序列化**：`np.asarray(vec, dtype=np.float32).tobytes()` 写入 BLOB（本机字节序）。
- **CLI**：Typer 框架，`paper-review` 统一入口。新增子命令时，docstring 即为 `--help` 文案，必须写清用法和选项含义。
- **CLI 设计红线**：见 SPEC.md § CLI 命令设计原则。核心约束：
  - `init` 做开箱即用的引导，生成完整脚手架确保用户能直接体验
  - `config` 是模型选择 / 设置管理的唯一入口
  - 模型发现逻辑统一在 `model_discovery` 模块，供 `config` 和 `install.sh` 共用
  - 不新增与现有命令职责重叠的命令——优先扩展现有命令参数

## 关键设计决策

### CPU‑only 优先（无 GPU / 无 CUDA）

本项目从诞生之初就面向 **2C/4G 无 GPU 的 Linux 机器**。所有模型推理使用 ONNX Runtime (CPU)，**零 PyTorch / CUDA 依赖**。embedding 和 reranker 模型需在开发机上通过 ``scripts/export_onnx.py``（需要 torch，只需执行一次）导出为 ONNX 格式，生产环境仅安装 ``onnxruntime`` + ``tokenizers``（~30MB 依赖，对比 PyTorch CUDA 的 ~2GB）。

详细讨论见 `SPEC.md`，此处仅列要点：

- **CJK 分词**：索引/查询时在汉字间插入空格，FTS5 unicode61 按空格分 token。
- **单一 chunk 级 FAISS 索引**：`chunks.index`，IndexFlatIP + L2 归一化 = 余弦相似度；文档级向量（`papers.index` / `doc_vectors`）已退役（ADR 0006）。
- **检索后过滤**：全库搜索 → RRF 融合 → 按 pool 过滤结果，保证不遗漏跨池匹配。
- **内容去重**：SHA-256 哈希 → content_dedup 表。
- **Embedding 指纹**：写入 `embed_fingerprint`，格式 `model/dim=N`；加载时对比，模型/维度变更则 warn 提示重建索引（旧权重后缀视为兼容）。
- **Agent 步骤**：通过 `subprocess.run(["pi", "-m", prompt])` 调用 pi。理由见 `docs/adr/0001-subprocess-pi-agent-steps.md`。
- **管线执行模型**：Pre/Post 批量模式，Review 逐篇模式。Step 排序优先级：pipeline.yaml 显式声明 > 文件名前缀 > OS 排序。
- **脚手架版本检测**：独立 `SCAFFOLD_VERSION`（当前 0.1.0）+ `{data_dir}/.scaffold-manifest` 清单。`review`/`init`/`status` 检测 Scaffold Drift（模板升级后用户侧副本未同步），`init --reset` 备份后清理孤儿文件。见 `docs/adr/0012-scaffold-version-detection.md`。

## 检索管道

```
query → BM25(FTS5, chunk级)  ┐
      → FAISS(chunk级向量)    ┘→ chunk 级 RRF 融合(k=60)
      → 聚合到论文（每篇 ≤3 chunk，总预算 20）→ 排除 content_hash 自身
      → (可选) Cross-Encoder 精排 chunk
      → 分池截断（history ≤5 / pending ≤3）→ 组装 SearchResult
```

`pool_filter` 在召回后作用（后过滤）；无精排时综合分 = RRF 归一化（ADR 0009）。

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

## 测试体系

### 三级分层

| 层级 | 目录 | 定位 | 运行方式 | 是否用 mock |
|------|------|------|----------|-------------|
| 单元测试 | `tests/test_*.py` | 纯 Python 函数/类级别的独立逻辑验证 | `uv run pytest` | 允许 mock 第三方依赖（onnxruntime、tokenizers 等） |
| 模型集成测试 | `tests/test_model_integration.py` | 双路径：有模型时真跑 ONNX 推理，无模型时 mock | `uv run pytest tests/test_model_integration.py` | mock 当模型不可用时；真实推理当模型可用时 |
| E2E 测试 | `tests/e2e/` | **以独立空间的 CLI 命令执行**，验证全链路行为 | `uv run pytest tests/e2e/ -v` | **禁止 mock**：必须通过 `subprocess.run([paper-review, ...])` 在隔离的 `--data-dir` 中执行 |

### E2E 测试的核心约束（红线）

E2E 测试是集成测试的唯一权威标准：

1. **CLI 命令独立空间执行**：每个测试在 `tmp_path` 中创建完整的数据目录，通过 `--data-dir` 隔离，不依赖外部文件或缓存。
2. **禁止 mock**：E2E 测试不能 mock 任何内部函数。唯一允许的 mock 是外部工具（如 `pandoc`、`pi` 的 mock 二进制）。
3. **验证产物**：检查管线产物文件（`output.json`、manifest、Excel、report）的存在性和内容正确性，不只检查 `returncode`。
4. **覆盖关键路径**：必须覆盖 Pre→Review→Post 完整链路、边界情况（空输入、去重、格式不支持）和特性开关（单篇 vs 多篇 Excel）。
5. **隔离性**：测试间互不依赖，每个测试独立创建 `tmp_path` 隔离。
6. **默认配置值可运行性**：任何影响外部工具调用（如 pi 子进程参数）的模块级默认常量必须有 E2E 测试。测试必须从源码动态导入常量值（`from module import _CONSTANT`），而非硬编码预期值——当常量被修改时，测试自动适用新值，同时验证新值不会导致运行时错误。见 `tests/e2e/test_pipeline_integration.py::TestDefaultConfigValidity` 示例。

### 进度卡 TUI 渲染测试（ANSI 残影回归）

进度卡（`progress.py`）在 stderr 上做 ANSI 原地重绘，**无法从原始字节流判断残影**——字节流里全是重复的盒子帧，看不出最终屏幕状态。必须重放为屏幕状态再断言：

1. **真实 TTY**：`pty.openpty()` 让 CLI 的 stdout+stderr 接同一 slave fd（模拟真实终端布局，stdout/stderr 同屏互相干扰的场景才暴露）。
2. **终端模拟器重放**：极简 VT100 模拟器（`tests/e2e/test_progress_tui.py` 内联 `Term` 类）把字节流重放为行缓冲，断言目标永远是**最终屏幕**，不是字节流。
3. **屏幕级断言**：完整盒子（`^┌─+┐$` / `^└─+┘$`）恰好 1 个、盒高固定 6、盒内无步骤输出混入、stdout 在 `finish()` 后恢复。
4. **已知坑**：
   - 勿用 `startswith("┌")` 找盒子——CLI 树形输出（`_build_cli_tree`）的 `└── POST` 会误匹配 `└`；必须用含结尾字符的完整边框特征。
   - PTY slave 默认 ONLCR（`\n`→`\r\n`），模拟器对 `\r`/`\n` 分别处理。
5. **回归价值验证**：本组测试抓过真实残影——临时禁用 `progress._mute_stdout()` 后立即失败（屏幕出现 4 处旧盒子顶框残留）。

进度卡显示期间的终端保护规则（`progress.py`）：

- **stderr 日志**：`_mute_console_logging()` 把 root 与 `paper_review` 的 stderr handler 提到 ERROR（logging.yaml 中共享同一 handler 实例，子 logger 一并静音）。
- **stdout 输出**：`_mute_stdout()` 把 `sys.stdout` 重定向到 devnull——`.py` 步骤经 `runpy` 在主进程内执行，其 `print()` 写 stdout，TTY 模式下会推动终端滚动、把进度盒往下推，固定行数的 ANSI 上移量错位 → 盒子上部残留旧帧。非 TTY 模式不静音。

### 测试数据

- **确定性 PDF**：`_make_pdf()` 生成最小有效 PDF（含文本内容的 PDF-1.4）
- **确定性 docx**：`_make_docx()` 生成最小 OOXML 文档
- **不测试**：FAISS 和 sentence-transformers 的第三方行为；HTTP 路由单独集成测试

### 运行测试

```bash
# 单元 + 模型集成测试（不需 onnxruntime，mock 兜底）
uv run pytest tests/ -q -m "not integration and not e2e_slow"

# 真模型集成测试（需要 onnxruntime + 已下载的模型）
uv run pytest tests/test_model_integration.py -q

# E2E 测试（需要 paper-review 已安装）
uv run pytest tests/e2e/ -v -m "e2e and not e2e_slow"

# 全量（与 CI / pre-push hook 一致）
make test-unit && make test-integration && make test-e2e
```

前置条件：`uv pip install -e .[dev]`（或 `make install`）。本地命令与 CI、
`.githooks/pre-push`、`Makefile` 完全一致（同一 `uv run pytest` + 同一 marker）。

## 安装与依赖管理

### 开发安装

```bash
# 安装运行时 + 开发依赖（pytest, ruff, huggingface-hub）
pip install -e .[dev]

# 仅运行时（不含开发工具）
pip install -e .
```

`[dev]` extras 定义在 `pyproject.toml` 的 `[project.optional-dependencies] dev` 中。
CI 中也使用 `pip install -e .[dev]` 方式安装，不依赖 `requirements.lock`（已被移除）。

### 日志系统

日志通过 `logging_config.py` 集中管理。初始化逻辑：

- **全局初始化**：`setup_logging()` 在 CLI 的 `_main_callback` 中调用，所有命令共享。
  `--log-level` 和 `--log-dir` 为全局选项（不是 `review` 子命令专属）。
- **输出位置**（双通道）：
  - console → **stderr**（DEBUG，brief 格式无时间戳）
  - file → **`{data_dir}/logs/paper-review.log`**（INFO，standard 格式带时间戳，
    TimedRotatingFileHandler 每日轮转保留 14 天）
  - 注意：CLI 总是把 log_dir 覆盖为 `{data_dir}/logs`（除非用户传 `--log-dir`）；
    logging.yaml 里的 filename 只是绕过 CLI 直接调用 `setup_logging()`（如测试）时的兜底占位
- **配置来源**（优先级递增）：
  1. `{data_dir}/logging.yaml` 或 `cwd/logging.yaml`
  2. 环境变量 `PAPER_REVIEW_LOG_LEVEL`、`PAPER_REVIEW_LOG_DIR`
  3. 内置默认：console (DEBUG) + file (INFO, 14 天轮转)
  4. CLI 显式参数 `--log-level` / `--log-dir`（最高）
- **logger 命名空间**：通过 `get_logger(__name__)` 或 `logging.getLogger(__name__)`
  获取，统一归入 `paper_review.*`。logging.yaml 中显式配置：
  `paper_review` (DEBUG)、`paper_review.orchestrator` (DEBUG)、`paper_review.retriever` (INFO)；
  root logger 仅 console (WARNING)。
- **Store.log()**：写入 `ops_log` 内存列表，不经过 logger 系统、不落盘，
  仅测试断言消费（如 DEDUP、FINGERPRINT MISMATCH）。关键操作同时使用
  `logger.info/warning` 和 `self.log()`。
- **管线步骤日志边界**：`.py` 步骤经 `runpy.run_path()` 进程内执行，logger 名不是
  `paper_review.*`，默认只走 root→stderr（WARNING+），**不会**进 paper-review.log；
  例外：Pre 模板步骤 04/05 显式用 `logging.getLogger("paper_review.pre")` 使逐篇
  耗时/失败日志进文件（runpy 下 `__name__` 是 `__main__`，默认 logger 不挂 FileHandler）；
  `.md` Agent 步骤经 subprocess 调用 pi，stdout/stderr 被捕获记入步骤中间产物。

### 测试运行

```bash
# 全部测试
uv run pytest tests/ -v

# 分层运行（与 CI / Makefile / pre-push hook 一致）
uv run pytest tests/ -q -m "not integration and not e2e_slow"   # 单元测试
uv run pytest tests/ -q -m "integration"                        # 集成测试
uv run pytest tests/e2e/ -v -m "e2e and not e2e_slow"           # E2E 测试
```
