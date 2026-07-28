# Architecture Refactor TODOs

## 1. 包重命名 paper_rag → paper_review (Strong)

- [x] 1.1 重命名 src/paper_rag/ → src/paper_review/
- [x] 1.2 全局替换所有 import 路径
- [x] 1.3 重命名环境变量 PAPER_RAG_*→ PAPER_REVIEW_*
- [x] 1.4 重命名日志文件 paper-rag.log → paper-review.log
- [x] 1.5 更新 pyproject.toml: name → paper-review
- [x] 1.6 更新所有测试中的 import
- [x] 1.7 运行全量测试确认

## 2. 文档更新 (Strong)

- [x] 2.1 SPEC.md 标题改为"论文评审流水线"，补管线架构
- [x] 2.2 CONTEXT.md 去除 paper-rag 引用
- [x] 2.3 更新 pyproject.toml description

## 3. Store 拆分 (Worth exploring)

- [x] 3.1 从 store.py 拆出 search_types.py（Paper, Chunk, DocVector 等数据类）
- [~] 3.2 (deferred: ~200 lines, tightly coupled to Store state) 从 store.py 拆出 vector_index.py（FAISS 操作）
- [x] 3.3 Store 保留 SQLite CRUD + 元数据管理
- [x] 3.4 更新所有调用方 import
- [x] 3.5 运行全量测试确认

## 4. 检索子包化 search/ (Worth exploring)

- [x] 4.1 创建 src/paper_review/search/ 子包
- [x] 4.2 移入 store, retriever, reranker, embedder, chunker, indexer, models
- [x] 4.3 __init__.py 暴露 hybrid_search() 和 build_index() 深接口
- [x] 4.4 更新所有调用方 import
- [x] 4.5 运行全量测试确认

## 5. CLI 瘦身 (Worth exploring)

- [x] 5.1 _open_store + _resolve_db_path → search/store.py 公共函数
- [x] 5.2 _maybe_show_first_use_hint → cli_ux.py
- [x] 5.3 运行全量测试确认

## 6. Pipeline 脚本边界 (Speculative)

- [x] 6.1 在 CONTEXT.md 标注脚本属于 user data 非 library
- [~] 6.2 (deferred: no immediate need) 评估是否需要 builtin_steps/ 目录
