# 检索链路失效复盘 + 哨兵策略 + Bug A 根因

**Context**: 20260814 run（56 篇美团 2020 文章）检索链路整体失效——`04-batch-search` 因 FTS5 索引损坏静默失败（`on_failure: skip` 吞掉），56 篇评审的「历史参考/本批次参考」全空，评分退化为纯 LLM 打分；同时 keywords 恒空、标签写回 `tags_written=0`。事后发现测试全绿（76 passed），但无一能抓到这些故障。

## Bug A 根因（确凿，已最小复现）

`Store.add_paper` / `bulk_add_paper` 用 `INSERT OR REPLACE INTO papers` 写入元数据。`chunks` 表有 `REFERENCES papers(paper_id) ON DELETE CASCADE`，因此**重复索引同一 paper_id 时，REPLACE 会级联删除旧 chunks 行**；但 `chunks_fts` 是 FTS5 **external content 表**（`content='chunks'`, `content_rowid='rowid'`），不参与外键级联，残留指向已删 rowid 的孤儿条目。

最小复现（内存库）：add p1 → add p2 → 重复 add p1 后，`chunks=4` 而 `chunks_fts_docsize=6`（多 2 个孤儿 rowid），`bm25_search` 抛 `fts5: missing row 1 from content table 'main'.'chunks'`。生产库同构证据：`chunks` 3171 行 vs `chunks_fts_docsize` 3203 行，恰差 32 个孤儿（chunks rowid 1..32 空洞）。

**修复方向**: 在 `INSERT OR REPLACE INTO papers` **之前**，先显式删除该 paper_id 在 `chunks_fts` 里的旧条目（抽为 `Store._delete_paper_fts` 复用，`remove_paper` 同样改为调用它）。

**修复时发现的第二个坑（同根）**: FTS5 external content 表的 `'delete'` 命令**必须提供所有列的值且 indexed 列用与 INSERT 时一致的值**——仅传 rowid、或传 NULL、或 indexed `text` 列用原始未 normalize 的文本，都会破坏索引（报 `database disk image is malformed`）。`remove_paper` 原有逻辑只传 rowid，同样会损坏（测试未覆盖「删除后检索」而漏掉）。正确写法：`INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, paper_id, text) VALUES ('delete', ?, ?, ?, normalize_cjk_for_fts(text))`。

## 决策 1：止血策略（哨兵断言）

问题本质不是「某函数写错」而是「静默降级」——检索失败、标签丢失、关键词空转全被 `skip`/`try-except` 吞掉，用户拿到看似完整的 Excel。

- **abort 级（链路断裂）**: pre/post 每个 step 的 `output.json` 必须存在且 `status == "ok"`；缺失或 `status=error/skipped` 且非显式降级 → 中断管线。终结「04-batch-search 失败被 skip 吞掉」。
- **warn 级（结果空）**: history 池恒空、keywords 恒空、`tags_written==0`、`data.tags` 缺失 → 只标注不中断（可能是合法冷启动）。
- **力度**: 默认 abort + 显式降级开关 `--allow-degraded`（把「检索没生效也继续」的决定权还给用户）。
- **呈现（warn 级）**: 最终报告/Excel 加降级标注区块 + run 结束终端醒目汇总（双通道，保证「一定看见」）。

## 决策 2：检索价值验证方法

不重跑 pipeline，离线直跑 `hybrid_search`，锚点用 5 组已知强相关文档对（Flutter 2 篇 / GC 2 篇 / OLAP 2 篇 / 云原生 3 篇 / 搜索 4 篇）。判定：A 的 query 检索时，强相关 B 应进 top-3 且分数高于无关命中；否则判定检索有实质问题。**前置阻塞：FTS5 损坏导致检索无法运行，必须先修 Bug A 才能验证。**

## 验证结论（Bug A 修复 + FTS 重建后）

5 组锚点全部 **HIT**（强相关对进 top-3，检索有召回能力），但暴露 **Bug F：排序/分数失真**——

- **向量腿（召回）正确**：Doris 组，强相关 Kylin `vec=0.857`，无关文章 `vec=0.000`，语义区分清晰；
- **BM25 腿虚高**：无关文章因共享「美团/实践」等高频词，`bm25=13.9` 反超强相关的 `11.2`；
- **Reranker 无区分度**：bge-reranker-v2-m3 对中文业务文章 sigmoid 后分数挤在 0.93~0.94（强相关 0.9408 vs 无关 0.9396，差 0.0012），而 `combined_score` 优先取 rerank 分，导致最终排序被污染。

结论：检索「值得救」，但除 Bug A 外还需修 **Bug F（分数/排序）**——向量腿可信，RRF 融合 + rerank 精排反而是降噪后的失真。修复方向待定（候选：combined 改取向量分/加权、rerank 分数校准、换 reranker、BM25 高频词降权）。

## Why

- 先止血后闭环：修 keywords/history 闭环（Bug D/E）的前提是「检索结果有价值」，该前提尚未验证；盲目投入会让垃圾结果被写回标签库形成负反馈。
- abort 只对「链路没跑成」这一确定性坏：「跑成了但结果空」用 warn 暴露，避免误杀冷启动合法空结果。

## 测试盲区（为何 76 个测试全绿却漏掉）

- E2E 把 `02-auto-index` 换成 noop + `Store.add_paper` 单次建索引，从未跑真实 `bulk_add_paper` 的重复索引（REPLACE 级联删）路径。
- `test_archive_reports` 的 fixture 只设 `PIPELINE_OUTPUT_DIR` 且产物写到 `PIPELINE_OUTPUT_DIR/intermediates`，与模板读的位置碰巧一致，没复现 orchestrator 的 `OUTPUT_DIR`/`RESULT_DIR` 分离布局。
- fake pi 只输出纯 JSON/纯文本，没覆盖真实 pi 的「Markdown 摘要+打分表+标签」混合形态；`test_pi_success_with_non_json_stdout` 把「非 JSON→raw_output 兜底」固化为正确契约。
- E2E 手动造 `pool="history"`，绕过 `02-auto-index` 的 sentinel 门控，没测「首次空 reference_dir→sentinel 锁死→history 永远空」。
