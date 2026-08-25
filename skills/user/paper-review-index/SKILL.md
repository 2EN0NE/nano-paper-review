---
name: paper-review-index
description: 建立论文索引。当用户想将一批 PDF 论文入库、建立可检索的本地索引、或添加更多论文到历史库时使用。索引后才能 search 和 review。
---

# paper-review-index

把 PDF 论文批量索引进本地库（SQLite FTS5 BM25 + FAISS 向量）。这是检索和评审的前置。

## 建索引

```bash
paper-review index --source-dir <pdf目录> --pool history
```

- 默认源目录：`{data_dir}/origin/pdf/`（不传 `--source-dir` 时用这个）。
- `--pool`：论文归属池，`history`（历史库，默认）或 `pending`（当前批次）。
- `--epoch-size`：每批处理论文数（默认 200），越小越省内存（2C/4G 机器上遇到 OOM 就调小）。

## 索引过程做的事

1. 提取 PDF 文本（PyMuPDF），从文件名解析元数据（标题/作者/年份，格式如 `01.提案表-XXX-张三.pdf`）。
2. 分块（512 字，overlap 128，段落边界优先，参考文献截断）。
3. 生成 chunk 向量 + 写 FTS5，**SHA-256 内容去重**（同内容论文只存元数据、共享向量）。
4. 增量写入：已索引的论文再次索引会去重跳过，无需全量重建。

## 验证

```bash
paper-review status     # 看 papers / chunks 数量和各池分布是否涨了
```

索引完成后，可用 `paper-review search` 检索，或 `paper-review review` 直接评审（Pre Phase 也会自动建索引）。
