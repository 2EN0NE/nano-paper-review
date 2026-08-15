# 技术相似检索：L2 召回 + L3 技术特征精排

检索要回答的不是"文本多像"，而是"技术方法多像"。为此引入 **Technical Similarity** 三层标尺（L1 领域 / L2 问题 / L3 方法），评审需要 L2/L3（尤其 L3）。检索架构定为：**L2 由 embedding 向量召回并做门槛过滤，L3 由 Technical Feature Set 的 Overlap 覆盖度做主排序键**；通用 reranker 短期降为审计信息（不参与排序），长期用"技术相似对"微调后回归。

## 决策链

- **Technical Feature Set（技术特征集）**：一篇论文"用什么技术方法解决什么问题"的可检索签名。与 **Tag Library（标签库）** 区分——`papers.features` 存 LLM 抽取的全量技术特征（检索用），`papers.tags` 存评审抽取的 3 个权威技术标签（元数据用）。评审打标签时以技术特征集为参考。
- **抽取形态**：LLM 主线 + 词表确定性兜底，两者并集汇总。词表匹配是 LLM 抽取的严格子集，其唯一价值是"已知词永不丢"（LLM 不稳定时的兜底），不做对等双线。
- **抽取步骤前移**：技术特征抽取必须在 `04-batch-search` 之前（L3 精排依赖它），由现有 `05-extract-keywords` 升级并前移实现。
- **写回时机**：Pre Phase 抽完立即写 `papers.features`，使同一批 review 内先抽完的论文立即成为后续 Subject 的可用 Reference 特征。
- **集合算法**：Overlap 覆盖度 `|subject特征 ∩ reference特征| / |reference特征|`（方向性：Reference 的技术方法是否落在 Subject 技术范围内）。不加权，粒度控制前置到抽取环节（让 LLM 优先抽具体方法词）。
- **combined 分层合成**：L3 是主排序键（Overlap 降序），L2 是软门槛（vec 过阈值优先，不硬删）+ vec tie-break。冷启动（Overlap 全 0）时 combined 退化为 L2 vec（不恒 0）。软门槛而非硬删的理由：vec 在哈希降级/维度不匹配时无意义，硬删会误杀全部候选。
- **冷启动**：渐进退化。auto-index 批量索引的 Reference 无 features 时 L3 失效、退 L2-only，随论文被 review（抽特征 + Subject Copy）features 自然积累。哨兵显式暴露 L3 覆盖率，不静默退化。
- **reranker 去留**：bge-reranker-v2-m3 在同域语料饱和（强相关 0.9408 vs 无关 0.9396，无区分度），短期退出排序、保留分数作审计信息；长期用"技术相似对(正) / 同域非相似对(负)"（评审结果可自动构造）微调后回归排序键。

## Considered Options

- **Jaccard（对称）vs Overlap（覆盖度）**：选 Overlap。Subject 特征（LLM 抽 5~8 词）与 Reference 特征（3 标签）大小天然不均衡，Jaccard 分母为并集会稀释分数；Overlap 精确表达"Reference 技术方法是否落在 Subject 范围内"，且可解释（命中 3/3）。
- **加权和 `α·vec + β·overlap` vs 分层**：选分层。加权和把"门槛"（L2）和"排序"（L3）两个不同层混成线性公式，会允许"领域无关但技术词巧合命中"的噪声靠 L3 挤进前列；分层用 L3 主键 + L2 软门槛（vec 过阈值优先）表达层级，语义干净。
- **混进 `papers.tags` vs 新开 `papers.features`**：选新字段。tags 是评审产出的 3 个权威标签（精炼），features 是 LLM 抽取的全量特征（可含噪声），粒度与用途不同，混用会让下游分不清权威与待清洗。
- **冷启动预热（对 origin 批量 LLM 抽特征）vs 渐进退化**：选渐进退化。预热是几百篇 × LLM 的一次性大账单，且大部分论文短期内不会被检索；渐进退化贴合 Subject Copy 自愈机制。
- **reranker 继续当排序键 vs 立即退役 vs 降为审计**：选降为审计。机制（cross-encoder 逐对交互）是对的，失效是训练目标错位（通用相关性而非技术相似）；降为审计止血，微调是独立后续立项而非本回合。

## Consequences

- `hybrid_search` 的 `combined` 计算从"rerank 单分覆盖"改为"L2 门槛 + L3 主键 + vec tie-break"分层逻辑（见 ADR 0014 记录的 Bug F）。
- `papers` 表新增 `features` 列（JSON），与 `tags` 并存；Store 需提供读写方法。
- 检索对"技术特征"产生新依赖：冷启动期间 L3 稀疏，检索质量靠 L2 兜底，随 review 次数渐进改善——需通过哨兵暴露覆盖率，避免静默退化（呼应 ADR 0014 的止血哲学）。
- 前移步骤编号：pre-review 内 `05-extract-keywords` 与 `04-batch-search` 互换（特征抽取前移），影响 Scaffold Template 与 pipeline.yaml。

## 真实验证发现（2026-08-15，4 篇美团技术文章）

隔离 data-dir 跑完整 review（ZGC / CMS GC / KV 存储 / Flutter），发现：

- **管线跑通，L3 覆盖率 1.0**：LLM 抽取成功（ZGC → `["ZGC","STW","CMS 收集器","G1 收集器"]`），features 写回 4/4。
- **L2 正确兜底**：ZGC 检索时 CMS GC 排第 1（vec=0.87），L1/L2 区分有效。
- **L3 失效（特征粒度不一致）**：ZGC 抽 `"CMS 收集器"`（泛称），CMS GC 抽 `"CMS 并发标记清除"`（具体算法），字符串不同 → 精确交集为空 → overlap 恒 0，L3 形同虚设。Overlap 精确匹配对"同一技术的不同表述"零容错。
- **哨兵误报已修**：原"关键词恒空"读 `keyword_count`（词表兜底，LLM 可用时恒 0），误报；改为读 `feature_count`（LLM+词表并集）。

**后续改进方向**（需单独决策，不在本 ADR 闭环）：特征归一化（去后缀/统一短名）或子串/模糊匹配，让"CMS 收集器"与"CMS 并发标记清除"可对齐；或接受精确匹配低命中率、L3 作高置信增强。
