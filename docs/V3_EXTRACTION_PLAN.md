# v3 抽取策略实验计划

## 结论先行

v3 不应直接把全库从 batch 15 改成 batch 1。已经证明的是“batch 1 产出更多”，尚未证明
“全量 batch 1 的端到端准确率足以覆盖 12 倍请求成本和新增噪声”。正确顺序是先改善选择性，
再比较 batch 5、batch 1 与自适应小批量。

本计划与冻结中的 v2 完全分开：新 config、新 store、新 fingerprint、新预注册；不得修改或覆盖
`train150`、`dev100`、`test100` 的 v2 artifact。

## 已知证据

60 个 session 的随机对照：

| batch | 零产出 | memory/session | 相对抽取请求 |
|---:|---:|---:|---:|
| 15 | 11.7% | 2.7 | 1.0×（约 400） |
| 5 | 8.3% | 4.8 | 2.5×（约 1000） |
| 1 | 0.0% | 12.7 | 12×（约 4800） |

但“12.7 / 2.7 = 4.7×”不能解释为准确率提升：

- 在 8 个目标 session 中，batch 1 总 memory 是 8.1×，真正的 user fact 只增加 2.5×；
- 新增 memory 的 69.5% 是 assistant recommendation，内容真实但多数与问题无关；
- 在专门挑选的 6 个“缺失事实导致失败”问题中，恢复事实只修复 2 个；另外 4 个仍失败，
  其中 2 个受到新增噪声或事实精度下降影响；
- 这 2 个修复若放回当时的 dev50，相当于观察到 +4 个百分点，但样本是按失败原因挑选的，
  不能外推为全量提升预测。

因此目前对全量 batch 1 的可信准确率预测是：**未知**。4.7× 是 memory 数，不是质量倍数。

## 研究问题

1. 能否在不淹没检索的前提下恢复数字、日期、变更和用户事件？
2. 哪些 session 值得使用 batch 1/5，能否把成本限制在 batch 15 的 2–4×？
3. 更小批次增加的是 answer-bearing fact，还是大量真实但低效用的列表？
4. 抽取更多之后，检索 cap、context token 和 answerer reasoning 是否成为新的瓶颈？

## 候选策略

### A. batch 5

最保守候选。请求约 2.5×，memory/session 约 1.8×。它不能消灭零产出，但可验证收益是否在
batch 1 之前已经饱和。

### B. 全量 batch 1

作为质量上限和成本上限，不直接作为默认产品方案。必须同时报告用户事实、assistant 内容、
检索候选数、上下文长度和端到端正确率。

### C. 自适应小批量

第一遍仍按 batch 15；只有同时满足“高价值风险”和“覆盖不足”的 session 才用 batch 1/5
重新抽取。不能只重跑零产出 session，因为损失是连续的：本应有 10 条却只得到 2 条不会触发
zero-yield retry。

候选的确定性信号在 train150 上建立，冻结后使用：

- 日期、金额、数量、持续时间、比较和更新措辞密度；
- 用户事实与 assistant recommendation 的比例；
- turn 数、字符数和实体数量；
- 第一遍 memory 数相对可解释内容单元明显偏低；
- Stage B fallthrough、缺失 provenance 或事实精度风险。

这些信号只能用来预测“值得重抽”，不能读取 benchmark gold 或 answer session 标签。

### D. 选择性感知的抽取提示

在缩小 batch 前先修目标函数：区分 durable user facts、被明确询问过/采纳的 recommendation，
以及一次性候选列表。不能简单删除所有 assistant recommendation，因为该类问题真实存在；应把
“逐项推荐清单”压成带 provenance 的集合/摘要，同时保留用户采纳、比较或追问的具体项。

## 实验顺序

### V3.0：离线标注与预测器

1. 只用 train150 和已有 batch pilot，建立固定的 answer-bearing / useful / grounded / noisy 标注表。
2. 测量 batch 15/5/1 的用户事实召回、精确数量保真、unsupported rate、推荐噪声和重复率。
3. 训练或制定一个简单、可解释的 risk rule；若规则不优于“随机选择同样比例 session”，停止
   自适应路线。

### V3.1：固定小样本抽取实验

在预注册的 train150 session cohort 上比较：

- batch15 control；
- batch5；
- batch1 ceiling；
- adaptive-2x；
- adaptive-4x；
- selective-prompt + 最佳 batch。

先过抽取门，才允许花 answerer/judge 配额。

### V3.2：端到端验证

1. 每个通过门槛的策略建立独立 store 和索引。
2. 固定相同 retriever、answerer、judge、context cap 和 fallback。
3. 至少三次重复，报告 majority accuracy、方差、上下文、fallback、请求和 token。
4. 使用新的未调参 holdout 或新增 benchmark；不得把已经看过 aggregate 的 dev100/test100 当成全新
   未见证据。

## 预注册决策门

建议在运行前固定以下门槛；这是采用标准，不是收益预测：

| 门 | 最低要求 |
|---|---|
| Fidelity | unsupported fact 不高于 batch15；数字/日期精度不下降 |
| Utility | answer-bearing user-fact recall 明显高于 batch15 |
| Noise | assistant 列表占比和检索候选膨胀有上限 |
| Cost | 默认候选 ≤4× 抽取请求；batch1 只作 ceiling |
| Context | 中位上下文 ≤1.5× v2，或有同等准确率下的明确 Pareto 优势 |
| Accuracy | majority accuracy 至少 +3pp，或 extraction-attributed failures 减少 ≥30% |

若 batch1 只增加 memory 数而不通过 accuracy/context 门，就保留为诊断上限，不进入产品。

## 成本预期

按现有测量：batch5 约 2.5×，全量 batch1 约 12×。若自适应策略把 batch1 限制在 10%/20% 的
session，粗略请求量约为 batch15 的 2.1×/3.2×；实际值必须把 Stage B、失败重试和 provider
token 限制计入 usage ledger 后重算。

## 能提升多少

当前唯一直接证据是目标反事实中的 2/6 修复；它说明提升是可能的，也说明多数缺失事实即使恢复
仍不会自动变成正确答案。合理承诺不是一个未经测量的百分比，而是：

- batch1 的**产量上限**已知为约 4.7×；
- 相关 user fact 的局部增量约 2.5×；
- 全量端到端准确率增益未知，必须由 V3.2 测量；
- 只有达到预注册的 +3pp/失败减少 30% 且成本 ≤4×，自适应方案才值得替换 v2。
