# 实验沿革

这里合并了原 README 中“一路是怎么走过来的”和“每个阶段，按顺序”两套重复记录。每行有各自题集，只能比较**同一行内的候选与对照**，不能把正确率跨行画成持续上升曲线。`dev60` 刻意包含较多时间推理和多会话问题。

| # | 阶段 | 改了什么 | 题集 | 结果 | 中位上下文 |
|---|---|---|---|---|---:|
| 1 | v1 | 第一个端到端系统：两阶段抽取、双时间事实、混合检索 | `heldout100` | **70%** · 重复 71、73 | 1,468 |
| 2 | v2 开发 | 选臂：平铺排序 vs 会话连贯上下文，外加基线 | `dev100` | 平铺 **67%** · 连贯 63% · naive RAG 68% | 576 |
| 3 | **v2 最终** | 什么都没改——冻结的 v2 在未见数据上对两个基线 | `test100` | **72%** · 整段对话 **86%** · naive RAG 65% | **574** |
| 4 | v3.0 | 答题策略：先说清这题要做什么操作，再回答 | `pilot48` | **72.9%** vs 对照 68.8% · `p=0.6875` | 605 |
| 5 | v3.1 | 同思路，调参 | `tune42` | **66.7% vs 66.7%** —— 打平，两个目标切片各退 10 点 · **负结果** | 558 |
| 6 | v3.2 | 为时间/聚合类问题注水逐字原文 | `tune42` | 73.8% vs 66.7%，但 2.52 倍上下文，超出注册的 2 倍上限 · **STOP** | 1,477 |
| 7 | v3.3 | 同样的证据，压缩：精确原句、跨会话公平分配、425-token 上限 | `tune42` | **73.8%** 保持，降到 1.92 倍 · 门通过 | 1,124 |
| 8 | **v3.3 验证** | 什么都没改——冻结的候选对 v2 对照 | `dev60` | **55.0%** vs **46.7%** · +8.3 点 · `p=0.1797` · 置信度门的限定见下文 | 1,115 |
| 9 | 诊断 | 222 个失败的分类，加一次离线探针 · **零 API 调用** | 3 个可读池 | 检索失败 **0–6%** · 有源却弃权 **44–50%** | — |
| 10 | 仪器 | 229 条合成探针，ground truth 由 SQL 推导 | 由 `train150` 派生 | `current_state` 86.7% · `comparison` 58.7% · `duration` 42.5% · `count` 4.0% | 1,120 |
| 11 | v4.0 flat | 操作先行的判定与 Python 算术；该臂关闭时间线渲染 | 142 条开发探针 | **61.3%** vs 54.9% · 22 胜 / 13 负 · `p=0.1755` | — |

在这些阶段之前，最初的 memory-only 在 `dev50` 上是 26.0%，中位上下文 465 token；同期 full-context 为 56.0%，naive RAG 为 54.0%。抽取重写后 memory-only 达到 56.0%，条件原文回退臂达到 72.0%，中位上下文 1,455 token。这些是开发集结果，不是后来的 `test100` 终测。

更早的 `heldout100` 三次运行是 70、71、73，均值 71.3%，极差 3 个点。v2 `test100` 每臂只运行一次，没有重复方差。v3 pilot 与 dev60 每臂三次，报告多数票正确率。v3.1 打平，v3.2 虽然涨分但超过注册的上下文上限，这两项负结果继续保留。

## 必须随结果一起阅读的限定

- v3.3 dev60 归档确实通过了当时实现的检查。后续 schema 审计发现 confidence 恒为默认值，所以置信度门没有实际测量预期条件；另外八项门仍可解释，准确率和配对统计不受影响。保留原归档，在其旁边补充限定。
- selected source recall 表示选中 memory 来自**至少一个** gold 会话。它不保证完整证据覆盖，更不保证答案事实没有被抽取丢掉。98.3% 与 55.0% 的差距可以指导诊断，不能全归因为答案合成。
- 229 条 synthesis probes 的目标来自可读 store 的 SQL 推导，是诊断工具，不是外部基准或未见终测。后续开发运行使用 142 条子集，不能把子集分数与最初整个池直接比较。
- v4.0 flat 已经测量，所以“v4 尚未测量”已经过时。+6.3 个点仍不具有统计确定性。关系路由和扫描是另外的研究方向，没有默认接入服务。

## 证据入口

| 阶段 | 记录 |
|---|---|
| 初始开发 | [冻结 v1 表](../results/frozen/chronomem-v1/table.md) |
| 早期保留集重复 | [Held-out variance](../results/heldout-variance.md) |
| v2 选臂 | [dev100 aggregate](../results/validation/dev100-aggregate.md) |
| v2 终测 | [test100 aggregate](../results/final/test100-aggregate.md) |
| v3 调整 | [Pilot](../results/validation/v3-answer-pilot-aggregate.md)、[tune1](../results/validation/v3-phase3-tune1.md)、[tune2](../results/validation/v3-phase4-tune2.md)、[tune3](../results/validation/v3-phase5-tune3.md) |
| v3.3 验证 | [dev60 aggregate](../results/validation/v3-dev60.md)、[schema 审计](../results/audit/v3-verdict-schema-never-sent-20260906.json) |
| 诊断 | [失败分类](../results/failure-taxonomy.md)、[合成探针](../results/analysis/synthesis-probes-v3.3.md) |
| v4 开发 | [第一次尝试](../results/archive/v4.0-attempt1/README.md)、[flat 臂](../results/archive/v4.0-flat/README.md) |

持续研究记录见[当前状态](CURRENT_STATUS.md)，实际代码连接见[架构图集](ARCHITECTURE.md)。
