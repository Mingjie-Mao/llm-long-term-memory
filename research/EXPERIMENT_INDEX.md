# 实验索引

更新日期：2026-09-21（Australia/Sydney）

本文件是实验入口，不是新的结果报告。数字仍以对应的预注册、冻结包、逐题结果和分析文件为准。
实验分为三类：

- **正式冻结结果**：可以在 README/项目报告中引用，但必须同时保留口径和限制；
- **开发结果**：只用于选方向，不能冒充未见终测或统计显著性证据；
- **停止/未晋级**：负结果同样保留，不在原标签、原题集上调参后重跑。

默认维护责任人为仓库维护者；每项实验的“主要入口”中，预注册约束运行，decision/analysis 文件给出
权威状态，冻结或封存目录保存不可改写的证据。若移交维护，应在本索引中显式改写责任人和状态。

## 主线实验

| 实验 | 类型 | 状态 | 核心结论 | 主要入口 | 允许的下一步 |
|---|---|---|---|---|---|
| v2 `test100` | 正式冻结终测 | 完成 | LLTM 72%，整段历史 86%，naive RAG 65%；LLTM 对 naive RAG 的 +7 个点不显著 | [预注册](../results/prereg-v2-final.md) · [汇总](../results/final/test100-aggregate.md) · [冻结包](../results/frozen/v2-final/) · [封存逐题结果](../results/sealed/test100/) | 只能在新的、独立冻结的数据上测新版本；不得回到 `test100` 调参 |
| v3.3 `dev60` | 冻结开发验证 | 通过但有限定 | 55.0% 对 v2 对照 46.7%，配对 p=0.1797；后续审计发现一个 confidence 门未测到预期含义 | [预注册](../results/prereg-v3-dev60.md) · [验证报告](../results/validation/v3-dev60.md) · [冻结包](../results/frozen/v3-candidate-dev60/) · [封存结果](../results/sealed/v3-dev60/) | 可作为开发证据，不可与 v2 `test100` 排成版本曲线，也不可宣称稳定提升 |
| v4.2 引用式枚举 | 冻结开发比较 | `not_promoted` | 计数主指标仅净增 1，非计数方向门失败；事后重评分仅作探索性分析 | [预注册](../results/prereg-v4.2-cited-enumeration.md) · [结论](../results/v4.2-result.md) · [冻结包](../results/frozen/v4.2-development-20260912e/) | 先完成人工金标复核并建立新版本仪器；不得把修订金标继续称为原隐藏集 |
| v2b batch-8 gate16 | 富集开发门 | `STOP_NO_48` | 抽取和 source-local fallback 有改善，但 memory-only 净改善未稳定复现 | [预注册](../results/prereg-v2b-gate16.md) · [决策](../results/v2b-gate16-decision.md) · [分析](../results/analysis/v2b-gate16-qa.md) | 保留 source-local 机制；不扩大原 v2b 到 reasoning-48 |
| v2c gate8 + reasoning-48 | 开发确认 | 通过 | gate8 通过；reasoning-48 单次 37/48（77.1%），来源会话召回 95.8%；其中一题金标经审计为内部冲突 | [gate 预注册](../results/prereg-v2c-gate8.md) · [48 题预注册](../results/prereg-v2c8-reasoning48.md) · [决策](../results/v2c-gate8-decision.md) · [48 题报告](../results/analysis/v2c-reasoning48-final.md) | 不在 reasoning-48 上继续改候选或重跑；若要对外比较，使用另行冻结的新终测集 |
| v2d gate16 | 富集开发门 | `STOP_NO_48` | 8/16 与冻结对照持平；14 道机制题 2 胜 2 负；source-detail 无退化，但未达到晋级条件 | [预注册](../results/prereg-v2d-gate16.md) · [分析与决策](../results/analysis/v2d-gate16.md) | 保留失败与运行账本；不在同一 16 题上修补并复用标签，不扩大到 48 题 |
| v2e | 未开始 | `BLOCKED_ON_DIAGNOSIS` | v2d 的 9 次拒算中，**0 次**能从已提交的逐题行里分类；候选方向未确定 | [拒算分类](../results/analysis/v2d-refusal-taxonomy.md) · [工具](../tools/v2d_refusal_taxonomy.py) | 先让拒算记录可分类（已完成），再在新的 12–16 题开发门上跑第一次 v2e；不得在 v2d 的 16 题上调参 |

## v3 调整链

| 阶段 | 状态 | 说明 | 证据 |
|---|---|---|---|
| v3.1 reasoned | 调参持平 | tune42 与 v2 都为 66.7% | [报告](../results/validation/v3-phase3-tune1.md) · [预注册](../results/prereg-v3-phase3.md) |
| v3.2 adaptive | 停止 | 准确率方向为正，但上下文预算门失败 | [报告](../results/validation/v3-phase4-tune2.md) · [预注册](../results/prereg-v3-phase4-adaptive.md) |
| v3.3 compact | 调参门通过 | 保持 v3.2 正确率并把中位上下文从 1,476 降至 1,124 | [报告](../results/validation/v3-phase5-tune3.md) · [预注册](../results/prereg-v3-phase5-compact.md) |

## 支撑性实验

| 实验 | 状态与用途 | 证据 |
|---|---|---|
| 抽取批大小 | 完成；证明 batch 15 → 8 的抽取保真度改善，但后续 v2b gate 表明不能直接等同于 QA 提升 | [预注册](../results/prereg-batch-size.md) · [结果](../results/batch-size-result.md) |
| 检索权重扫描 | 完成；当前语料中仅语义排序优于加入 BM25、importance、entity 等组合 | [结果](../results/retrieval-weights.md) |
| 早期 heldout 方差 | 历史证据；用于说明重复运行波动，不作为当前终测 | [报告](../results/heldout-variance.md) |

## v2e 的前置诊断

v2d 的 gate16 里 16 题只有 2 题真正由代码算出数值，9 题因操作数无效被拒算（其中 6 题答错）。
但 `tools/v2d_refusal_taxonomy.py` 对已提交逐题行的分类结果是：**9 次拒算全部不可分类**。

原因是当时 `v2d.compute` 每次拒算只记一句话，而 `an item has no valid source label` 同时
代表两种需要相反修复的失败——模型没给出成员文本，和模型引用了它根本没见过的记忆标签；
被拒的操作数本身没有落盘。

因此 v2e 的第一步不是改 prompt，而是让拒算可诊断：`v2d.compute` 现在为每次拒算记录
机器可读的 `cause`、模型实际提交的操作数，以及当时可引用的标签集合。判定本身没有变化
（接受/拒绝的条件、顺序、`reason` 文案和 `computed` 标志都不变），由
`tests/test_v2d_refusal_diagnostics.py` 固定。**已有的 v2d 行不会因此变得可分类**，
它们写在这次改动之前；下一次运行才具备分类所需的证据。

## 已知复现缺口

- `results/analysis/v2c-offline-gate.{json,md}` 记录的 exact-reference 规则文本，来自跑该零调用
  门时的 `evaluation/runners/v2c.py`，早于最终进入 reasoning-48 确认的 v2c 规则。今天重跑
  `tools/v2c_offline_gate.py` 不会复现这份文件，**也不应覆盖它**：它记录的是当时实际运行的内容。
- `tools/v2b_offline_gate.py` 的输出曾随解释器哈希种子变化。`ingest/fidelity._extract_facets`
  返回集合，哪个转移先被计数因此不固定，序列化出的键序也随之变化——同一份证据在不同运行下字节
  不同，重跑就不再是一种校验。计数本身从未变动。现已按固定顺序序列化；其余七个 v2 分析工具
  连跑三轮字节一致。

## 复现与保护规则

1. `results/frozen/`、`results/sealed/` 和签署后的预注册只读；修正文义时新增审计说明，不覆盖原件。
2. 同一数据、answerer、judge、prompt 和 baseline store 未变化时复用冻结 baseline，不为每个候选重跑。
3. 新机制先过零调用 evidence-coverage gate，再运行最小题集；只有预注册门通过才扩大。
4. 开发题集一旦看过逐题结果，就不能再充当未见终测。修复后的候选必须换新标签，并按协议决定是否换新题集。
5. README、项目报告或发布清单新增数字前，运行 release audit，并同时核对预注册、逐题行、usage、测试数和树摘要。

目录规则、active/archived 的进入条件，以及 `scripts/` 与 `tools/` 的分工见 [README](README.md)。
完整实验方法、统计限制和证据入口见[评测文档](../docs/EVALUATION.md)。仓库文件分层与瘦身建议见
[仓库审计](../docs/REPOSITORY_AUDIT.md)。
