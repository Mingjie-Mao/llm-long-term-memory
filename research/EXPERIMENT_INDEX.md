# 实验索引

每条实验线一行：类型、状态、结论、证据入口、允许的下一步。数字以对应的预注册、冻结包、
逐题结果和分析文件为准；本文件不是新的结果报告。过去的运行流水记录已删除，需要时查
`results/` 下的预注册、修订文件和 git 历史。

历史实验工具可能引用当时的源码路径；复现须使用该实验的冻结源码快照或注册 Git 版本。
主分支的代码整理不改变历史注册，也不能把旧工具直接跑在当前源码上当成同一实验。

实验分三类：**正式冻结结果**（可在 README 引用，须带口径）、**开发结果**（只用于选方向，
不能冒充未见终测）、**停止／未晋级**（负结果同样保留）。

## 当前阶段的开发记录

2026-10-12 起，用户决定先在 LongMemEval-S 上把已知问题修完（见[路线图](ROADMAP.md)）。
这一阶段的每次开发运行在下表加一行：命名空间、配置、git SHA、题数×次数、结果、结论。

| 日期 | 命名空间 | 配置／改动 | git SHA | 题数×次数 | 结果 | 结论 |
|---|---|---|---|---|---|---|
| 2026-10-12 | `t4-baseline-v1`（零调用） | 阶段 A：复用已有 t4 结果作基线，重放原话选取，逐题标首损 | 基于 b264034 | 200×3（已有） | 34 题错过、72 份错答；回答 26 题／52 份、评分 4／9、金标范围 2／5、检索 2／6；歧义 6 题单列 | 回答层是主要损失；新增“错误前提未拒答”为 C 的目标；[结果](../results/analysis/t4-baseline-v1.md) |
| 2026-10-12 | `anchor-v2-backfill-v1`（零调用） | 阶段 B：v2 出处回填到 dev100／heldout100 的副本，重放 t4 证据补全 | 基于 4008ad4 | 200×3（重放） | 锚点准确率 87.8→96.2%、87.0→96.6%；补全含金标的行 +6／+6，变化的 8 题金标原话本已在上下文 | 不做端到端；记忆层后续改用 `*-anchor-v2` 库；[结果](../results/analysis/anchor-v2-backfill-v1.md) |
| 2026-10-12 | `two_stage_raw_primary_t4c.t4c-v1-*` | 阶段 C v1：计数／日期／先后由程序计算并覆盖；另以缓存回放“只记录不覆盖” | 基于 4008ad4 | 85 触发题×3（其余 115 题请求与 t4 逐字一致，未重跑） | t4 231/255；覆盖 203；只记录 230 | 覆盖为负结果，只记录持平；不接入产品；[结果](../results/analysis/t4c-v1.md) |
| 2026-10-13 | `t4c-v2-replay`（零调用） | 阶段 C v2：只对纯计数题、只记录不覆盖（变体 `t4c2`）；请求均已采样过，复用样本 | 基于 b4abb43 | 27 触发题×3（复用） | 无歧义 64→66/75；v1 的 75 次覆盖无一把错答改对 | 噪声内，不接入；[结果](../results/analysis/t4c-v2-replay.md) |
| 2026-10-13 | `fact-pairs-v1` | 阶段 D：标注 51 道知识更新题；向量候选匹配；关系判定；应用到时间线（存储副本） | 基于 b4abb43 | 21 替换对＋6 并存对；判定 3 次 | 候选召回 21/21@1；判定召回 63/63、误关 0/18；正确时间线 5/21→21/21（按序）／19/21（乱序）；26 题抽取丢失 | 修复解析器数字词折叠缺陷；小标注集结果，后续实际库审计见下；[判定](../results/analysis/fact-pairs-v1.judge.md)、[时间线](../results/analysis/fact-pairs-v1.apply.md) |
| 2026-10-13 | `fact-pairs-v1` 逐对判定 | 阶段 D 第二轮：逐对、按时间先后呈现的判定（`transition-pair-v3`），相似度 ≥0.65 门控；接入 `ingest.update_linking`（默认关） | 基于 8dac788 | 21 替换对＋6 并存对；37 对×3 | 时间线按序 21/21、乱序 21/21；并存 6/6；顺序无关 52/52；未标注关闭 0 | 小标注集通过，但实际库回填发现误关，当前版本停止；[结果](../results/analysis/fact-pairs-v1.pairwise.md) |
| 2026-10-13 | `link-backfill-d1-resume-20261013` | D：原库副本，`transition-pair-v3`、0.65、K=5；每对一次 | 基于 d02ccdb | dev100 100 用户、heldout100 18 用户 | 497／110 对判定；实际关闭 27／4 处。助手复查草稿分别为 11 合理、12 错误、4 不确定，以及 1／2／1；dev100 两条状态有遍历顺序差异 | **停止扩大当前版本回填**；先人工复核关系错误和生命周期问题，暂不做全量端到端。[运行记录](../results/analysis/link-backfill-d1-resume-20261013.run.json) · [dev100](../results/analysis/link-backfill-d1-dev100-reviewed-20261013.md) · [heldout100 部分](../results/analysis/link-backfill-d1-heldout100-reviewed-20261013.md) |
| 2026-10-13 | `oracle-evidence-v1` | 只给金标原话，误删原有记忆上下文 | 基于 d02ccdb | 22 题×3 次 | 大量错误弃答 | **设计无效**，保留作失败记录，不据此估计上限。[报告](../results/analysis/oracle-evidence-v1.md) |
| 2026-10-13 | `oracle-evidence-v2` | 回答侧证据选择诊断：保留 t4 记忆上下文，只替换检索原话 | 基于 d02ccdb | 22 题×3 次 | t4 22/66、金标原话 21/66；题型有得有失 | 暂不投入证据选择模型；不解释为所有方法的上限。[报告](../results/analysis/oracle-evidence-v2.md) |
| 2026-10-14 | `original-test100-v2-t4-dispatch-v1.server-recovery-v1` | 原评分模型的额外有限恢复；从独立目录加载已核验冻结源码 | 冻结 1339587；当前 f59e2f7 | 对照第 3 轮续存 50 份回答、4 份评分；额外恢复 3 次 | 原 test100 累计 500/600 回答、404/600 评分；额外恢复仍遇 500，`EXHAUSTED_AGAIN` | 保留原评分模型，等待恢复后另行登记有限续跑；未完成、未更新正确率。[停点审计](../results/analysis/original-test100-v2-t4-dispatch-v1.server-recovery-v1.stop-audit.json) · [源码核验](../results/analysis/original-test100-v2-t4-dispatch-v1.frozen-runtime-v1.json) |

## 主线实验

2026-10-19 原 test100 已完成并通过零调用完整性审计。新 `server-recovery-v3` 独立阶段
给此前卡住的同一份答案评分成功后，父流程保存全部 600 回答／600 评分并生成通过的注册门。
`server-recovery-v2` 原型仅零调用预检，因账本路径问题不执行。模型、旧失败记录和旧预算保留。
[门判定](../results/analysis/original-test100-v2-t4-dispatch-v1.gate.json) ·
[完成审计](../results/analysis/original-test100-v2-t4-dispatch-v1.completion-audit-v1.json)

| 实验 | 类型 | 状态 | 核心结论 | 主要入口 | 允许的下一步 |
|---|---|---|---|---|---|
| 原文为主 t3／t4 对 v1（heldout100，每组 3 次） | 回归级开发比较 | **`PASS`** | v1 82/81/80、t3 87/86/87、t4 89/86/88，均值 81.00 / 86.67 / 87.67；t4 净 +6.67。回答侧 +5.67 达贡献门，检索侧 +1.00 未达 +2。t4 中位总上下文 6,147 | [预注册](../results/prereg-raw-primary-t3-t4-heldout100-v1.md) · [门判定](../results/analysis/raw-primary-t3-t4-heldout100-gate.md) · [完成审计](../results/analysis/t3t4-completion-audit-v1.json) | t4 已是服务默认配置（`answer_profile=t4`） |
| grounded v1–v18（分页通读、逐条核验引用） | 开发线 | **`FAIL`，停止** | dev100 每组 3 次：v18 平均 74.33 对 v1 77.67，净 −3.33；48 份错答是证据齐全的拒答，来自自身的引用核验；调用 3.6 倍 | [决策](../results/paged-paired-dev100-v18-v1-decision.md) · [错因分类](../results/analysis/error-taxonomy-dev100-v18-v1.md) | 不再推进；代码移到 `evaluation/grounded/`，仅供复现 |
| t4 选择／计算修复（固定 21 题组件验收 v1–v4） | 开发组件门 | v1–v3 失败，**v4 通过** | v4：18/18 可答题、3/3 忠实说明缺口，中位上下文 6,393；20 题为严格回放、1 题新回答 | [v4 门判定](../results/analysis/t4-scoped-selection-train21-v4.gate.json) | — |
| t4 修复版完整回归（dev100，每组 3 次） | 回归级开发比较 | **`FAIL`** | 原 t4 88/89/88（88.33）、修复版 84/91/83（86.00），净 −2.33；配对区间含零。原因：修复规则被加到所有请求上 | [预注册](../results/prereg-t4-scoped-regression-dev100-v3.md) · [门判定](../results/analysis/t4-scoped-regression-dev100-v3.gate.json) · [逐题诊断](../results/analysis/t4-scoped-dev100-postmortem-v1.md) | 改为限定触发范围（`tools/t4_scoped_dispatch.py`） |
| 限定范围修复的结构回放 | 零调用回放 | 通过 | dev100 原 t4 的 300 份请求与输出逐字回放一致；21 题 18/18 + 3/3 保持 | [预注册](../results/prereg-t4-scoped-dispatch-replay-v1.md) | 只是历史回放，不是新的正确率 |
| 原 test100：旧 v2 对 限定范围 t4（每组 3 次） | 回归级比较，已暴露题集 | **PASS，审计通过** | 旧版 70/68/68（68.67%）；新版 86/87/85（86.00%），平均净 +17.33；600 回答／600 评分完整，中位上下文 6,254.5；模型不变 | [注册门](../results/analysis/original-test100-v2-t4-dispatch-v1.gate.json) · [完成审计](../results/analysis/original-test100-v2-t4-dispatch-v1.completion-audit-v1.json) · [资源与口径](../docs/EVALUATION.md#原-test100-新对照回归已完成) | 允许进入摘要总成本与正确率的独立预注册；保留历史 72% 和开发 87.67%，不是未见终测 |
| 记忆层审计（零调用） | 开发诊断 | 完成 | 标为替换的记忆约 89–91% 同键无旧值；51 道多会话知识更新题最多 11 道证据记忆共享某个键（线索，非更新正确率）；21 道偏好题 3 道抽到偏好记忆 | [结果](../results/analysis/layer-audit-v1.md) · [工具](../tools/layer_audit.py) | 见[路线图](ROADMAP.md) 阶段 D、E |
| v2 `test100` | 正式冻结终测 | 完成 | LLTM 72%，整段历史 86%，naive RAG 65%；LLTM 对 naive RAG 的 +7 个点不显著 | [预注册](../results/prereg-v2-final.md) · [汇总](../results/final/test100-aggregate.md) · [冻结包](../results/frozen/v2-final/) · [封存逐题结果](../results/sealed/test100/) | 历史终测保持不变；原题后续仅作已暴露回归，新未见终测需独立冻结题集 |
| v3.3 `dev60` | 冻结开发验证 | 通过但有限定 | 55.0% 对 v2 对照 46.7%，配对 p=0.1797；后续审计发现一个 confidence 门未测到预期含义 | [预注册](../results/prereg-v3-dev60.md) · [验证报告](../results/validation/v3-dev60.md) · [冻结包](../results/frozen/v3-candidate-dev60/) · [封存结果](../results/sealed/v3-dev60/) | 可作为开发证据，不可与 v2 `test100` 排成版本曲线，也不可宣称稳定提升 |
| v4.2 引用式枚举 | 冻结开发比较 | `not_promoted` | 计数主指标仅净增 1，非计数方向门失败；事后重评分仅作探索性分析 | [预注册](../results/prereg-v4.2-cited-enumeration.md) · [结论](../results/v4.2-result.md) · [冻结包](../results/frozen/v4.2-development-20260912e/) | 先完成人工金标复核并建立新版本仪器；不得把修订金标继续称为原隐藏集 |
| v2b batch-8 gate16 | 富集开发门 | `STOP_NO_48` | 抽取和 source-local fallback 有改善，但 memory-only 净改善未稳定复现 | [预注册](../results/prereg-v2b-gate16.md) · [决策](../results/v2b-gate16-decision.md) · [分析](../results/analysis/v2b-gate16-qa.md) | 保留 source-local 机制；不扩大原 v2b 到 reasoning-48 |
| v2c gate8 + reasoning-48 | 开发确认 | 通过 | gate8 通过；reasoning-48 单次 37/48（77.1%），来源会话召回 95.8%；其中一题金标经审计为内部冲突 | [gate 预注册](../results/prereg-v2c-gate8.md) · [48 题预注册](../results/prereg-v2c8-reasoning48.md) · [决策](../results/v2c-gate8-decision.md) · [48 题报告](../results/analysis/v2c-reasoning48-final.md) | 不在 reasoning-48 上继续改候选或重跑；若要对外比较，使用另行冻结的新终测集 |
| v2d gate16 | 富集开发门 | `STOP_NO_48` | 8/16 与冻结对照持平；14 道机制题 2 胜 2 负；source-detail 无退化，但未达到晋级条件 | [预注册](../results/prereg-v2d-gate16.md) · [分析与决策](../results/analysis/v2d-gate16.md) | 保留失败与运行账本；不在同一 16 题上修补并复用标签，不扩大到 48 题 |
| v2e 日期出处 | 开发比较 | `STOP` | 37/48 对 37/48，5 胜 5 负净 0；10 处变动里 7 处落在机制影响不到的题型上，与 answerer 方差一致。代码保留、默认关闭 | [预注册](../results/prereg-v2e-reasoning48.md) · [决策](../results/v2e-reasoning48-decision.md) · [门判定](../results/analysis/v2e-gate.md) | 不得在这 48 题上调参复用标签；要判这个机制需要对时间题富集的新题集 |
| v2d 拒算诊断 | 诊断已完成 | `未安排` | v2d 的 9 次拒算中 0 次能从已提交行分类；诊断管道已建好，但下一次 v2d 类运行才会产生可分类证据 | [拒算分类](../results/analysis/v2d-refusal-taxonomy.md) · [工具](../tools/v2d_refusal_taxonomy.py) | 若重启算术方向，用新标签和新的 12–16 题开发门 |
| v5.0 并行原文 | 开发比较 | `STOP` | B 37/48 净 0，C 36/48 净 −1。**三句目标原文全部送达，只有 1 句转化成正确答案**；离线门测的是证据位置，不是证据可用性 | [预注册](../results/prereg-v5-reasoning48.md) · [决策](../results/v5-reasoning48-decision.md) · [离线门](../results/analysis/v5-offline-gate.md) · [B](../results/analysis/two_stage_v5_fixed.v5-reasoning48-fixed.md) · [C](../results/analysis/two_stage_v5_planned.v5-reasoning48-planned.md) | 不得在这 48 题上调窗口预算复用标签 |
| 记忆即索引（离线） | 零调用开发诊断（`train150`） | `STOP` | 同预算下，用记忆定位原文轮次远不如直接用问题检索原文：4,000 token 时全部金标轮次覆盖 50.0% 对 78.1%，9 胜 50 负。**第一个出错的层是抽取，不是检索**：39 个未命中里 38 个是金标轮次没有任何记忆，另有锚点指错轮次的实例；有记忆的金标轮次 100% 被排进前 20 | [预注册](../results/prereg-memory-as-index-offline-v1.md) · [决策](../results/memory-as-index-offline-decision.md) · [分析](../results/analysis/memory-as-index-offline-v1.md) | 不为「记忆即索引」付费；「原文检索为主、记忆提供时间状态」需另行预注册 |
| 失败归因调和 | 零调用复核 | 完成 | dev50「10/14 在抽取」与 reasoning-48「6/11 证据在眼前仍答错」放到同一套最早丢失层分类后不矛盾：前者是 batch 15 的冻结 v2 系统，后者是读 batch 8 库、带原文细节补回的 v2c；题型构成相近。前者瓶颈是抽取，后者残余是推理（错误前提、计数、算术） | [调和记录](../results/failure-attribution-reconciliation.md) | 当前 batch 8 线的下一个机制应针对回答推理；关于冻结系统的结论仍指向抽取 |
| 时序生命周期金标 v1 | 零调用契约测试 | 通过 | 16 个案例（新增、替换、终止、并存、有目标/无目标、歧义弃权、目标不存在、复述折叠、链中插入、终止后新值、孤立终止、跨租户），每例正序、倒序、轮转三种写入顺序逐条摄取，均达到事先写定的时间线，第二次全量解析 0 写入。变异测试：关掉旧数据修复、把终止当继任、按行序猜目标，分别被抓到 3、21、15 个失败 | [金标](../tests/fixtures/temporal_lifecycle_gold_v1.json) · [测试](../tests/test_temporal_gold.py) | 只验证解析器；抽取阶段 B 是否给出正确操作类型，由 `temporal_pairs` 键控门另测（需调用） |
| 历史增长下的保持（模拟） | 零调用开发诊断（`train150` 副本） | 完成 | 把每个用户的历史复制扩大到 2×、4×：金标记忆仍在前 20 的比例 73.3% → 72.6% → 71.9%，几乎不动；4,000 token 内问题检索原文覆盖全部金标 78.1% → 71.9% → 67.8%，掉 10 个点。**记忆层对长期增长稳健，原文定位在固定预算下退化。** 顺带发现并修复一个扩展性缺陷：检索把全库命中的 id 塞进一条 SQL，库超过约 3.2 万条记忆就报错；修复后在 train150 上 300/300 组候选与旧逻辑逐项相同 | [结果](../results/analysis/history-growth-retention-v1.md) · [工具](../tools/history_growth_retention.py) | 捐赠历史来自其他用户，只竞争不矛盾；未跑时序解析；不测答案。真实跨时间实验仍需实际等待 |
| 按会话取回（离线） | 零调用开发诊断（`train150` 判定，`heldout100` 只看汇总） | 判定：继续用轮次 | Emergence 式「按轮次匹配、取回整段会话」在 8,000 token 以内远不如直接给轮次：train150 上 4,000 时 26.0% 对 78.1%（2 胜 78 负），16,000 时 82.2% 对 89.7%；heldout100 方向相同。会话太长，占满预算。只用了 BM25 匹配轮次（库里没有逐轮向量），是下限 | [预注册](../results/prereg-session-retrieval-offline-v1.md) · [train150](../results/analysis/session-retrieval-offline-v1.md) · [heldout100](../results/analysis/session-retrieval-offline-v1.heldout100.md) | 已注册的原文为主实验保持按轮次；会话层只在预算大得多时才值得再看 |
| 原文为主（heldout100） | 回归级开发比较 | `STOP`（成本规则） | R 83 对对照 70/71/73/72；对多数票净 **+14**（18 胜 4 负），无题型为负，时间推理 +6；R 比只有原文多 +7。**中位上下文 5,598 超过注册上限 5,500**：上限设在预期值上、没有余量，是注册失误，判定照写 | [预注册](../results/prereg-raw-primary-heldout100-v1.md) · [决策](../results/raw-primary-heldout100-decision.md) · [门判定](../results/analysis/raw-primary-heldout100-gate.md) | 不在 heldout100 上重跑或调参；同一方案在 train150 上以总上下文 ≤ 6,000 重新注册 |
| 原文为主 v2（train150） | 回归级开发检查 | **`PASS`** | 同一方案不变。R 127/150 对对照 98/150，净 **+29**（32 胜 3 负），多会话 +15、时间推理 +9，无题型为负；中位总上下文 5,686（上限 6,000，零调用预估 5,683）；二次回退调用 44 → 16。精确 McNemar p = 4.2e-07，不宣称显著。选择效应：4,000 预算按本题集的证据覆盖选定 | [预注册](../results/prereg-raw-primary-train150-v2.md) · [门判定](../results/analysis/raw-primary-train150-v2-gate.md) | 通过只许可在新的未见题集上注册比较 |
| 按题型分配原文预算（离线） | 零调用开发诊断 | 判定：保持固定 4k | 只看问题文本分类，汇总计数题给 8k：train150 覆盖 78.1% → 81.5%，但预计中位总上下文 6,119 超过 6,000；heldout100 上 86.0% → 86.0% 无提升。缺失的原话不是差一点没进预算，而是排不上 | [预注册](../results/prereg-adaptive-raw-budget-offline-v1.md) · [train150](../results/analysis/adaptive-raw-budget-offline-v1.md) · [heldout100](../results/analysis/adaptive-raw-budget-offline-v1.heldout100.md) | 预算不是杠杆，转向轮次检索 |
| 原始轮次融合检索（离线） | 零调用开发诊断（本地向量） | 判定：下一轮用融合 | 4,000 token 内全部证据覆盖：train150 关键词 78.1%、向量 80.8%、融合 **83.6%**（14 胜 6 负）；heldout100 86.0% / 83.9% / **88.2%**（3 胜 1 负）。单用向量不稳，融合在两套题上都不变差 | [预注册](../results/prereg-hybrid-turn-retrieval-offline-v1.md) · [train150](../results/analysis/hybrid-turn-retrieval-offline-v1.md) · [heldout100](../results/analysis/hybrid-turn-retrieval-offline-v1.heldout100.md) | 预期是小幅改进：train150 上 R 的 14 道检索类错题里 5 道可补全证据，另有 2 道反而缺失 |
| 记忆出处定位 v2（离线） | 零调用、无金标 | 通过；新摄取改用 v2 | 以「记忆里只在一轮出现的具体信息」为参照：v1 87.8% → v2 96.5%（train150）/ 96.2%（dev100），保留 v1 正确定位的 99.1%；去掉具体信息加分（参照无法偏袒的部分）仍有 93.6% / 93.4%。事后提出的「只在说话人自己的轮次里找」v3 保留率 97.9% / 98.2%，未达 99%，放弃；UCLA 那例未解决 | [预注册](../results/prereg-anchor-v2-offline-v1.md) · [train150](../results/analysis/anchor-accuracy-v1.train150.md) · [dev100](../results/analysis/anchor-accuracy-v1.dev100.md) | 指纹新增 `anchor` 字段，旧库不能用新规则续跑 |
| 事实作原文检索键（离线） | 零调用开发诊断 | 通过 | LongMemEval 的 key expansion：4k 全部证据覆盖 train150 77.4% → 81.5%（7 胜 1 负），heldout100 84.9% → 89.2%（4 胜 0 负）；只扩展向量一路在 train150 上为 +0，收益在关键词一路 | [预注册](../results/prereg-fact-key-expansion-offline-v1.md) · [train150](../results/analysis/fact-key-expansion-offline-v1.md) · [heldout100](../results/analysis/fact-key-expansion-offline-v1.heldout100.md) | 已实现：`fact_keys` 与 `build-turn-index --fact-keys` |
| 历史增长下的融合（离线模拟） | 零调用开发诊断 | 通过 | 融合里加入「检索到的记忆指向的原话」：4 倍历史 74.7% → 80.8%，1 倍 83.6% → 85.6%。按时间范围过滤未测（模拟里捐赠历史被标为晚于提问，会虚高） | [预注册](../results/prereg-growth-fusion-offline-v1.md) · [结果](../results/analysis/growth-fusion-offline-v1.md) | 已实现：`memory_anchors` |
| 偏好置顶块（离线） | 零调用开发诊断 | 不实现 | 三套题 21 道偏好题，偏好或画像记忆覆盖金标原话 0 道；任何记忆覆盖 7 道。偏好根本没被抽取成偏好记忆，置顶块无从携带 | — | 对应做法在抽取侧（Mastra 记录随口的偏好），需另行付费探针 |
| 原文为主 v4 对 v1（dev100，每组 3 次） | — | **运行前撤回** | 未跑。dev100 已由另一条线的 `paged-paired-dev100-v18-v1` 使用（600 份回答中已存 356 份），为保持一条主线、不重复消耗 dev100，v4 撤出；其改动保留在代码中（默认关闭），待 v18 结论出来后在其他题集另行注册 | [预注册](../results/prereg-raw-primary-dev100-v4.md) | — |
| 原文为主 t1：先算日期（heldout100，每组 3 次） | 回归级开发对比（已暴露集） | **已预注册，未跑** | v1 加两处显示改动：对话头写星期与距提问天数；用户原话里的相对日期保留原词并加方括号日期（如 "a month ago [≈ 2023-04-24]"）。依据 dev100 错题分类：v1 时间推理错 9 题，7 题关键日期可由代码先算。零调用：train150 时间推理题 30/40、heldout100 16/27 在标注原话上得到日期；桩客户端中位上下文 5,735（v1 5,587）。判定：时间推理均值净 ≥ +2，总体 ≥ 0，无题型 < −2，中位上下文 ≤ 6,000 | [预注册](../results/prereg-raw-primary-time-notes-v1.md) | — |
| 原文为主 t2：新值优先＋计数逐条（heldout100，对 t1，每组 3 次） | 回归级开发对比（已暴露集） | **已预注册，未跑** | t1 加回答策略 `v2_cl`：所有题加"最近一次对话的值为当前值"；计数／求和题先逐条记下候选项并标 COUNTS/REPEAT/OUTSIDE 再计数（输出 1,024）。与 t1 同批跑。判定：多会话＋知识更新均值净 ≥ +2，总体 ≥ 0，无题型 < −2 | [预注册](../results/prereg-raw-primary-count-latest-v1.md) | — |
| 强回答模型探针（最小额度） | 诊断、按失败挑题 | **真实杠杆**（模型效应 +5） | v1 在 train150 上证据齐全仍答错的 9 题：`gemini-3.6-flash` 修好 5 题，flash-lite 重跑 0 题。手读确认为真修复；1 题评分可疑，剔除后仍 +4，达门槛 | [预注册](../results/prereg-strong-answerer-probe-v1.md) · [决策](../results/probes-2026-10-01-decision.md) · [结果](../results/analysis/strong-answerer-probe-v1.md) | 完整比较：免费档约 20 次/天，每组约一周或付费 |
| 偏好抽取探针 | 抽取诊断，不答题 | 不推进 | 偏好记忆挂到金标轮次 3/24 → 6/24（需 +6，实 +3），记忆量不变；任何记忆挂到金标轮次 14 → 21（未注册的旁证）；新偏好无编造 | [预注册](../results/prereg-preference-extraction-probe-v1.md) · [决策](../results/probes-2026-10-01-decision.md) · [结果](../results/analysis/preference-extraction-probe-v1.md) | 旁证若要追，需另行注册 |
| 强回答模型回归探针（10 题） | 诊断 | **风险较低**（按注册规则推进完整比较） | 强模型在 lite 答对的 10 题中答错 1 题（偏好题 `a89d7624`：给出通用丹佛攻略，没用用户的个人经历；手读确认为真回归）。噪声基准 0.59。推算净收益保守 −7.7、扣噪声 −0.2，**都不能证明净收益**：10 题只能排除大幅回归 | [预注册](../results/prereg-strong-answerer-regression-probe-v1.md) · [决策](../results/probes-2026-10-01-decision.md) · [结果](../results/analysis/strong-regression-probe-v2.md) | 完整比较需付费或多日；重点看偏好题是否被答成通用建议 |
| 原文为主 v2 对 v1（dev100，每组 3 次） | — | **运行前撤回** | 未跑；由 v4 取代（保留全部 v2 改动，另加两项零调用验证过的检索改进） | [预注册](../results/prereg-raw-primary-dev100-v3.md) | — |
| 抽取器模型对照 | 开发比较 | `STOP` | `3.5-flash-lite` 64.9% → 71.6%，但记忆数 +40%、每条记忆保住的具体信息 −21%；5 道门中 3 道失败。**同时测出抽取尺子的噪声：134 个具体信息 0 翻转** | [预注册](../results/prereg-extractor-model.md) · [决策](../results/extractor-model-decision.md) | 不换模型；要把「写得多」和「抽得好」分开，需要一把能看见幻觉的尺子 |

## v3 调整链

| 阶段 | 状态 | 说明 | 证据 |
|---|---|---|---|
| v3.1 reasoned | 调参持平 | tune42 与 v2 都为 66.7% | [报告](../results/validation/v3-phase3-tune1.md) · [预注册](../results/prereg-v3-phase3.md) |
| v3.2 adaptive | 停止 | 准确率方向为正，但上下文预算门失败 | [报告](../results/validation/v3-phase4-tune2.md) · [预注册](../results/prereg-v3-phase4-adaptive.md) |
| v3.3 compact | 调参门通过 | 保持 v3.2 正确率并把中位上下文从 1,476 降至 1,124 | [报告](../results/validation/v3-phase5-tune3.md) · [预注册](../results/prereg-v3-phase5-compact.md) |

## 支撑性实验

| 实验 | 状态与用途 | 证据 |
|---|---|---|
| 抽取批大小 | 完成；batch size 是因果变量，但**效应量依赖队列**（见[项目报告](../docs/PROJECT_REPORT.md)），且不能直接等同于 QA 提升 | [预注册](../results/prereg-batch-size.md) · [结果](../results/batch-size-result.md) · [v2b 决策](../results/v2b-gate16-decision.md) |
| 具体信息修复试点 | 通过 `PASS_TO_QA_GATE`；44.8% → 59.3%（+14.5pp），21 胜 0 负，25/60 触发修复，新增 0.33 条/会话，grounding 失败 0。**仅为抽取侧机制结果，不是 QA 结果** | [预注册](../results/prereg-specificity-repair-pilot.md) · [结果](../results/analysis/specificity-repair-pilot.md) · [工具](../tools/specificity_repair_pilot.py) |
| 检索权重扫描 | 完成；当前语料中仅语义排序优于加入 BM25、importance、entity 等组合 | [结果](../results/retrieval-weights.md) |
| 早期 heldout 方差 | 历史证据；用于说明重复运行波动，不作为当前终测 | [报告](../results/heldout-variance.md) |
