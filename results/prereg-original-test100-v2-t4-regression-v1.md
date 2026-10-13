# 原 test100：历史 v2 与修复后 t4 的同期完整回归

2026-10-10 Australia/Sydney。在本命名空间任何真实回答或评分调用前注册。
用户选择原来产生 72% 的题集，并授权完成后更新实测正确率、合成一条 commit 和 push，
不添加助手署名。期间不换模型，不重做抽取，不把未验证的摘要方案纳入本次候选。

## 前置条件和证据类别

必须先完成 `t4-scoped-regression-dev100-v3` 的全部 600 份回答及评分，且原注册 gate 通过。
失败或不完整都禁止本轮调用。核对父实验 execution SHA 和其全部冻结文件，不修改旧实验。

数据为公开 LongMemEval-S，原 `results/manifests/test100.json` 的全部 100 个题号，顺序不变。
此题集已暴露，证据类别为 **exposed-original-test100-regression**，不是未见终测。
原封存的 72% 单次成绩及原 Gold 不修改。本轮只做全量运行、预声明题型汇总及运行审计，
不看逐题错误调参，不挑题、不换 Gold、不增加轮次择优。

## 系统及身份

对照 `control`：历史 `two_stage_fallback`、`memory-aware-v2`。
工具 `restore_v2_baseline_source.py` 已从 Git 恢复冻结记录的全部 105 个源文件，逐文件 SHA256
一致；记录见 `test100-v2-source-restore-v1.json`。在独立目录和独立 Python 进程运行。
原捕获配置 `captured-v2.yaml`、原数据库、原记忆向量索引不变。
仅采用只读初始化适配，禁止原 store 的 schema/meta 写入；SDK 超时设为 180 秒。
历史客户端将 `max_retries=2` 解释为最多 2 次 API 尝试，另允许最多 2 次 transport 尝试
且 transport 重试不扣 API 计数，最坏共 3 次物理尝试。新版客户端将 2 解释为 2 次重试，
共享最多 3 次物理尝试。这是显式的恢复实现差异；没有修改请求内容或模型推理配置。

候选 `candidate`：`two_stage_raw_primary_t4` 加 `SavingsRepairRunner.from_t4`，
版本 `memory-t4-scoped-selection-v3`，与通过 dev100 的冻结候选完全相同。
题目和原记忆存储共用；原文按时间、实体及项目约束选择，必要时进行确定性计算。
没有新抽取、新框架或新数据库。

模型保持原 ID：answerer `gemini-3.5-flash-lite`，judge `gemma-4-31b-it`，
extractor `gemini-3.1-flash-lite`（本轮不调用），embedder MiniLM-L6-v2，384 维。
Temperature 0，服务不提供 seed；题集 seed 保持原 manifest。评分使用现有原 Gold、
`lme-type-aware-v2`，每条评分绑定已保存回答的 SHA256；没有评分缓存或自定义缺口重评分。
托管模型权重不能内容核验，两个版本同期运行以减小服务时间变化的混淆。

零调用 preflight：对照上下文中位数 535，新版 6368；新版核验全部 49,013 段原文的
用户、会话、角色、序号、完整文本及记忆来源归属，并核对原数据库逻辑 SHA 与历史冻结值一致。
`candidate-preflight-r2` 替代保留的初版 preflight：改动只涉及本轮统计、运行状态与 gate 的
完整性保障，候选算法没有改变。这些 stub 结果不作为正确率。

## 次数、分辨能力及决策

每组固定 3 次新回答和新评分：100 题 × 2 组 × 3 次 = 600 回答 + 600 评分。
顺序每轮 control 后 candidate；一个 block 的全部 100 份回答先保存，再评分。
已有完整回答和评分只用于本轮断点恢复，其他实验的回答或评分不得称本轮新结果。

`original-test100-v2-t4-regression-v1.resolution.json` 由既有重复证据和原 test100 的题型组成
零调用生成：约 10.93 道题可能随机翻转，单轮差异分辨下限约 7 题；3 次在独立假设下约 4。
这是粗略规划下限，不能当显著性检验。预计相对旧 v2 的总增益可能超过此下限，但不预设成绩。
100 题重复 3 次仍是 100 个独立问题，不能宣称样本量为 300 道独立题。

完整性要求：每个 block 恰好原 100 题，typed boolean verdict、固定提示版本及模型、
execution identity、reader SHA 绑定一致。缺失、外来或不匹配的行禁止发布正确率。

操作性 gate：

- 新版平均答对数减同期旧版平均答对数 >= 0。
- 任一题型平均净损失不得低于 -2 题。
- 新版实际供应的上下文中位数 <= 6500 token。

报告所有轮次、两组平均值、按题型净差、配对多数结果的 Bonferroni 97.5% Wilson
win/loss 区间组合出的 95% 差值区间。该区间针对每题多数结果，不是三轮平均值的区间。
只有区间下界 > 0 时才支持对应多数结果的可靠提升表述；操作 gate 通过本身不是统计证明。
历史 72% 和同期 control 必须分开列出；平均结果如实报告，即使失败也保留。

## 总资源与恢复

原记忆抽取、已有索引建设是两组共享的历史沉没成本，本轮不重复支出。
六轮离线上下文总量估算 2,072,244 token，尚不含完整提示、输出、fallback、评分及失败尝试。
600 回答，fallback 最坏另加 600 生成；两组每次生成最多 3 次物理尝试，最坏 3600。
Judge 每条最多 7 次持久恢复、480 秒累计冷却预算，每次生成最多 3 次客户端尝试，
最坏 14,400 物理尝试；不重跑已取得 verdict，包括 False。transport 与服务器瞬时故障
使用同一有限持久预算；配额不足、永久错误和预算耗尽停止，不能靠重启清空计数。
没有 reader 的额外过程恢复许可：客户端耗尽后记录 STOPPED，另行注册才可追加恢复。

每组分别报告按模型的调用数、失败数、服务返回的输入/输出 token；失败未返回 usage 时
其 token 未知，绝不视为证明零费用。报告回答全链路 p50/p95/max 耗时，包含检索和 fallback，
以及所有调用的实测 latency。token 是工作量，不是货币；不捏造账单。
按 500 次回答日配额，名义至少两天，服务失败与 fallback 可能增加天数，不能保证截止时间。

## 输出与冻结

`tools/run_original_test100.py` 与 `tools/test100_v2_worker.py` 执行和生成报告。
执行前将代码、测试、配置、manifest、数据库/索引、恢复的历史源码、前置 gate、
preflight、resolution 及本注册录入 `results/analysis/original-test100-v2-t4-regression-v1.execution.json`，
并保存源码快照、Git SHA、Python 版本、模型和操作预算。
所有 resume 先核验冻结哈希，变更则停止，不复写 frozen/sealed 历史。

新结果写入 `results/raw/original-test100-v2-t4-regression-v1-{control,candidate}-rep{1,2,3}`；
gate JSON/Markdown 只在 600 对完整结果后生成，progress 为独立可恢复状态。
完成后按实际结果更新 README、项目报告、评测说明及发布展示，保留历史成绩和证据类别。
之后运行完整验证，按用户授权合成一条 commit 并 push，使用已配置的人类 Git 身份，
不添加助手署名或 Co-authored-by。
