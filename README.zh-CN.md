[![English](https://img.shields.io/badge/English-555555?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-2962FF?style=for-the-badge)](README.zh-CN.md)

# ChronoMem

面向 LLM Agent 的自适应长期记忆引擎——时序事实消解、记忆合并、带衰减的混合检索、
以及 token 预算下的上下文组装，在
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) 上评测。

> **状态：P4 已测量，而且输了。** 下表四行都是真实运行的结果。记忆系统 26%，对朴素
> RAG 的 54%，原因已诊断清楚：抽取丢掉了问题所问的具体信息，时间线因此无从排序。
> 下一个约束是抽取保真度，不是排序（[D26](docs/DECISIONS.md)）。

## 为什么做这个

向量库加 top-k 检索能回答「用户说过什么」，但回答不了「现在什么是真的」。比如：

```
一月   我用 TensorFlow。
三月   我在学 PyTorch。
八月   我已经完全转用 PyTorch 了。
```

朴素 RAG 会把三条都捞出来，让模型自己猜。ChronoMem 把它们消解成一条带有效期的
active 事实和两条已失效事实，再决定其中哪些值得花上下文窗口的 token。

## 结果

LongMemEval-S · 50 题分层子集（seed 0）· answerer `gemini-3.5-flash-lite` ·
judge `gemma-4-31b-it`，两者在所有行中锁定不变。延迟为 API 时间，不含免费层限流排队。
每个数字都从 `results/raw/` 的 JSONL 产物重新生成。

| 变体 | n | 准确率 | Temporal | Know-update | Abstention | 证据召回 | ctx tokens | p95 |
|---|---|---|---|---|---|---|---|---|
| `full_context` | 50 | **56.0%** | 23.1% | 87.5% | 50.0% | — | 109,260 | 7.9s |
| `naive_rag` | 50 | **54.0%** | 46.2% | 75.0% | 100% | 94.0% | 13,057 | 2.1s |
| `chronomem_no_temporal` | 50 | **26.0%** | 7.7% | 37.5% | 100% | 80.0% | 331 | 7.0s |
| `chronomem` | 50 | **26.0%** | 7.7% | 62.5% | 100% | 80.0% | 465 | 1.2s |

**记忆系统输了,而且原因不在 P4 瞄准的地方。** 26% 对朴素 RAG 的 54%,时序消解没有
可检测的差异(3胜3负,p = 1.000)。

失败不在检索。证据 session 的记忆在 **50 题里有 40 题**被召回,而这 40 题里只有
**12 题答对**。**50 题里有 28 题**回答「我不知道」。正确的记忆进了 prompt,答案不在
记忆里。

一个完整追踪的例子——*「我收集古董相机多久了?」*,gold 是 `three months`。证据
session 产出的三条记忆,排名第一:

```
- 用户拥有 17 台古董相机,包括 2023 年 5 月入手的 Brownie Hawkeye。
- 用户有一张 1978 年版的 Fleetwood Mac《Rumours》。
- 用户有一张霍格沃茨的 Mondo 海报。
```

**时长从未被抽取出来。** 下一题(`25` 张明信片)以同样方式失败,模型答了 `17`——
上下文里最近的那个数字。

这是覆盖率关卡在端到端上的兑现。它在任何评测跑之前就测出 50% 的答案覆盖率,并被
写作上限;26% 是检索和推理在此之上还剩下的部分。**排期判断错了**:P4 之所以被提前,
是因为时序推理是两个基线最差的分项——那是把症状当成了诊断。时序题需要时长和日期,
而那恰恰是抽取丢掉的东西。时间线机制本身是对的(35 次 supersede、手工核对过链条、
16 个测试),但在它排序的表示还不包含答案之前,它不可能产生收益。下一个约束是抽取
保真度,见 [D26](docs/DECISIONS.md)。


用 `chronomem eval report` 重新生成；含 single-session-assistant 和 preference
细分的完整表格在 [results/table.md](results/table.md)。

### 该看配对检验，不是准确率那一列

同一份配置原样重跑两次，分别得到 **48.0%** 和 **54.0%**。`temperature=0` 并不能让
托管模型变得确定，judge 的边界判定也会移动；50 题里翻转 4 条就是 8 个百分点。
**任何小于这个幅度的差距都不构成证据**，这就排除了在当前样本量下直接比较总准确率。

但两个变体回答的是同一批题，所以这是配对数据，共有的噪声可以被消掉。
`chronomem eval compare` 只在分歧项上做精确 McNemar 检验：

```
full_context vs naive_rag
  都答对   19        naive_rag 胜   8
  都答错   14        naive_rag 负   9
  17 条分歧，p = 1.000  ->  检测不到差异
```

**所以真正的结论不是「全上下文赢了两分」，而是 109,260 个 token 相对 13,057 个
什么都没买到。** 8.4 倍的上下文成本，两个系统在 50 题里有 17 题分歧，而且两个方向
数量相当。

这比赢一点点是更强的结果，而且它让目标更清晰了：天花板不是「更接近全上下文」，
因为全上下文并不高于朴素检索。两者都在 54–56%，而全上下文在时序推理上**更差**——
**23.1% 对 46.2%**——这是唯一一个大到能穿透噪声底、且方向有意义的分项差距。
把 10.9 万 token 未经整理的历史丢给模型，反而让它更判断不出哪条事实是当前有效的。
这正是 P4 当初要填的缺口——而上面的测量显示它没有填上，原因与时间线无关。

另有两个分项差距大到值得追查：

- **Abstention，100% 对 50%。** 上下文稀疏时模型能可靠地说「不知道」；把整段历史
  摆在面前，它有一半时间开始编。任何往上下文里塞进更多相关材料的变体都可能把这个
  指标换掉，所以这一列一直单独显示，不并进平均值。
- **Single-session-user，100% 对 71.4%。** 检索丢掉了全上下文拥有的证据。证据召回
  是 94%，所以这是缺的那 6% 加上排序问题，不是结构性缺陷。

这些是 ChronoMem 自己的变体与两个基线的对比，**不构成**对任何第三方系统的断言：
跨系统的记忆数字只在同一个 judge 和同一套 prompt 下才可比，而已发表的结果并不满足
这个条件。

**judge 的可靠性不做假设。** 免费层没有比 answerer 更强的模型可用来判分，因此判分
结果与一份独立标注在全部 50 题上做了交叉核对：**94% 一致（n=50）**——1 条 judge 比
标注更宽松，2 条更严格。没有系统性偏向，这比一致率本身更重要。

两个前提如实写出而非埋掉。第一，标注是由 LLM 而非人完成的，所以它证明的是评分标准
可复现，不是 judge 判得对。第二，第一遍得到的是 92%，而其中一条分歧是**标注方自己
的错误**：它读的是被截断到 220 字符的答案，漏掉了最后一句里的结论。现在标注流程改为
读全文。剩下三条是真正的边界判断——拼错的 app 名（`Memorse` 对 `Memrise`）、一个数字
正确但把「计划购买」也算进去的计数、以及一个在 gold 只描述了「未点名的乐队」时给出
具体艺人名的回答。

**这里每个数字都从 JSONL 产物重新生成**，而不是取自控制台输出。当两者不一致时
`run_eval` 会拒绝返回结果：此前有一次运行在只有 30 条记录的文件上打印了
`50 questions, 56.0%`，而两个数字看上去都很合理（[D25](docs/DECISIONS.md)）。

## 抽取

写入路径把 session 变成带类型、三元组形式、附有效期的记忆。两道关卡守着它，
而且都已经抓到过真实缺陷：

**答案覆盖率**（`chronomem ingest coverage`）在只含证据的子集上抽取，检查 gold 答案
是否还留在记忆里。它发现第一版抽取 prompt 把 benchmark 真正要问的具体信息概括掉了
——`The Glass Menagerie` 变成了「对表演感兴趣」——改写 prompt 后可测覆盖率从
**26.3% 升到 50.0%**，`single-session-user` 从 1/3 升到 3/3。这个指标是严格的下界，
文档里也这么写：LongMemEval 的答案大多是从已存事实**算出来的**，而不是直接陈述的。

**去重的元数。** 嵌入相似度只是召回过滤器，真正判定 DUPLICATE / UPDATE / DISTINCT
的是 LLM——因为「喜欢 Python」和「不喜欢 Python」的余弦相似度约 0.95，没有任何阈值
能把它们分开。把 `(subject, predicate)` 碰撞限制在单值谓词上，使裁决调用从每 60 个
session **72 次降到 3 次**，同时抓到的重复反而**增加**了。

## 时序消解

每条事实都带有效期。对每个 `(subject, predicate)` key，消解器按事件时间排序并重写
整条链，于是 `TensorFlow` 在 `PyTorch` 开始的那天被关闭，只剩一个取值仍然生效。

**重建时间线而不是两两比较**是这里最关键的选择。session 是按任意顺序摄入的，所以
「刚到的记忆 supersede 已经在的那条」会让一条晚到的*一月*事实变成当前值；而且一条
记忆一旦被 supersede 就不再是链头，因此后来落在两条之间的事实，永远改不了那条已经
跨过它的链接。重建则是幂等的、与顺序无关、可自我修正，而且不消耗任何 LLM 调用。

写测试时浮现出的两点细化：

- **重述不是变更。** 「我住堪培拉」在三月和六月各说一次，合并成一个区间且归属*三月*，
  这样「你什么时候搬的？」才答得对。计为 `restatements`，不计入 `superseded`。
- **元数是默认规则，不是全部规则。** `uses_tool` 确实是多值的——用 PyTorch 不妨碍用
  NumPy——所以纯静态谓词表会让开篇那个例子完全不被消解。因此抽取器在用户说
  「我切换到 X 了」时额外输出 `replaces_previous`，不增加任何请求成本，两个信号任一
  成立即可消解一个 key。

第一次真实摄取立刻证明了那张表是错的。`lives_in` 产出
`东京 → 南湾 → 拉斯维加斯`，这是对的；`scheduled` 产出
`伦敦中转 → 烹饪课 → 9:15 的火车 → 周五游戏夜`，这根本不是一串互相竞争的取值。
`scheduled` 和 `has_goal` 被移除。

这个错误只花了几秒而不是一天，而且是刻意如此：消解读取已存数据、不调用模型，所以
`chronomem resolve` 能原地重建每一条时间线。**元数是谓词的属性，不是数据的属性**，
把它挡在摄入产物之外，才使得「搞错」变得便宜。

## 快速开始

```bash
uv sync --group dev
uv run chronomem data download --variant s
uv run chronomem data stats --variant s
uv run chronomem data plan --variant s

uv run chronomem doctor
uv run chronomem ingest coverage --n 20      # 抽取有没有把答案留住？
uv run chronomem ingest run                  # 构建记忆库（可断点续传）
uv run chronomem eval run naive_rag
uv run chronomem eval report
```

`data plan` 会报告在你的 API 配额下，不同批大小各需要几天才能完成一次完整摄取——
在按请求数限流的免费层上，这是排期而不是成本。

## 设计

详见 [docs/DECISIONS.md](docs/DECISIONS.md)——选了什么、否掉了什么、以及测量说明了什么。

| 层 | 选择 |
|---|---|
| 存储 | SQLite（WAL）+ FTS5 提供 BM25——单文件，无需额外服务 |
| 向量 | numpy 上的精确内积暴力搜索 |
| Embedding | `all-MiniLM-L6-v2`，本地运行，MPS 加速 |
| LLM | 经 `google-genai` 调用 Gemini，三角色配置（extractor / answerer / judge） |
| Benchmark | LongMemEval-S，配可断点续传、配额感知的摄取管线 |

## 目录结构

```
src/chronomem/
  store/       schema.sql、SQLiteMemoryStore、NumpyFlatIndex
  llm/         配额感知限流器（RPM / TPM / RPD）、带重试的客户端
  ingest/      抽取、去重、断点续传管线、覆盖率关卡
  evaluation/  benchmark 加载器、runner、judge、报告
  config.py    每个 ablation 变体一个 YAML
  cli.py
tests/
docs/DECISIONS.md
```

## 许可

MIT
