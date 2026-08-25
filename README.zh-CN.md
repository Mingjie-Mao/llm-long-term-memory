[![English](https://img.shields.io/badge/English-555555?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-2962FF?style=for-the-badge)](README.zh-CN.md)

# llm-long-term-memory

**面向 LLM 应用的持久化长期记忆层。**

它把对话压缩成紧凑的记忆、在事实发生变化时更新它们，并在压缩恰好丢掉了问题所需的细节时，
回到原始对话把细节找回来。

---

## 一次对话就能说清的问题

三个月前助手回答过一个问题。今天用户问起它：

> **问：** 你之前推荐的那个 Mayo Clinic 视频是什么？

记忆库里关于 Mayo Clinic 什么都没有——抽取保留了大意，丢掉了链接。向量库到这里就失败了，
而且会一直失败下去。

```
结构化记忆        不足
原始对话回退      已触发 —— 全档案检索
证据来源          session answer_sharegpt_81riySf_0 · turn 1 · assistant

答案
  "How to Sit Properly at a Desk to Avoid Back Pain"
  https://www.youtube.com/watch?v=UfOvNlX9Hh0
```

压缩必然有损。这个设计接受这一点，只要求损失是**可恢复的**。

---

## 架构

```
                     Conversation
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
        原始对话归档              抽取器
        （turns 表）        （每批 2 次 LLM 调用）
              │                       ↓
              │                 结构化记忆
              │            带类型 · 双时间 · 锚定出处
              │                       │
  Query ──────┼───────────────────────┘
              │      混合检索（建了 5 路信号，
              │       线上配置只给 1 路权重）
              │                       ↓
              │                 够回答吗？
              │              ┌────────┴────────┐
              │             够                不够
              │              ↓                  ↓
              └──────────> 回答          回原文检索
                                                ↓
                                          带出处的回答
```

| | |
|---|---|
| **记住** | 把对话变成带类型、带时间窗的事实——谁说的、说的是谁、为什么值得留 |
| **更新** | "我搬到悉尼了"会取代"我住在堪培拉"，但不删除它 |
| **检索** | 五路加权信号——语义、BM25、时近、重要性、实体重叠。**线上配置只给语义一路权重**（[原因](#局限)） |
| **恢复** | 结构化记忆不足时，检索原始对话，而不是直接失败 |
| **解释** | 每条记忆都能追回来源轮次；每个**遗漏**都有明确理由 |

最后一行是这个项目和普通记忆库最不一样的地方。检索返回的不只是命中了什么，还有**什么被
淘汰了、为什么**——`superseded`（这条事实不再成立）或 `below_rank`（仍然成立，分数没进前列）。
一个没有理由的"没有结果"，跟一个 bug 长得一模一样。

**原文归档是一个存储决策，不是一个检索算法。** 保留原始轮次是有损抽取得以恢复的前提；
*怎么找到*它们是一个独立的选择。

| 层 | 选择 | 为什么 |
|---|---|---|
| 存储 | SQLite（WAL）+ FTS5 | 一个文件，无服务依赖；BM25 免费拿到 |
| 向量 | 精确扁平内积（numpy） | 几万条记忆；ANN 索引只为省掉微秒，却多一个依赖 |
| 嵌入 | `all-MiniLM-L6-v2`，本机，384 维 | 全语料嵌入约 5,300 万 token；API 额度才是绑定约束 |
| LLM | 三个角色，三个 model id | `gemini-3.1-flash-lite` 抽取，`gemini-3.5-flash-lite` 回答，`gemma-4-31b-it` 判分。固定版本，绝不用 `-latest`——浮动别名会让不同周次的运行悄悄变得不可比 |
| 服务 | FastAPI，单一组装根 | REST、MCP 和 Inspector 都是同一个 service 对象的客户端 |

三个独立的 model id 而不是一个：判分器不能给写出这段文字的模型打分，而且免费额度是
**按模型**计量的，分开才能让一次大规模摄入不饿死评测。四个模型全部固定在
[`configs/fallback.yaml`](configs/fallback.yaml)。

---

## 看它跑起来

| Demo | 展示什么 |
|---|---|
| [`/?demo=mayo`](http://localhost:8000/?demo=mayo) | 压缩丢掉了 URL，档案把它找了回来 |
| [`/?demo=battery`](http://localhost:8000/?demo=battery) | 记忆足够——回答**运用**了记住的偏好，没有回退 |
| [`/?demo=timeline`](http://localhost:8000/?demo=timeline) | 同一个事实的五次变化，以及取代链 |
| [`/?demo=collectibles`](http://localhost:8000/?demo=collectibles) | 多值键：这些事实并存，谁也没有取代谁 |

每个打开的都是**录制的运行结果**——真实模型的真实输出，明确标注为录制，所以打开页面不消耗
额度；**Run live answer** 可以按需重跑。录制结果带有产生它的库和 prompt 版本指纹，任何一个
变了就显示 **stale**，而不是冒充当前行为。

---

## 结果

两个数字最重要。第一个是唯一一个在"从未参与任何决策"的数据上测出来的。

**保留集：100 道来自 [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) 的未见过的题，在冻结并哈希过的系统上跑一次：**

| | |
|---|---:|
| **最终正确率** | **70.0%** |
| 只靠结构化记忆答对 | 50.0% |
| 被原文档案救回 | 20.0% |
| 中位上下文 token | 1,468 |

同一配置又跑了两次，得 71 和 73，所以区间是 **70–73，均值 71.3**。对外报告的仍是
`70.0%`，因为协议在重复运行存在之前就承诺了报告这一枪。

**开发集，带基线——`dev50`，所有设计决策都是在这里做的：**

| 变体 | 正确率 | 中位上下文 token |
|---|---:|---:|
| `full_context` —— 整个聊天记录 | 56.0% | 109,260 |
| `naive_rag` —— 对原始会话做向量检索 | 54.0% | 13,057 |
| 只有结构化记忆 | 56.0% | 1,415 |
| **结构化记忆 + 条件回退** | **72.0%** | **1,455** |
| v1 结构化记忆 | 26.0% | 465 |

**上下文只有转录的 1/75，在这个集合上还更准。**

两条必须紧挨着这些数字的提醒，不放在脚注里：`dev50` 被用来做 prompt 迭代和阈值调参，
所以它的 72.0% 衡量的是这些选择的拟合程度，不只是系统本身；而且**基线从未在保留集上跑过**，
所以"结构化记忆打平 naive RAG"在未见数据上没有任何证据。
[完整依据](#依据) · [局限](#局限)

---

## 2 分钟跑起来

```bash
docker compose up --build
```

```bash
curl localhost:8000/healthz
```

然后打开 <http://localhost:8000> 看 Inspector。

镜像里不烘焙任何数据集、模型、凭据或数据库——库从挂载卷进来。`GEMINI_API_KEY` 是可选的：
没有它服务以只读模式运行，并且会明确说出来。

<details>
<summary>镜像体积，以及一个更小的构建</summary>

**2.95GB**，大部分是嵌入模型需要的 PyTorch。默认的 PyTorch 安装会拉 CUDA wheel——24.4GB
的 GPU 运行时，装在一个永远见不到 GPU 的容器里——所以 Dockerfile 装 CPU wheel 并删掉替换后
孤立的 CUDA 包。这仍然不小；真要小，应该把编码器放到进程外运行，那件事在
[路线图](docs/ROADMAP.md)上。

如果只要一个不带语义搜索的只读镜像，用 `--build-arg EXTRAS="api"` 构建。此时 `/healthz`
报告 `degraded`，搜索返回 503，而不是以某种难懂的方式失败。
</details>

<details>
<summary>不用 Docker</summary>

```bash
uv sync --group dev --extra api --extra embed
uv run uvicorn llm_long_term_memory.api.app:app --port 8000
```
</details>

---

## 5 分钟接进你的应用

```bash
curl -X POST localhost:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "alice", "role": "user", "content": "I stopped drinking coffee last month."}'
```

```bash
curl -X POST localhost:8000/v1/memories/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "alice", "query": "what does she drink?", "explain": true}'
```

<details>
<summary><code>search</code> 返回什么，以及完整端点列表</summary>

```json
{
  "memories": [
    {
      "content": "The user stopped drinking coffee.",
      "scope": "profile",
      "source_role": "user",
      "score": 0.81,
      "signals": {"semantic": 0.79, "bm25": 0.92, "recency": 0.4, ...},
      "source": {"session_id": "s_4f2a", "turn_index": 0, "char_start": 0, "char_end": 41}
    }
  ],
  "rejected": [
    {"memory_id": "m_9c1", "reason": "superseded", "superseded_by": "m_a20",
     "content": "The user drinks two coffees a day."}
  ],
  "candidates_considered": 37
}
```

| 端点 | |
|---|---|
| `POST /v1/messages` | 摄入一轮对话；返回它产生的记忆 |
| `POST /v1/memories/search` | 检索，带信号、出处和被拒记录 |
| `POST /v1/answer` | 完整回答路径，必要时含回退 |
| `POST /v1/raw/search` | 回退层，可直接查询 |
| `GET /v1/memories` | 浏览一个命名空间；可按状态/类型/scope/说话人过滤 |
| `GET /v1/memories/{id}` | 一条记忆及其来源轮次 |
| `GET /v1/timeline` | 某个 `(subject, predicate)` 的取代链 |
| `DELETE /v1/memories/{id}` | 遗忘——标记为 evicted，绝不硬删除 |
| `GET /healthz` · `GET /v1/config` | 健康状态与运行中的清单 |
</details>

每次读都必须带 `user_id`，并且按命名空间隔离。读另一个命名空间的记忆返回 **404 而不是 403**
——403 等于确认这个 id 存在。

---

## 从 MCP 客户端接入

[MCP](https://modelcontextprotocol.io) 是 agent 接入它的方式，REST 是程序接入的方式。
两者都包着同一个 service 对象，所以工具的行为不可能和被测量的行为发生漂移。

```bash
uv sync --extra mcp --extra embed
uv run lltm mcp                    # stdio
uv run lltm mcp --transport http   # streamable HTTP
```

<details>
<summary>Claude Desktop / Cursor 配置</summary>

```json
{
  "mcpServers": {
    "long-term-memory": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/llm-long-term-memory", "lltm", "mcp"]
    }
  }
}
```
</details>

| 工具 | |
|---|---|
| `search_memory` | 回忆已知的事，附每条记忆被选中的理由——加上 `explain` 还能看到其它记忆为什么没被选 |
| `remember` | 存一轮对话；返回它产生的记忆 |
| `search_conversations` | 当记忆切题但缺少细节时，找回原始措辞 |
| `get_timeline` | 一个事实随时间如何变化 |
| `forget` | 标记为 evicted，绝不硬删除 |

每个工具都要显式的 `user_id`。没有隐式会话身份：一个服务多个人的 agent 必须说清楚它此刻
在为谁行动，而边界由存储层强制，不靠信任调用方。

---

## 依据

### 保留集那次运行

100 道从未参与任何决策的题，在[冻结并哈希过的系统](results/frozen/p10-final/README.md)上
**跑一次**，前面有五道闸门——其中一道断言结果文件尚不存在。

| | dev50 —— 所有决策都在这里做的 | **heldout100 —— 未见过** | |
|---|---:|---:|---:|
| **最终正确率** | 72.0% | **70.0%** | -2.0pp |
| 只靠结构化记忆答对 | 54.0% | 50.0% | -4.0pp |
| 被原文档案救回 | 18.0% | 20.0% | +2.0pp |
| 回退触发 | 32% | 36% | +4pp |
| ……并且答对了 | 56.2% | 55.6% | -0.6pp |
| 来源会话召回 | 93.5% | 94.0% | +0.5pp |

**标题数字泛化了，它下面几乎没有一样东西泛化。**

对三次运行的均值，与 dev50 的差距是 **-0.7pp** 而不是 -2.0pp：那一枪落在了自己区间的底部，
所以系统的泛化能力比预注册的数字说的更好。100 道题里有 6 道在三次运行之间不一致，
而且**六道全都是 `temporal-reasoning` 或 `multi-session`**——答案需要**推导**而不是查找的
那两类。另外四类三次运行逐位一致。[测量过程](results/heldout-variance.md)。

| 题型 | dev50 | heldout100 | Fisher p |
|---|---:|---:|---:|
| knowledge-update | 8/8 = 100% | **10/15 = 66.7%** | 0.122 |
| single-session-user | 7/7 = 100% | 12/14 = 85.7% | 0.533 |
| single-session-assistant | 6/6 = 100% | 9/11 = 81.8% | 0.515 |
| multi-session | 7/13 = 53.8% | 19/27 = 70.4% | 0.480 |
| single-session-preference | 2/3 = 66.7% | 4/6 = 66.7% | 1.000 |
| **temporal-reasoning** | 6/13 = 46.2% | **16/27 = 59.3%** | 0.509 |

`dev50` 每一格只有 3 到 13 道题。它的三个 100% 类别全部下跌，两个最差的类别全部上升，
没有一处能与噪声区分——**`dev50` 的分题型数字从来没有携带过信息**，而在它们被用来决定该
建什么的时候，没有人把这句话说出口。两处变化改变了下一步该做什么：

- **`temporal-reasoning` 是 59.3%，不是 46.2%。** 仍然是最差的类别，但不再是这个 README
  以前说的那堵墙。旧估计建立在 13 道题上。
- **`knowledge-update` 从 100% 跌到 66.7%**，现在并列最差，而在 `dev50` 上完全看不出来，
  因为 8 中 8 不是证据。五个失败全部召回了 gold session **和** gold evidence；四个发生在
  下游，在"拿到的事实如何被读取和组合"这一层。

有一道题的证据被服务商的内容过滤拒绝了。排除它之后，99 道题上的数字是 69.7% / 49.5% / 20.2%。

`heldout100` 已经花掉了。不允许再针对它调任何东西；`dev100` 在这个结果被读到之前就已冻结。

### 提升来自哪里

早期的草稿把功劳记给了"每次都附原文证据"。一次因果消融把这个结论推翻了：

| 步骤 | Δ | 配对 |
|---|---:|---|
| 抽取重写 | **+29.0pp** | 11 胜 2 负，**p = 0.022** |
| 常开的证据挂载 | +3.2pp | 2 胜 1 负，p = 1.000，代价 3 倍上下文 |

两臂的来源会话召回都是 93.5%，这佐证了上面的判断——挂载发生在检索之后，不可能改变召回了什么。
所以挂载从一个常开的阶段降级成了条件回退。

### 72% 是由什么构成的

54.0% 来自结构化记忆本身，18.0% 是回答器声明答不了之后由档案救回来的。
**记忆单独跑正好打平 `naive_rag`**——把产品拉到基线之上的是档案。整个引用 72.0% 而不拆分，
读起来会像是记忆层答对了全部。

两个臂只在三个 `fallback` 配置项上不同，其余完全一致——这是运行前检查过的，不是事后声称的。
在同样 50 道题上配对，回退修好 9 道、弄坏 1 道，精确 McNemar **p = 0.022**。

> **这个 p 值不成立。** 它是每臂**单次运行**，而且是在同一批暴露了那个缺陷的 50 道题上测的
> （缺陷修复把该臂从 66.0% 抬到 72.0%）。重复实验测出每 100 题有 6 题会在相同配置下翻转；
> 50 道题上**一次翻转**就把 9 胜 1 负变成 8 胜 2 负、`p = 0.109`。效应量很大，大概率是真的。
> 但发表出去的显著性超出了单次运行能支撑的范围。

### 测过但没有上线的

**交叉编码器重排。** k=10 时重排臂和普通臂对 31 道题给出了**完全相同的答案**——零处分歧——
同时在每个 k 上都**降低**来源召回（k=20 时 93.5% → 90.3%）。保留在可选的 `rerank` extra 后面，
默认关闭。[详情](results/rerank-pareto.md)。

**对原文档案做稠密检索。** 一组构造出来的查询显示 BM25 在改述面前崩溃（R@1 只有 6.9%）。
人工检查发现构造过程删掉的是**问题本身**，不只是它的词汇——*"How long have I been collecting
vintage cameras?"* 变成了 `"long"`。那份证据被作废；手写的改述能把 gold 轮次检索到**第 1 位**。
**推迟，不是否决。**[详情](results/raw-recall-diagnostic.md)。

**为打包预算学一个效用预测器。** 保留集 RMSE 0.310，而"直接预测均值"是 0.263——它输给了
常数基线。[详情](results/p6-pilot.md)。

### Live 回归

七道此前每个变体都答错的题现在是 **6/7，三次运行完全一致**，包括 Mayo 那条端到端的路径。
[详情](results/live-regression-v2.md)。

<details>
<summary>干净的库一度让 Mayo 案例失败，而那个显而易见的诊断是错的</summary>

一次**改善了检索**的重建之后，这个 README 开头那道题在两个正式臂里都失败了。诱人的解释是
级别分支——`RawFallback.recover` 只要检索返回了任何东西就取来源轮次，只有检索为空才去档案，
没有任何东西检查这些记忆是不是关于这个问题的。这**是**一个真实缺陷，但**不是这一个**。

真正让它坏掉的是截断。检索**确实**找到了正确的对话。但 `turns_for_memories` 返回的轮次是
**按 session id 排序**的，代码取了 `[:max_turns]`，而 gold 轮次在字母序里排第十（共十六条）
——保留下来的那三条全部来自一段关于现场音乐的对话。**召回变好意味着候选变多，候选变多意味着
答案被切掉了。**在更小的混合库上，同一道题检索不到东西、直接落到档案、从没遇到那个切片，
这就是这个缺陷长期没被发现的原因。

两个缺陷是同一个疏漏——没有任何东西拿候选去和问题比较——所以对候选排序同时修好了两个。
`single-session-assistant` 从 5/6 变成 **6/6**，该臂从 66.0% 变成 72.0%。
</details>

---

## 失败分析

不是每一个错误答案都是检索失败。把 dev50 全部 14 个失败追到丢失答案的那一层：

| 阶段 | n | |
|---|---:|---|
| S0 来源 | 0 | 这个事实压根不在对话里 |
| **S1 抽取** | **10** | 它从来没变成一条记忆 |
| S2 生命周期 | 1 | 变成了记忆，然后被合并或取代掉了 |
| S3 资格 | 0 | 排序之前被过滤掉了 |
| **S4 检索** | **0** | 有资格，但没进上下文 |
| S4b 组装 | 1 | 进了上下文而且排最前，仍然没被用上 |
| S5 推理 | 2 | 全部供上而且正确，答案还是错 |

**十四个里十个死在抽取，检索一个都没有。**然后每个模块在被建之前先量一个 oracle 上限——
把缺失的事实注入进去，跑真实流水线，数有多少题改变：

| 模块 | 上限 | 实测 |
|---|---:|---|
| 抽取 | 10 | 试了 9 道，修好 4 道 |
| **检索** | **0** | 没跑——没有失败死在这里 |
| 组装 | 1 | 4 次运行里 3 次修好 |
| 推理 | 2 | 0 |

抽取的 oracle 只修好 9 道里的 4 道，这是更有用的那一半：缺失事实之外还有第二个缺陷，
所以光把抽取做到完美也到不了那个上限。[详情](results/failure-stages.md)。

**批量抽取按位置丢记忆，而且这是因果测量出来的。** 同一段对话、同样的批大小、同样的 prompt、
同样的邻居——挪到请求后面产出只剩三分之一（4.5 → 1.6 条/会话，会话内配对，精确 McNemar
p < 0.0001）。批大小为 1 时零产出降到 0.0%、产出升到 12.7 条，代价是 12 倍的请求数。
目前**没有**据此改动任何东西：这个 README 里的每一个数字都是在批大小 15 下产生的。
[详情](results/batch-position-pilot.md)。

---

## 工程

- **529 个测试**，CI 覆盖 ubuntu / windows / macos，行覆盖 80%
- **结果版本化**——每行评测记录都带回答 prompt、判分 prompt 和抽取器版本；抽取器版本取自
  **库**而不是取自代码检出，因为它描述的是被评测的数据
- **摄入指纹**——续跑时比对一个由 prompt、schema、模型 id、批大小和去重阈值算出的哈希，
  并报出**哪个部件动了**。这是在发现某个库曾被两代抽取器写入、而没有任何东西能察觉之后加的
  （[事故记录](docs/ENGINEERING_REPORT.md)）
- **冻结清单**——报告的运行必须指名一个明确的问题集合，因为样本量不是实验身份
- **闸门非零退出**——一个打印出判决然后退出 0 的闸门不是闸门
- **结构化日志**——每请求一条 JSON，含 request id、延迟和配置指纹；不含记忆内容，不含凭据
- **编码器预热**：首个请求 12,486 ms → 181 ms

完整记录：[系统设计报告](docs/REPORT.zh-CN.md) · [设计决策](docs/DECISIONS.md) ·
[路线图](docs/ROADMAP.md) · [工程报告](docs/ENGINEERING_REPORT.md) ·
[English](docs/REPORT.md)

---

## 开发

```bash
uv sync --group dev --extra api --extra llm --extra embed
uv run pytest
uv run lltm --help
```

```
src/llm_long_term_memory/
  api/         FastAPI 服务、Inspector、录制的 demo
  store/       schema.sql、SQLiteMemoryStore、NumpyFlatIndex
  ingest/      两阶段抽取、去重、带检查点的流水线、质量闸门
  retrieve/    混合检索、原文回退、重排器（可选）
  evaluation/  基准加载、runner、判分器、清单、报告
  llm/         配额感知的限流器（RPM / TPM / RPD）、带重试的客户端
```

<details>
<summary>基准测试流程</summary>

```bash
uv run lltm data download --variant s
uv run lltm doctor
uv run lltm ingest run --store-name two-stage-p10
uv run lltm eval freeze dev50 --n 50
uv run lltm eval run two_stage --questions results/manifests/dev50.json
uv run lltm eval compare naive_rag two_stage
```

`--questions` 接一个冻结的清单文件。`--limit` 会重新采样，只用于探索——分层子集是散布在
整个 split 里的，不是它的前缀。
</details>

---

## 局限

- **保留集上没有基线。** `heldout100` 只跑了本系统，所以 `70.0%` 没有任何未见过的对照点，
  "结构化记忆打平 naive RAG" 是一个 `dev50` 上的说法。**这是证据链里最大的缺口。**
  补上它是下一轮第一个注册的臂，在 `dev100` 和 `test100` 上。
- **这里每一个配对 p 值都是每臂单次运行，那太少了。** `heldout100` 三次运行里有 6/100 翻转。
  重复运行现在是协议（[data-protocol.md](results/data-protocol.md)）；2026-08-20 之前发表的
  任何结果都不遵守它。
- **`dev50` 的分题型数字没有携带信息，但它们仍然被使用了。** 每一格只有 3 到 13 道题。
- **`knowledge-update` 未见数据上 66.7%，`dev50` 上曾是 100%。** 五个失败全部召回了 gold
  session 和 gold evidence，四个发生在检索的下游。库里有一个形状吻合的症状——622 条记忆带着
  `replaces_previous` 而只有 42 条真的被取代，另有 2,275 条的 predicate 是 `none`——
  **但这个统计尚未被证明是这些失败的原因。**
- **抽取器是已知有损的，而且没有改。** 批大小 15 每会话产出 2.6 条记忆，批大小 1 是 12.7 条。
  这里所有数字都在那个天花板之下。
- **14.5% 有实质内容的会话一条记忆都没产出**（304 / 2,096）。记录为基线，尚未解释。
- **五路检索信号里有四路权重是 0。** 混合检索已实现、已测试；线上配置只给 `semantic` 权重。
  另外四路是**未测量**，不是被否决。
- **`stores/two-stage-hydrated.db` 混了两代抽取器**——4,843 行在修复前，2,265 行在修复后。
  在它上面测的结果一律标注为诊断。
- **单写的 SQLite**，加了跨进程锁——那是在两次同时进行的摄入互相覆盖了配额账目之后加的。
- **时序算术尚未解决**，回退也碰不到它——日期必须被计算，不能被查找。
- **还不是一个产品。** 没有可信身份（`user_id` 取自请求正文）、没有硬删除、没做过恢复演练、
  没有费用核算。各项的验收标准见[产品化计划](docs/PRODUCTIZATION_V2_PLAN.md)。

## 许可

MIT
