[![English](https://img.shields.io/badge/English-555555?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-2962FF?style=for-the-badge)](README.zh-CN.md)

# llm-long-term-memory

**面向 LLM 应用的持久化长期记忆层。**

它把对话压缩成紧凑的记忆、在事实发生变化时更新它们，并在压缩恰好丢掉了问题所需的
细节时，回到原始对话把细节找回来。

---

## 一次对话就能说清的问题

三个月前助手回答过一个问题。今天用户问起它：

> **问：** 你之前推荐的那个 Mayo Clinic 视频是什么？

记忆库里关于 Mayo Clinic 什么都没有——抽取保留了大意，丢掉了链接。向量库到这里就
失败了，而且会一直失败下去。这个系统不会：

```
结构化记忆        不足
原始对话回退      已触发 —— 全档案检索
证据来源          session answer_sharegpt_81riySf_0 · turn 1 · assistant

答案
  "How to Sit Properly at a Desk to Avoid Back Pain"
  https://www.youtube.com/watch?v=UfOvNlX9Hh0
```

压缩必然有损。这个设计接受这一点，只要求损失是**可恢复的**。

### 自己看

按下面的方式启动服务，然后打开：

| Demo | 展示什么 |
|---|---|
| [`/?demo=mayo`](http://localhost:8000/?demo=mayo) | 压缩丢掉了 URL，档案把它找了回来 |
| [`/?demo=battery`](http://localhost:8000/?demo=battery) | 记忆足够——回答**运用**了记住的偏好，没有回退 |
| [`/?demo=timeline`](http://localhost:8000/?demo=timeline) | 同一个事实的五次变化，以及取代链 |
| [`/?demo=collectibles`](http://localhost:8000/?demo=collectibles) | 多值键：这些事实并存，谁也没有取代谁 |

每个打开的都是**录制的运行结果**——真实模型的真实输出，并明确标注为录制，所以打开
页面不消耗任何额度。**Run live answer** 按钮可以按需重新执行。录制结果带有产生它的
库和 prompt 版本的指纹；任何一个变了，页面会显示 **stale**，而不是冒充当前行为。

---

## 它做什么

| | |
|---|---|
| **记住** | 把对话变成带类型、带时间窗的事实——谁说的、说的是谁、为什么值得留 |
| **更新** | 追踪变化的事实。"我搬到悉尼了"会取代"我住在堪培拉"，但不删除它 |
| **检索** | 五路加权信号——语义、BM25、时间新旧、重要性、实体重合 |
| **恢复** | 结构化记忆不足时，去检索原始对话，而不是直接失败 |
| **解释** | 每条记忆都能追到它来自的那句话；每一次**遗漏**都有明确原因 |

最后一条是差异所在。检索不仅返回命中了什么，还返回**什么被排除了、为什么**——
`superseded`（事实已不成立）或 `below_rank`（仍然成立，但分数不够）。没有解释的
缺失，和 bug 无法区分。

---

## 30 秒看懂架构

```
                     对话
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
    原始对话档案               抽取器
      （turns）           （每批 2 次 LLM 调用）
          │                       ↓
          │                  结构化记忆
          │              带类型 · 双时间轴
          │                 带溯源锚点
          │                       │
  查询 ───┼───────────────────────┘
          │        混合检索（5 路信号）
          │                       ↓
          │                 足够回答吗？
          │              ┌────────┴────────┐
          │             是                 否
          │              ↓                  ↓
          └──────────> 回答          回到原始证据
                                            ↓
                                    带出处的回答
```

**原始档案是一个存储决策，不是检索算法。** 保留原始对话是让有损抽取变得可恢复；
至于*怎么*找到它们——按来源直查、BM25、稠密向量、混合——是另一个独立的选择。

| 层 | 选择 | 理由 |
|---|---|---|
| 存储 | SQLite (WAL) + FTS5 | 单文件、无服务依赖；BM25 白送 |
| 向量 | numpy 精确内积 | 几千条记忆；上 ANN 索引是为省微秒而加一个依赖 |
| 嵌入 | `all-MiniLM-L6-v2`，本地 | 全语料嵌入约 5300 万 token，而 API 额度才是硬约束 |
| LLM | Gemini，三个独立角色 | 抽取器、回答器、裁判的要求完全不同 |
| 服务 | FastAPI，单一组合根 | API 和 Inspector 是同一个 service 对象的客户端 |

---

## 2 分钟跑起来

```bash
docker compose up --build
```

```bash
curl localhost:8000/healthz
```

然后打开 <http://localhost:8000> 就是 Inspector。

镜像里不打包任何数据集、模型、凭证或数据库——记忆库通过挂载卷进来。`GEMINI_API_KEY`
是可选的：没有它服务以只读模式运行，并且会明说。

首次构建要下载嵌入模型的依赖，需要几分钟；之后启动只要几秒。如果只要一个不带语义
检索的只读小镜像，用 `--build-arg EXTRAS="api"` 构建——那时 `/healthz` 会报
`degraded`，搜索返回 503 而不是莫名其妙地失败。

**镜像大小 2.95GB。** 其中绝大部分是 PyTorch，嵌入模型需要它。默认安装拉的是 CUDA
版轮子——给一个永远见不到 GPU 的容器装 24.4GB 的 GPU 运行时——所以 Dockerfile 换成
CPU 版，并删掉换装后遗留的 CUDA 包。这仍然不算小；真要小就得把编码器移到进程外，那件事
[在路线图里](docs/ROADMAP.md)，不适合在评测跑到一半时动。

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
| `POST /v1/messages` | 写入一句对话，返回它产生的记忆 |
| `POST /v1/memories/search` | 检索，附带信号、出处和排除原因 |
| `POST /v1/answer` | 完整回答路径，必要时触发回退 |
| `POST /v1/raw/search` | 回退层，可单独查询 |
| `GET /v1/memories` | 浏览一个命名空间，可按状态/类型/scope/说话人过滤 |
| `GET /v1/memories/{id}` | 单条记忆及其源对话 |
| `GET /v1/timeline` | 某个 `(subject, predicate)` 的取代链 |
| `DELETE /v1/memories/{id}` | 遗忘——标记为 evicted，绝不硬删 |
| `GET /healthz` · `GET /v1/config` | 健康检查与运行清单 |

每次读取都必须带 `user_id`，且严格按命名空间隔离。读别人命名空间的记忆返回
**404 而不是 403**——403 会泄露那个 id 确实存在。

---

## 从 MCP 客户端接入

[Model Context Protocol](https://modelcontextprotocol.io) 是 agent 接入它的方式，
而 REST 是程序接入的方式。两者都是同一个 service 对象的薄封装，所以工具的行为不会
和被测量过的行为产生偏离。

```bash
uv sync --extra mcp --extra embed
uv run lltm mcp                    # stdio
uv run lltm mcp --transport http   # streamable HTTP
```

在 Claude Desktop 或 Cursor 的 MCP 配置里加：

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

| 工具 | |
|---|---|
| `search_memory` | 回忆已知的事情，附带每条被选中的原因——加上 `explain` 还会说明别的为什么没被选 |
| `remember` | 存一句对话，返回它产生的记忆，让 agent 能确认系统理解到了什么 |
| `search_conversations` | 当记忆切题但缺少精确细节时，回到原文把它找回来 |
| `get_timeline` | 某个事实随时间如何变化 |
| `forget` | 把记忆标记为 evicted，绝不硬删 |

每个工具都要求显式的 `user_id`。这里没有隐含的会话身份：为多个人服务的 agent 必须
说明它此刻代表谁，而边界由存储层强制，不依赖调用方自觉。

---

## 依据

上面的设计不是偏好。每个选择都经过测量，而且有两个看起来合理的增强**测过之后没有上线**。

### 结构化记忆作为一种压缩

在 [LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) 已完整摄入的 31 道题上，
全程同一个回答器和裁判（[完整记录](results/a2-pilot.md)）：

| 变体 | 准确率 | 中位上下文 tokens |
|---|---:|---:|
| `full_context` | 64.5% | 109,605 |
| `naive_rag` | 51.6% | 13,057 |
| **`two_stage` (k=10)** | **51.6%** | **242** |
| v1 结构化记忆 | 19.4% | 465 |

**上下文比 naive RAG 少约 54 倍，准确率无可检测差异**（6 胜 6 负，配对精确 McNemar，
p = 1.000）。

> **请当作 pilot 来读。** 31 道题、非分层抽样、各跑一次。`p = 1.000` 的意思是
> *没有检测到差异*，不等于两个系统等价。分层 50 题的正式运行还没跑，也还没有任何
> held-out 结果。

### 提升来自哪里

早期草稿把功劳归给了证据 hydration。因果消融推翻了这个归因：

| 步骤 | Δ | 配对结果 |
|---|---:|---|
| 抽取管线重写 | **+29.0pp** | 11 胜 2 负，**p = 0.022** |
| 证据 hydration | +3.2pp | 2 胜 1 负，p = 1.000 |

两者的 source-session recall **都是 93.5%**，这一点印证了归因——hydration 在检索之后
运行，本来就不可能改变召回了什么。于是 hydration 从"总是开启"降级为上面那个条件回退。

### 测过但没有上线的

**Cross-encoder 重排。** 在 k=10 时，开启与关闭重排在 31 道题上给出了**逐题完全相同**
的答案——零个分歧——而重排在每个 k 上都**降低**了 source recall（k=20 时 93.5% → 90.3%）。
它作为可选的 `rerank` extra 保留在仓库里，让这个结论可复现，但默认关闭。
[详情](results/rerank-pareto.md)。

**对原始档案做稠密检索。** 一组构造出来的查询显示 BM25 在改写下崩溃（R@1 只有 6.9%）。
人工审计发现这种构造把**问题本身**也删掉了，而不只是删掉了词汇——*"How long have I been
collecting vintage cameras?"* 变成了 `"long"`。这条证据因此被作废。手写的同义改写全部在
**rank 1** 命中，包括完全不含原文专有名词的表述。结论是暂缓，不是否决，并写明了什么条件
会重新开启这个议题。[详情](results/raw-recall-diagnostic.md)。

**用于预算打包的学习式效用预测器。** held-out RMSE 0.310，而预测均值的基线是 0.263——
它输给了预测均值。[详情](results/p6-pilot.md)。

### Live 回归

七道此前所有变体都答错的题，现在是 **6/7，三轮结果完全一致**——包括 Mayo 那道题的
完整链路。[详情](results/live-regression-v2.md)。

---

## 工程

- **363 个测试**，CI 覆盖 ubuntu / windows / macos
- **结果版本化**——每一行评估结果都记录它的回答器 prompt、裁判 prompt 和抽取器版本；
  抽取器版本读自**记忆库**而不是当前代码，因为它描述的是被评估的那批数据
- **冻结的题目清单**——正式运行必须指定明确的题目集，因为"样本量"不是"实验身份"
- **结构化日志**——每个请求一条 JSON，含请求 id、延迟和配置指纹；不含记忆内容和凭证
- **启动时预热编码器**：首个请求从 12,486 ms 降到 181 ms

完整记录：[工程报告](docs/ENGINEERING_REPORT.md) ·
[设计决策](docs/DECISIONS.md) · [路线图](docs/ROADMAP.md)

---

## 开发

```bash
uv sync --group dev --extra api --extra llm --extra embed
uv run pytest
uv run lltm --help
```

<details>
<summary>基准测试流程</summary>

```bash
uv run lltm data download --variant s
uv run lltm doctor
uv run lltm ingest run --store-name two-stage-hydrated
uv run lltm eval freeze dev50 --n 50
uv run lltm eval run two_stage --questions results/manifests/dev50.json
uv run lltm eval compare naive_rag two_stage
```

`--questions` 接受一个冻结的清单。`--limit` 会重新抽样，只用于探索——分层子集散布在
整个数据集里，并不是它的前缀。
</details>

```
src/llm_long_term_memory/
  api/         FastAPI 服务、Inspector、录制的 demo 运行
  store/       schema.sql、SQLiteMemoryStore、NumpyFlatIndex
  ingest/      两阶段抽取、去重、断点续跑的流水线、质量闸门
  retrieve/    混合检索、原始对话回退、重排器（可选）
  evaluation/  数据集加载、运行器、裁判、清单、报表
  llm/         配额感知的限流器（RPM / TPM / RPD）、带重试的客户端
```

## 局限

- 没有 held-out 结果。所有数字都来自同时用于调 prompt 和调闸门的开发题集。
- 已摄入完整的那个记忆库混了两代抽取器——P10 修复前写入 4,843 条，修复后 2,265 条——
  所以它不对应单一系统版本，在它上面测出的唯一结果被标注为诊断用途。干净重建正在进行。
  事故经过和之后加的守卫见[工程报告第 10 节](docs/ENGINEERING_REPORT.md)。
- 单写入者的 SQLite。摄入和评测现在都会取跨进程锁；除此之外不支持并发。
- 时间推算和跨会话聚合仍未解决——见[工程报告](docs/ENGINEERING_REPORT.md)里的失败分类。

## 许可

MIT
