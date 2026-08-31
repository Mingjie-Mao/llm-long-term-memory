[![English](https://img.shields.io/badge/English-555555?style=for-the-badge)](README.md)
[![中文](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-2962FF?style=for-the-badge)](README.zh-CN.md)

# llm-long-term-memory

面向 LLM 应用的持久化长期记忆层。它把对话转成结构化、带时间边界的事实，在事实变化时更新
它们，并在压缩丢掉了问题所需的细节时回到原始对话。

它围绕三个约束设计：整段聊天记录塞进上下文既贵又无上限地增长；对话原文上的向量库表达不了
"事实会变"；把对话压缩成事实能同时解决这两点，但压缩是有损的。

**技术栈** · Python · FastAPI · SQLite + FTS5 · all-MiniLM-L6-v2 · Gemini · MCP · Docker

**核心能力** · 时间性事实 · 事实取代 · 溯源到来源轮次 · 条件回原文兜底 · 可解释检索

**在线演示 — <https://lltm-memory.pages.dev>**（中英双语）。三个问题分别演示：一条事实被
取代、抽取丢掉的链接从原文档案里捞回、以及记忆如何改变回答的内容。页面上的数据是为演示写的
虚构内容；实测数字在下面。

## 核心设计

**结构化记忆。** 每条事实是一行带类型的记录，以 `(subject, predicate, object)` 为键，
带一个说明"为什么值得留"的 `scope`。谁说的（`source_role`）和这条事实关于谁（`subject`）
是两个分开的字段，所以第三方事实和助手的推荐都有地方存放。

**事实更新。** 记忆是双时间的：`event_time` 是这件事在世界上何时成立，`ingested_at` 是
系统何时知道它。同一个 `(user_id, subject, predicate)` 键上的新值会关闭旧值的有效期并
标记为 `superseded`，而不是删除它。

**原文回退。** 原始轮次留在同一个数据库里，带 BM25 索引。回答器返回一个结构化判定而不是
散文，只有 `need_source` 这个判定才会为第二次读取原文付费。给每个回答都附上证据的方案也
测过，代价是 3 倍上下文，换不到可测的收益。

它存在的理由是这一类情况：一条记忆写着*"助手推荐了一个关于坐姿的 Mayo Clinic 资源"*，
它是真的，在关于那次推荐的问题上也排第一，却答不了*"那个 URL 是什么"*——抽取留下了大意，
丢掉了链接。归档能把原始轮次找回来。

**出处追踪。** 每条记忆都能解析到来源会话、轮次序号和字符区间。检索返回的不只是命中了什么，
还有**什么被拒绝了、理由是什么**——`superseded` 或 `below_rank`。

## 架构

```
                     Conversation
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
        原始对话归档                抽取器
        （turns 表）          （每批 2 次 LLM 调用）
              │                       ↓
              │                 结构化记忆
              │            带类型 · 双时间 · 锚定出处
              │                       │
  Query ──────┼───────────────────────┘
              │              记忆检索
              │                       ↓
              │                 够回答吗？
              │              ┌────────┴────────┐
              │             够                不够
              │              ↓                  ↓
              └──────────> 回答          回原文检索
                                                ↓
                                          带出处的回答
```

| 层 | 选择 |
|---|---|
| 存储 | SQLite（WAL）+ FTS5，单文件，无服务依赖 |
| 向量 | 对归一化的 384 维嵌入做精确扁平内积（numpy） |
| 嵌入 | `all-MiniLM-L6-v2`，本机运行——API 额度才是绑定约束 |
| LLM | `gemini-3.1-flash-lite` 抽取、`gemini-3.5-flash-lite` 回答、`gemma-4-31b-it` 判分。固定 id，绝不用 `-latest`，三个独立配额池 |
| 服务 | FastAPI；REST、MCP 和 Inspector 共用同一个 service 对象 |

## 实验结果

**保留集**——100 道来自
[LongMemEval-S](https://github.com/xiaowu0162/LongMemEval) 的题，未参与任何决策，
在冻结并哈希过的系统上跑一次：

| | |
|---|---:|
| 最终正确率 | **70.0%** |
| 仅靠结构化记忆答对 | 50.0% |
| 由原文归档找回 | 20.0% |
| 中位上下文 token | 1,468 |

同一配置又跑了两次，得 71 和 73，区间是 70–73、均值 71.3。报告 `70.0%` 是因为协议在重复
运行存在之前就承诺了报告这一枪。

**开发集（`dev50`），带基线**——所有设计决策都是在这里做的：

| 变体 | 正确率 | 中位上下文 token |
|---|---:|---:|
| `full_context` —— 整段聊天记录 | 56.0% | 109,260 |
| `naive_rag` —— 对原始会话做向量检索 | 54.0% | 13,057 |
| 仅结构化记忆 | 56.0% | 1,415 |
| 结构化记忆 + 条件回退 | **72.0%** | 1,455 |
| v1 结构化记忆 | 26.0% | 465 |

有两条限制应该紧挨着这些数字，而不是放在脚注里。`dev50` 被用来做 prompt 迭代和阈值调参，
所以它的 72.0% 衡量的是这些选择的拟合程度，不只是系统本身。而且**基线从未在保留集上跑过**，
所以"结构化记忆打平 naive RAG"在未见数据上没有证据；补上这个缺口是下一轮第一个注册的臂。

消融把增益归给抽取重写，而不是默认附上原文证据；配对数字见[系统设计报告](docs/REPORT.zh-CN.md)。

## 快速开始

```bash
docker compose up --build
curl localhost:8000/healthz
```

Inspector 在 <http://localhost:8000>。镜像里不含任何数据集、模型、凭据或数据库——store 从
挂载卷进来。`GEMINI_API_KEY` 是可选的；没有它服务以只读模式运行，并且会明确说出来。

写入一轮对话再检索回来：

```bash
curl -X POST localhost:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "alice", "role": "user", "content": "I stopped drinking coffee last month."}'

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

<details>
<summary>不用 Docker</summary>

```bash
uv sync --group dev --extra api --extra embed
uv run uvicorn llm_long_term_memory.api.app:app --port 8000
```

开发与基准测试：

```bash
uv sync --group dev --extra api --extra llm --extra embed
uv run pytest
uv run lltm --help
```
</details>

## 接口

每次读都必须带 `user_id`，按命名空间隔离。读另一个命名空间的记忆返回 **404 而不是 403**
——403 等于确认这个 id 存在。

<details>
<summary>REST 端点</summary>

| 端点 | |
|---|---|
| `POST /v1/messages` | 写入一轮对话；返回它产生的记忆 |
| `POST /v1/memories/search` | 检索，带信号、出处和被拒记录 |
| `POST /v1/answer` | 完整回答路径，必要时含回退 |
| `POST /v1/raw/search` | 回退层，可直接查询 |
| `GET /v1/memories` | 浏览一个命名空间；可按状态/类型/scope/说话人过滤 |
| `GET /v1/memories/{id}` | 一条记忆及其来源轮次 |
| `GET /v1/timeline` | 某个 `(subject, predicate)` 的取代链 |
| `DELETE /v1/memories/{id}` | 遗忘——标记为 evicted，绝不硬删除 |
| `GET /healthz` · `GET /v1/config` | 健康状态与运行中的清单 |
</details>

<details>
<summary>MCP server</summary>

```bash
uv sync --extra mcp --extra embed
uv run lltm mcp                    # stdio
uv run lltm mcp --transport http   # streamable HTTP
```

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
| `search_memory` | 回忆已知的事，附每条记忆被选中的理由 |
| `remember` | 写入一轮对话；返回它产生的记忆 |
| `search_conversations` | 当记忆缺少细节时，找回原始措辞 |
| `get_timeline` | 一个事实随时间如何变化 |
| `forget` | 标记为 evicted，绝不硬删除 |

每个工具都要显式的 `user_id`，没有隐式会话身份。
</details>

## 局限

- **保留集上没有基线。** `heldout100` 只跑了本系统，所以 `70.0%` 没有未见过的对照点。
  这里每一个基线数字都来自 `dev50`。
- **抽取器是已知有损的，而且没有改。** 随机对照显示批大小 15 每会话产出 2.6 条记忆，
  批大小 1 是 12.7 条。这里所有数字都在这个天花板之下。
- **配对结论都来自每臂单次运行。** `heldout100` 三次重复里有 6/100 翻转，这个量级足以
  单独把几十道题的比较推过显著性阈值。
- **五路检索信号里有四路权重是 0，打开之后实测更差。** `recency` 还用修正后的半衰期重测过，
  依然没有信息量——这个基准的 gold 会话并不偏向更新的那些。
- **还不是产品。** `user_id` 取自请求正文而非可信令牌，删除只是状态变更而非真正抹除，
  SQLite 单写，且从未做过恢复演练。

## 文档

| | |
|---|---|
| [系统设计报告](docs/REPORT.zh-CN.md) · [English](docs/REPORT.md) | 按数据流的完整说明：写入通路、存储、检索、评测协议、消融、失败分析 |
| [设计决策](docs/DECISIONS.md) | 30 条编号决策，每条附支撑它的测量 |
| [工程报告](docs/ENGINEERING_REPORT.md) | 更早的按时间线写的记录，含混合抽取器 store 事故 |
| [路线图](docs/ROADMAP.md) | 已完成的和接下来的 |
| [当前状态](docs/CURRENT_STATUS.md) | 持续更新的实验状态、冻结边界和最近一步 |
| [架构分层计划](docs/ARCHITECTURE_SEPARATION_PLAN.md) | 分阶段拆分产品核心、研究代码、负结果和审计证据 |
| [数据协议](results/data-protocol.md) | 五个问题集合各自允许怎么用 |
| [`results/`](results/) | 预注册、原始运行记录和逐个实验的写作 |

## 许可

[MIT](LICENSE)
