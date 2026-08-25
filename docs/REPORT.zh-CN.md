# LLTM 系统设计报告

一个 LLM 长期记忆层的完整技术说明。写给熟悉 Python、LLM 和 RAG 基础、但第一次接触
这个项目的人：读完之后应该能说清楚一条对话进入系统后经过了哪些组件、每一步具体做了什么、
为什么这样设计、以及哪个实验支持这个决定。

**本文所有实现细节以当前代码和冻结配置为准。**凡是代码无法证明的，明确写"当前材料无法确认"。
文中不区分"设计支持"和"生产启用"的地方会被特别标注，因为这两件事在本项目里差别很大。

配套文档：[`ENGINEERING_REPORT.md`](ENGINEERING_REPORT.md) 是更早的按时间线写的英文报告，
[`DECISIONS.md`](DECISIONS.md) 是编号的设计决策，[`REPORT.md`](REPORT.md) 是本文的英文版。

文档版本：2026-08-25。

---

## 技术栈

| 层 | 用了什么 | 版本约束 | 备注 |
|---|---|---|---|
| 语言 | Python | `>=3.11` | 用到 `StrEnum`（3.11+）、`Literal` 类型别名 |
| 包管理 / 构建 | `uv` + `hatchling` | — | `uv.lock` 锁定全部传递依赖 |
| 数据校验 | `pydantic` / `pydantic-settings` | `>=2.7` / `>=2.3` | 抽取输出、配置、API 模型全部走 pydantic schema |
| CLI | `typer` + `rich` | `>=0.12` / `>=13.7` | `lltm` 与 `llm-long-term-memory` 两个入口指向同一个 app |
| 数值 | `numpy` | `>=2.0` | 向量索引就是一个 numpy 矩阵 |
| HTTP 客户端 | `httpx` | `>=0.27` | 调 LLM API |
| 日志 | `structlog` | `>=24.1` | 每请求一条 JSON 事件 |
| 配置 | `pyyaml` | `>=6.0` | `configs/*.yaml` |
| 时区 | `tzdata`（仅 Windows） | — | Windows 不带系统时区库，`zoneinfo` 解析不了 `America/Los_Angeles`——而服务商的每日配额正是在太平洋午夜重置，限流器要数的就是它。这个缺陷是 Windows CI 抓到的 |

**可选依赖（extras），刻意拆开：**

| extra | 内容 | 为什么是可选的 |
|---|---|---|
| `embed` | `sentence-transformers>=3.0` | 拉 torch，约 2GB。只做数据集/统计工作时不需要 |
| `llm` | `google-genai>=1.0` | 只有抽取和回答需要 API key |
| `api` | `fastapi>=0.115`、`uvicorn[standard]>=0.30` | CLI 和测试套件不应该依赖一个 web 框架 |
| `rerank` | `sentence-transformers>=3.0` | 交叉编码器重排**已测量且未启用**。单独成一个 extra，是为了让这个依赖变成"刻意引入"而不是"顺带引入" |
| `mcp` | `mcp>=1.2` | 一个纯 REST 部署不该携带它从不使用的协议实现 |
| dev | `pytest>=8.2`、`pytest-cov>=5.0`、`ruff>=0.6` | — |

**存储与检索栈：**

| | |
|---|---|
| 关系存储 | SQLite（`journal_mode=WAL`，`foreign_keys=ON`），单文件 |
| 全文检索 | SQLite FTS5 虚拟表，`tokenize='porter unicode61'`，外部内容表 + 触发器同步 |
| 词法排序 | FTS5 内置 `bm25()`（**返回负数**，越负越相关） |
| 向量存储 | `.npy`（float32 矩阵）+ `.ids.json`，与 `.db` 并列 |
| 向量检索 | numpy 精确内积 + `argpartition` 取 top-k，**不用 FAISS/ANN** |

**模型（全部固定版本，绝不用 `-latest`）：**

| 角色 | model id |
|---|---|
| 抽取器（Stage A + Stage B） | `gemini-3.1-flash-lite` |
| 回答器（被测系统） | `gemini-3.5-flash-lite` |
| 判分器 | `gemma-4-31b-it` |
| 嵌入器（本机运行） | `sentence-transformers/all-MiniLM-L6-v2`，384 维 |

**工程：**

| | |
|---|---|
| 测试 | pytest，**529 个测试**，行覆盖 80% |
| 静态检查 | ruff，`line-length = 100` |
| CI | ubuntu / windows / macos 三平台矩阵 |
| 告警即错误 | `filterwarnings = ["error::EncodingWarning"]`——编码缺陷不能悄悄回来 |
| 容器 | Docker + docker compose，镜像 2.95GB（CPU 版 torch） |
| 对外接口 | REST（FastAPI）、MCP、Memory Inspector（网页），三者共用同一个 service 对象 |

**这个技术栈里没有的东西，以及为什么**（详见第 4 章）：没有独立数据库服务、没有向量数据库、
没有 Elasticsearch、没有消息队列、没有 ORM。当前规模（单机、单进程写入、最大 18,017 条记忆）
下它们都只增加运维面而不解决问题；一旦变成多租户服务，第一个要换掉的是 SQLite。

---

## 0. Executive Summary

**解决什么问题。** LLM 应用要记住用户几个月前说过的事。把全部聊天记录塞进上下文太贵、
且随时间线性增长；用向量库检索原始对话片段，则无法表达"事实会变"——用户搬过一次家之后，
旧地址和新地址会一起被检索出来，没有任何机制说明哪个还成立。

**核心设计思想。** 把对话压缩成结构化事实（compact、可更新、可解释），**同时永久保留原始
对话**。压缩必然有损，本项目不试图消除这个损失，只要求它**可恢复**：当结构化记忆不足以回答
时，系统回到原文里把那句话捞出来。

**最重要的架构。** 两条数据通路：

- **写入通路**：对话 → 原文归档 → 两阶段抽取 → 结构化记忆 → 时序消解 → 嵌入 → 索引
- **读取通路**：查询 → 混合检索 → 候选打分 → 上下文组装 → 回答器 → 充分性判定 → （不足时）
  原文回退 → 带出处的答案

**最重要的实验结果。** 在 100 道从未参与任何决策的题上一次性运行：**70.0% 正确率**，其中
50.0% 来自结构化记忆、20.0% 由原文归档救回，中位上下文 1,468 token。开发集上同一系统
72.0%，而把整个聊天记录塞进去的基线是 56.0%、普通 RAG 是 54.0%，上下文分别是它的 75 倍
和 9 倍。

**当前最大限制。** 三条并列：
1. **保留集上没有基线。** 那次只跑了本系统，所以"结构化记忆打平普通 RAG"目前只是开发集结论。
2. **抽取器已知有损且没有修。** 批大小 15 时每会话产出 2.6 条记忆，批大小 1 时是 12.7 条，
   这是随机对照测出来的。所有数字都在这个被压低的天花板之下。
3. **五路检索信号只有一路在生产中生效。** 架构支持五路加权融合，冻结配置里
   `semantic=1.0`，其余四路权重是 `0.0`。

---

## 1. 问题定义与设计目标

### 1.1 方案 A：全量上下文（full context）

**怎么工作。** 每次提问，把用户全部历史对话原样拼进 prompt，交给模型自己找。

**优点。** 不丢任何信息，实现只需要字符串拼接，没有检索误差。

**问题。**
- **成本随历史线性增长。** 本项目实测中位 **109,260 token** 一次提问。
- **"塞进去"不等于"用得上"。** 它在开发集上只有 **56.0%**，被上下文小 75 倍的方案打败。
  长上下文中的信息会被稀释，这是本项目实测到的，不是引用的结论。
- **没有事实状态的概念。** 用户三年前说住堪培拉、去年说搬到悉尼，两句话都在上下文里，
  模型每次都要重新推断哪句还成立。

### 1.2 方案 B：对话原文上的普通 RAG（naive RAG）

**怎么工作。**

```
对话 → 切成 session / turn 片段 → 每片段算 embedding → 存向量库
查询 → 算 query embedding → 找最相似的 k 个片段 → 拼进 prompt → 回答
```

**优点。** 上下文降到中位 **13,057 token**（约为全量的 1/8），实现简单，检索到的是原始
措辞——精确的 URL、型号、数字都原样保留。开发集 **54.0%**。

**它天然解决不了什么。**
- **检索的是"片段"，不是"事实"。** 一个 session 里可能有二十件事，只有一件跟问题相关，
  但整段都会被拉进上下文。
- **无法表达事实演化。** 向量相似度不知道"我搬到悉尼了"应该让"我住堪培拉"失效。两条都会
  被检索到，且旧的可能因为措辞更接近问题而排得更前。
- **无法回答需要跨会话聚合的问题。** "我二月份去了几个博物馆"——没有任何单个片段说了这个数字。

### 1.3 方案 C：结构化长期记忆

**怎么工作。** 用 LLM 把对话读成一条条独立的、带类型和时间的事实：

```
原文： "我上个月戒咖啡了，现在改喝抹茶。"
     ↓
记忆： content   = "The user stopped drinking coffee."
      subject   = "user"
      predicate = "drinks"
      scope     = "profile"
      event_time = 2026-07-15
      update_op = "replaces"
```

**为什么值得做。**
- **紧凑**：一条事实几十个 token，中位上下文降到 **1,455**。
- **可更新**：事实有 `(subject, predicate)` 这个键，新值可以让旧值失效，而不是并存。
- **可解释**：每条事实都能追回它来自哪一轮对话；检索时被淘汰的事实也带理由。

### 1.4 根本取舍：压缩是有损的

这是本项目最重要的一节。

抽取会丢东西，而且丢的往往正是问题要问的那一个标识符。真实案例：

```
原始助手回复（三个月前）：
  "...I recommend this Mayo Clinic video:
   How to Sit Properly at a Desk to Avoid Back Pain
   https://www.youtube.com/watch?v=UfOvNlX9Hh0 ..."

抽取出的记忆：
  "The assistant recommended a Mayo Clinic resource about desk posture."

今天的提问：
  "What was the Mayo Clinic YouTube video you recommended?"
```

这条记忆**是真的**，而且检索会正确地把它排到第一位——它就是关于 Mayo Clinic 的。但它答不了
这个问题，因为 URL 在压缩中被扔掉了。纯向量库在这里会失败，而且会**一直**失败：重跑、换
embedding 模型、加大 top-k 都没用，因为那个字符串根本不在库里。

**本项目的答案不是"把抽取做得更好"，而是"承认它有损，并保证损失可恢复"：**

> **有损的结构化记忆 + 可回溯的原文归档**

原文永久保存在同一个数据库里，带全文索引。结构化记忆答不了的时候，系统回到原文检索。

#### 必须区分的两种"用原文"

这两件事在很多系统里被混为一谈，本项目把它们分开测过，结论相反：

| | **Always-on hydration（常开挂载）** | **Conditional fallback（条件回退）** |
|---|---|---|
| 什么时候取原文 | **每次**回答都把记忆对应的原文附上 | **只有**回答器声明记忆不足时 |
| 上下文代价 | 约 **3 倍** | 平均几乎为零（本项目实测触发率 36%） |
| 实测收益 | **+3.2pp**，2 胜 1 负，p = 1.000 | 在开发集上 **+16.0pp**（56.0% → 72.0%） |
| 当前状态 | **已实现，但降级为不默认启用** | **产品默认路径** |

常开挂载的收益在统计上不可区分于零，却要付三倍上下文——那是往 naive RAG 方向滑回去。
条件回退把这笔钱只花在真正需要的那 36% 的问题上。

> ⚠️ 这里有一个容易读反的地方。本项目另有一个结论是"收益来自抽取重写（+29.0pp），而不是
> 常开挂载（+3.2pp）"。**这句话说的是常开挂载，不是原文归档。**准确的表述是：
>
> *两阶段抽取带来了最大的上游改进；常开的原文挂载不值它的 token 成本；而条件回退承担的是
> 另一个角色——只在结构化记忆不足时，把压缩丢掉的细节找回来。*
>
> 在开发集上，原文归档独立贡献了 18.0 个百分点（54.0% → 72.0%）。它不是可有可无的装饰。

### 1.5 设计目标

| 目标 | 可检验的形式 |
|---|---|
| 紧凑 | 上下文比全量转录低一个数量级以上 |
| 可更新 | 同一 `(subject, predicate)` 的新值能让旧值退出检索，且旧值不被删除 |
| 可恢复 | 抽取丢掉的细节，能通过原文路径答出来 |
| 可解释 | 每条记忆能追到来源轮次；每个**遗漏**有明确理由 |
| 隔离 | 跨用户读写不可能，猜到 id 也不行 |

---

## 2. 端到端架构

```
                          ┌──────────────────────┐
                          │      Conversation     │
                          └───────────┬──────────┘
                                      │
        ┌─────────────────────────────┴─────────────────────────────┐
        │                     WRITE PATH                            │
        │                                                           │
        │   turns 表（原文归档）          Stage A：抽事实串          │
        │   sessions 表                          ↓                  │
        │   turns_fts（BM25 索引）        Stage B：配 key + update_op│
        │        │                               ↓                  │
        │        │                        memories 表               │
        │        │                     （双时间 + 出处锚点）         │
        │        │                               ↓                  │
        │        │                    时序消解（supersession）       │
        │        │                               ↓                  │
        │        │                    嵌入 → NumpyFlatIndex (.npy)  │
        └────────┼──────────────────────────────┬───────────────────┘
                 │                              │
        ┌────────┼──────────────────────────────┼───────────────────┐
        │        │          READ PATH           │                   │
        │        │                              │                   │
        │  Query ─────────────────────────→ 混合检索                │
        │        │                        （5 路信号，融合打分）     │
        │        │                              ↓                   │
        │        │                     候选过滤 + top_k 截断         │
        │        │                              ↓                   │
        │        │                        上下文组装                 │
        │        │                   （v1 平铺 / v2 会话连贯）        │
        │        │                              ↓                   │
        │        │                        回答器（1 次调用）         │
        │        │                              ↓                   │
        │        │                    status ∈ {answer,             │
        │        │                     need_source, no_evidence}    │
        │        │                     ┌────────┴────────┐          │
        │        │                  answer          need_source     │
        │        │                     ↓                 ↓          │
        │        └──────────────────→  │          原文回退检索       │
        │                              │        （BM25 over turns）  │
        │                              │                 ↓          │
        │                              │         回答器（第 2 次）   │
        │                              └────────┬────────┘          │
        │                                       ↓                   │
        │                            答案 + 出处 + 被拒理由          │
        └───────────────────────────────────────────────────────────┘
```

两条通路是本文的骨架。第 3 节讲写入，第 4 节讲它们共同依赖的存储层，第 5 节讲读取。

**一个贯穿全文的约定：** 一次"提问"在默认情况下花费 **1 次**回答器调用；只有回答器自己
声明记忆不足时才花第 2 次。抽取则是离线批量进行的，每 15 个会话花 2 次调用。

---

## 3. 写入通路：一段对话如何变成记忆

### 3.1 原文归档（Raw Conversation Archive）

**这个组件解决什么问题。** 抽取有损（见 1.4）。如果原文不保存，损失就是永久的，系统在
"记忆里没有"和"这件事没发生过"之间无法区分。

**输入。** 一个会话，即一串按顺序排列的 `(role, content, timestamp)`。

**输出。** 写进两张表，并自动建立全文索引。

**数据结构。**

```sql
sessions(id PK, user_id, started_at, source)
turns(id PK, session_id FK→sessions, turn_index, role, content, ts)
INDEX idx_turns_session ON turns(session_id, turn_index)
```

注意 `turns` 表**没有 `user_id` 字段**。用户归属通过 `session_id` 外键 join 到 `sessions`
获得。这不是疏忽：它保证了任何绕过 join 的查询都拿不到用户信息，隔离靠 schema 而不是靠
每处查询记得加条件。原文全文检索的 SQL 因此必须三表 join：

```sql
FROM turns_fts f
JOIN turns t     ON t.rowid = f.rowid
JOIN sessions s  ON s.id = t.session_id
WHERE turns_fts MATCH ? AND s.user_id = ?
```
（`store/sqlite.py:378` `search_turns`）

**出处锚点（provenance）。** 每条记忆在 `memories` 表里带四个字段指回原文：

```
source_session_id  → sessions.id
source_turn_index  → 第几轮
source_char_start  → 该轮文本中的起始字符
source_char_end    → 结束字符
```

这让"这条记忆是怎么来的"可以精确回答到字符区间，而不是"来自某个会话"。在干净的 P10 store 上，
**4,843 / 4,843 条记忆全部能解析到一个来源轮次**——这个数字是本项目敢做条件回退的前提，
因为回退的第一级就是"取回这些记忆锚定的原文轮次"。

**为什么这是存储决策而不是检索算法。** 保留原文和"怎么找到原文"是两个独立的问题。前者是
一个 schema 决定，一旦做了就无法事后补救；后者（source-local 查找、BM25、稠密检索、混合）
可以随时替换，而且本项目确实换过一次实现（见 5.8）。把它们混为一谈会导致一个常见错误：
因为"我们已经有向量检索了"就不保存原文。

| | |
|---|---|
| **选择** | 原文全量保存在同一个 SQLite 文件里，带 FTS5 索引 |
| **为什么** | 与记忆同库，回退路径不需要跨系统一致性；索引让归档不是只写不读的死数据 |
| **替代方案** | 只存对象存储（无法检索）；只存摘要（就是抽取本身，无法恢复）；不存 |
| **牺牲** | 数据库体积。train150 的 `.db` 是 185MB，其中大部分是原文 |
| **证据** | Mayo 案例端到端可答；开发集上归档独立贡献 18.0pp |
| **限制** | 内容策略拒绝的会话，原文仍保留但没有记忆；这类会话单独计数（heldout100 上 1 个） |
| **代码** | `store/schema.sql`、`store/sqlite.py` |

### 3.2 两阶段记忆抽取

这是写入通路最重要的组件，也是本项目失败最集中的地方（14 个开发集失败里有 10 个死在这里）。

**这个组件解决什么问题。** 把一段自然语言对话变成一组结构化、可检索、可更新的事实。

**输入。** 15 个会话打成一个批次（`ingest.sessions_per_request: 15`）。

**输出。** 一组 `Memory` 行。

#### 一个完整的例子

```
── 输入 turn ─────────────────────────────────────────────
session s_4f2a, turn 3, role=assistant, ts=2026-05-02
"Since you mentioned your desk setup, I recommend this Mayo Clinic
 video: How to Sit Properly at a Desk to Avoid Back Pain
 https://www.youtube.com/watch?v=UfOvNlX9Hh0"

── Stage A 输出（裸事实串，1 次 LLM 调用 / 批） ────────────
{ "session_index": 0,
  "fact": "The assistant recommended a Mayo Clinic video about desk posture." }

── Stage B 输出（配 key，1 次 LLM 调用 / 批） ──────────────
{ "temporal_key": "assistant_recommendation",
  "update_op":    "coexists",
  "object":       "Mayo Clinic desk posture video" }

── 规则补齐（0 次调用） ────────────────────────────────────
type=episodic  entities=[mayo clinic]  importance=0.5
event_time=2026-05-02（从 turn 的 ts 解析）
source_session_id=s_4f2a  source_turn_index=3  source_char_start/end=…

── 最终 memory 行 ─────────────────────────────────────────
content    "The assistant recommended a Mayo Clinic video about desk posture."
subject    "assistant"        ← 这条事实是关于谁的
source_role "assistant"       ← 谁说的
predicate  "assistant_recommendation"
object     "Mayo Clinic desk posture video"
scope      "recommendation"
update_op  "coexists"
status     "active"
```

注意 URL 在 Stage A 就丢了。这就是 1.4 描述的有损压缩，条件回退存在的全部理由。

#### 为什么分两阶段，而不是一次调用做完

Stage A 只负责"这段对话里有哪些值得记的事实"，Stage B 只负责"给这条事实配一个时序键，
并说明它对历史事实做了什么"。两次调用而不是一次，每批的成本从约 190 次请求涨到约 380 次。

理由是 Stage B 的**替代方案被实测过**。最初 Stage B 是用正则规则实现的（`_rule_keying`，
代码保留可运行），在 148 条真实记忆上：

| Stage B 实现 | predicate 准确率 | 产生的 supersession 数 |
|---|---:|---:|
| 规则（正则） | **43%** | **0** |
| LLM 调用 | 通过时序门的全部四项指标 | 非零 |

正则无法给开放域关系配键——这是测出来的，不是假设的。而把 Stage A 和 Stage B 合并成一次
调用的问题是：单次调用的输出预算要同时承载"读懂十五段对话"和"为每条事实设计一个键"，
而位置衰减实验（3.2 末尾）表明输出预算正是这个模型的瓶颈。

**规则保留在它们有优势的地方**：`type`、`entities`、`importance`、日期解析全部用规则，
不花 LLM 预算。Stage B 的整个输出预算只用在正则做不到的那一部分。

#### v1 抽取为什么失败

第一版（`chronomem`）在开发集上只有 **26.0%**，被普通 RAG 打了 28 个点。两个原因：

**(1) 用规则描述该抽什么，模型不照做。** 后来改成在 prompt 里放**具体示例**。这个改动在
本项目里被单独量化过一次：加入 worked examples 让来源保真度从 **33.6% → 36.6%**。而且有
一个很尖锐的观察——*唯一一条没有配示例的指令，正好是模型一直不执行的那一条*。

**(2) `subject` 被当成了说话人开关。** v1 的 `subject` 是从字符串前缀推出来的，实际只有
`user` / `assistant` 两个值。后果是：

| | v1 | 两阶段 |
|---|---:|---:|
| `subject='assistant'` 的记忆占比 | 12 / 2,007 = **0.6%** | 326 / 4,843 = **6.7%** |

而"Andy 穿了蓝衬衫"这种**第三方事实无处可放**。这个缺陷的可观测后果是：
`single-session-assistant` 这一类题，所有记忆变体全部 **0/4**，而全量上下文和普通 RAG
都是 4/4。这么整齐的差距不可能是排序问题。

**修复**（见 3.3）：把"谁说的"和"这条事实关于谁"拆成两个字段。

#### 批大小 15 是怎么来的，以及它的已知代价

`sessions_per_request: 15` 的原始理由是配额：免费额度每个模型每天 500 次请求，2,400 个
会话在批大小 15 下需要 189 个批次 / 378 次抽取调用，能在一天内跑完；当时的质量依据是
"批大小 10 时零跨会话误归属"。

**后来的随机对照实验证明这个配置在丢信息。** 60 个会话（30 个在生产中零产出、30 个正常，
按轮次数 ±2 和助手发言占比 ±0.08 配对），批大小固定 15，两组随机排列各配一个自己的逆序——
同一个会话在一个排列里坐第 `p` 位，在另一个里就坐第 `14-p` 位。模型、prompt、schema、
解码参数全程相同，抽取器被直接驱动而不经过流水线，**不写任何 store**。

配对结果（51 个会话同时见过首尾两种位置）：

| | 覆盖率视角 | 数量视角 |
|---|---|---|
| 前面出记忆、后面零产出 | **8** | 每会话记忆数，前 **4.5** |
| 后面出记忆、前面零产出 | **0** | 每会话记忆数，后 **1.6** |
| 精确 McNemar | **p = 0.0078** | **p < 0.0001** |

同一段对话、同样的批大小、同样的 prompt、同样的邻居，挪到请求后面产出只剩三分之一。
**零产出不是一个独立的失败模式，它是一条连续衰减曲线的尾巴。**

再测批大小本身：

| 批大小 | 零产出率 | 每会话记忆数 | 全语料需要的抽取请求 |
|---|---:|---:|---:|
| **15（生产配置）** | 11.7% | **2.7** | 400 |
| 5 | 8.3% | 4.8 | 1,000 |
| 1 | **0.0%** | **12.7** | 4,800 |

60 个会话里**没有一个**在更大批次下产出更多，两次比较都是如此（p = 1.7e-18）。

**当前代价，说清楚：本项目所有数字都是在批大小 15 下产生的，也就是在一个每会话丢掉约
四分之三记忆的抽取器之上。** 这个配置没有改，因为改了意味着所有 store 重建（免费额度下
多花 28 个配额日），而实验结论还没有冻结。这是一个被压低的天花板，不是一个未知数。

**为什么原计划的补救措施是错的。** 原本要做的是"把零产出的会话捞出来重跑，看恢复率"。
这个设计会给出一个自信的错误答案：把它们单独拿出来重新组批，等于把它们全部挪到了**前面**，
恢复率当然高，结论就会写成"随机丢失，加个重试就行"，而真实机制是"我们改变了它们的位置"。
更根本的问题是，重试治的是错的东西——一个本该产出十条、实际产出两条的会话永远触发不了
重试，而按上表，那才是损失的大头。

| | |
|---|---|
| **选择** | 两阶段 LLM 抽取，批大小 15，规则补齐非语义字段 |
| **为什么** | 规则 Stage B 实测 43% 键准确率、0 次覆盖；批大小 15 是当时的配额约束 |
| **替代方案** | 单次调用做完（输出预算冲突）；纯规则（已测，失败）；批大小 1（已测，更好但 12 倍请求） |
| **牺牲** | 每会话约 3/4 的记忆产出；每批多一次 LLM 调用 |
| **证据** | `results/batch-position-pilot.md`（随机对照）；`docs/DECISIONS.md` D28/D30 |
| **限制** | 位置衰减机制未定（输出预算耗尽 / 枚举漂移 / 输入位置效应 / schema 长度压力，本实验无法区分） |
| **代码** | `ingest/two_stage.py`、`ingest/extract_facts.py`、`ingest/keying.py`、`ingest/structure.py` |

### 3.3 记忆的 Schema

以下字段全部来自 `store/schema.sql`，不是示意。

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | TEXT PK | 记忆 id |
| `user_id` | TEXT | 命名空间。每次读都必须带，隔离的实施点 |
| `type` | TEXT | `semantic` / `episodic` / `preference` / `procedural` / `profile` |
| `content` | TEXT | 人读的那句话，也是被嵌入和被 BM25 索引的文本 |
| `subject` | TEXT | **这条事实关于谁** |
| `predicate` | TEXT | 时序键。覆盖检测匹配 `(user_id, subject, predicate)` |
| `object` | TEXT | 谓语的值 |
| `source_role` | TEXT | **谁说的**：`user` / `assistant` / `system` |
| `scope` | TEXT | 为什么值得留：`profile`/`preference`/`plan`/`recommendation`/`commitment`/`shared_context`/`event` |
| `importance` | REAL | [0,1]，写入时赋值，是第四路检索信号 |
| `confidence` | REAL | [0,1]，被合并时降低 |
| `event_time` | TEXT | **有效时间轴**：这件事在世界上发生/成立的时刻 |
| `valid_from` / `valid_to` | TEXT | 有效区间；`valid_to IS NULL` 表示仍然成立 |
| `ingested_at` | TEXT | **记录时间轴**：我们何时知道的 |
| `replaces_previous` | INTEGER | 用户措辞明确表示这是替换（"我换成了 X"） |
| `update_op` | TEXT | Stage B 的判定：`coexists`/`replaces`/`removes`/`none` |
| `superseded_by` | TEXT FK | 被哪条记忆取代 |
| `status` | TEXT | `active` / `superseded` / `evicted` |
| `strength` / `access_count` / `last_accessed_at` / `strength_updated_at` | | 衰减与强化（P5），当前生产配置未启用（`use_strength=False`） |
| `token_count` | INTEGER | 写入时算好，打包器把选记忆当背包问题，需要便宜地拿到重量 |
| `source_session_id` / `source_turn_index` / `source_char_start` / `source_char_end` | | 出处锚点 |

索引：

```sql
idx_mem_user_status  (user_id, status)
idx_mem_sp           (user_id, subject, predicate) WHERE status='active'   -- 部分索引
idx_mem_type         (user_id, type, status)
idx_mem_valid        (user_id, valid_from, valid_to)
```

`idx_mem_sp` 是**部分索引**（带 `WHERE status='active'`）：覆盖检测只关心还成立的事实，
把已失效的行排除在索引外，索引更小、查询更快。

#### 为什么"谁说的"和"关于谁"必须是两个概念

这是本项目一个关键的 schema 决定，schema.sql 里有专门的注释解释它。

考虑三句话，都由用户说出：

| 原句 | `source_role` | `subject` |
|---|---|---|
| "我住在悉尼。" | user | user |
| "Andy 穿了蓝衬衫。" | user | **andy** |
| （助手说）"我推荐 Mayo Clinic 那个视频。" | **assistant** | assistant |

如果只有一个字段，第二行无处安放——它既不是关于用户的，说话人也不是 Andy。v1 就是这样，
结果是**所有记忆都变成用户画像条目，第三方事实全部丢失**。

而这一对字段是"你之前推荐了什么"这类问题能被回答的前提：这个问题过滤的是
`source_role='assistant'`，跟 subject 的任何属性都无关。

### 3.4 双时间表示（Bitemporal）

系统里有两条独立的时间轴，从第一次迁移就存在：

| 轴 | 字段 | 回答的问题 |
|---|---|---|
| **有效时间**（valid time） | `event_time`, `valid_from`, `valid_to` | 这件事**在世界上**什么时候成立？ |
| **记录时间**（transaction time） | `ingested_at` | 我们**什么时候知道**的？ |

**为什么必须两条。** 举例：

```
2026-03-01  用户说："我上个月搬到悉尼了。"
```

- `event_time` / `valid_from` = **2026-02**（搬家实际发生的时间）
- `ingested_at` = **2026-03-01**（系统知道这件事的时间）

如果只有一条时间轴，两个问题会互相污染：
- "我二月份住在哪？" 需要按**有效时间**查询。
- "系统三月一号之前知道我住悉尼吗？" 需要按**记录时间**查询。

而且用户经常追溯地陈述过去（"我上个月……"、"去年我……"），这时两条时间轴的顺序甚至可以
相反：一条 `event_time` 更早的事实，`ingested_at` 反而更晚。覆盖逻辑排序用的是
`event_time`，不是 `ingested_at`——否则"我上个月搬到悉尼"会被更早录入但描述更早事件的记忆
盖掉。

**为什么从第一天就加。** 事后给已有记忆补时间维度需要重写所有历史行，而重建一个 store
要花几小时的配额。一开始就留着几乎不花成本。这是 `docs/DECISIONS.md` D9。

本节没有数学公式，也不需要强行公式化。

### 3.5 事实更新与覆盖（Supersession）

**这个组件解决什么问题。** "我住堪培拉" 和 "我搬到悉尼了" 都是真话，但只有一条现在成立。
检索必须知道哪条。

**输入。** 同一个 `(user_id, subject, predicate)` 键下的全部记忆，包括已失效的。

**输出。** 每条记忆的 `status` / `valid_to` / `superseded_by` 被更新。

**机制，一步一步。**（`temporal/resolve.py:195` `_resolve_key`）

```
1. 取出该键下全部记忆（含 superseded）。少于 2 条 → 直接返回。

2. 判断这个键是否"可消解"：
      is_single_valued(predicate)  或  任意一条带 replaces_previous
   都不满足 → 进入修复分支（见下），返回。

3. 只保留有 event_time 的记忆；不足 2 条 → 返回。
   （无日期的计入 skipped_undated，不猜测）

4. 按 (event_time, id) 排序。

5. 把连续的等值记忆折叠成"运行段"(run)：
   同一个值被重复陈述多次，最早的那条拥有整个区间，其余是复述。

6. 逐段处理：只有当【下一段的首条】自己带 replaces_previous 时，
   才关闭当前段的有效期。
```

**第 6 步是一个真实事故的修复。** 早先的实现是"这个键可消解"就把链上每一对连续记忆都退休。
线上观察到的后果：

> "averaging $100 per week on groceries" 被 "spent $75 at Walmart last Saturday" 退休了。
> 两句话都是真的。原因是 `grocery_spending` 这个键上**某第三条**记忆带了 replaces 信号，
> 于是整条链被当作单值时间线处理。

Stage B 的判定单独测试时误覆盖率是 0%，损失全部发生在集成层。所以现在**逐条事实的判定必须
被逐条尊重**，不能汇总成一个"这个键是单值的"结论。

**第 2 步的修复分支同样重要。** 如果一个键曾经在错误的 arity 判断下被消解过，它的事实此刻
正躺在 `superseded` 状态、对检索不可见。所以当发现这个键其实不可消解时，代码会把它们
**恢复成 active**（`mark_current`），而不是仅仅跳过。理由写在注释里：否则一次错误的判断
会永久留在每一个已经建好的 store 里，而重建一个 store 要花几小时配额。

**两条判据的关系。**

```python
SINGLE_VALUED_PREDICATES = frozenset(
    {
        "lives_in",
        "works_as",
        "works_at",
        "uses_framework",
    }
)
```

只有四个词。这是刻意保守的：一个错误的覆盖会让**还成立的事实**从所有后续查询里消失，
而一个漏掉的覆盖只是让 store 退化成一个平铺列表。所以默认是 `coexists`，
**不对称的代价决定了不对称的默认值**。

`docs/DECISIONS.md` D23 记录了这个列表曾经七个里错两个，两个条目因为"把不相关的事实串起来"
被移除。`update_op` 就是为了取代这个手工列表而引入的——直接问模型"这条事实对历史做了什么"，
而不是靠谓语名字去猜它是不是单值的。

#### 当前 knowledge-update 失败暴露的限制

`knowledge-update` 这一类题在开发集上 8/8 = 100%，在保留集上 **10/15 = 66.7%**，是跌幅
最大的一类。五个失败全部召回了 gold session **和** gold evidence，所以没有一个是检索失败；
四个发生在"拿到事实之后如何理解和组合"这一层。

store 里有一个形状吻合的症状（以下为本报告在 `heldout100.db` 上直接查询所得）：

```sql
SELECT subject, predicate, COUNT(*) c,
       SUM(replaces_previous) rp, SUM(status='superseded') sup
FROM memories WHERE subject<>'' AND predicate<>''
GROUP BY subject, predicate HAVING c>1 AND rp>0 ORDER BY c DESC;
```

| 键的形状（heldout100） | 键数 | 记忆数 | |
|---|---:|---:|---|
| 只有 1 条记忆的键 | 5,348 | 5,348 | 没有东西可替换 |
| 多于 1 条记忆的键 | 1,266 | 7,054 | |
| …其中**可消解** | 56 | 160 | 单值谓语，或该键上带替换信号 |
| …其中**惰性** | **1,210** | **6,894** | 一个键上多个值，却**永远无法**覆盖——**占全库 55.6%** |

替换信号本身：

| | heldout100 | train150 |
|---|---:|---:|
| 带 `replaces_previous` 的记忆 | 622 | 902 |
| …其中**独占一个键** | **546（88%）** | **777（86%）** |
| …这些之中，其 `(user, subject)` 下**还有别的谓语** | **525（96%）** | **757（97%）** |
| 实际被覆盖的记忆 | 42 | 76 |
| 信号 : 效果 | **14.8 : 1** | **11.9 : 1** |

**中间那块的最后一行就是机制。**覆盖按精确的 `(user_id, subject, predicate)` 三元组匹配。
抽取器给每条事实现编一个谓语字符串，于是"这条替换了更早的事实"这个信号，和它本该替换的
那条事实，落进**同一命名空间下的不同键**——96–97% 的孤立信号旁边就躺着同一 subject 的
兄弟谓语。消解器因此永远不触发。

直接佐证：在同一命名空间、同一 subject 下，train150 有 **17 组单复数撞名**——
`assistant_recommendation` vs `assistant_recommendations`、`recipe_recommendation` vs
`recipe_recommendations`、`aquarium` vs `aquariums`。同一个概念，两个键，互相看不见。

键的形状对的时候机制是好的：heldout100 上有 37 个键、train150 上有 71 个键确实发生了覆盖。

**必须谨慎表述：这是一个有证据的机制假设，不是已证明的原因。**它解释了为什么 622 个信号
只换来 42 次覆盖，也与那五个失败的形状吻合。但它还没有证明**正是这些键**让那五道题答错——
那需要一个把单个失败连到键形状上的实验，而这个实验还没有做（见第 12 节）。

> **一处更正。**本节的早先版本按 `(subject, predicate)` 分组，得出"全库 43% 落在两个键上"。
> 那是错的：这里每道题是独立命名空间，而消解器按 `(user_id, subject, predicate)` 匹配。
> 按正确的键分组，集中度其实很轻——最大的键只占 0.4%——真正的发现是上面这个，它既更干净
> 也更有力。本节数字来自对 `stores/heldout100.db` 和 `stores/train150.db` 的只读 SQL。

---

## 4. 存储与索引

用户反馈这一章原来讲得最不清楚，所以这里从"SQLite 到底是什么"开始讲。

### 4.1 SQLite：它不是一个数据库服务器

**常见的误解。** 提到"数据库"，多数人想到的是 PostgreSQL 或 MySQL：一个独立运行的进程，
监听一个端口，应用通过网络连上去。**SQLite 不是这样的。**

SQLite 是**一个 C 语言库**，被链接进你的应用进程。它没有服务器进程、没有端口、没有网络
协议。你的整个数据库就是磁盘上的**一个文件**。当 Python 执行 `conn.execute("SELECT …")`，
发生的事情是：Python 进程内的 SQLite 库解析 SQL、在同一个进程里执行、直接对那个文件
`read()` / `write()`。没有任何进程间通信。

在本项目里这条链是：

```
HTTP 客户端
    │  (网络)
    ▼
FastAPI 应用          ←─ 这是唯一的服务器进程
    │  (普通函数调用)
    ▼
MemoryService         ←─ 组装根，REST / MCP / Inspector 共用同一个对象
    │  (普通函数调用)
    ▼
SQLiteMemoryStore     ←─ store/sqlite.py
    │  (进程内 C 库调用)
    ▼
sqlite3 库
    │  (文件系统 read/write)
    ▼
stores/<name>.db      ←─ 一个普通文件
```

**FastAPI 与 SQLite 是什么关系。** FastAPI 是应用后端，负责 HTTP、路由、校验、序列化。
SQLite 是它进程内的持久化层。二者不是"应用 ↔ 数据库服务"的关系，而是"应用 ↔ 它自己调用的
一个库"的关系。这解释了本项目一个重要约束：**并发能力由进程模型决定，而不是由数据库配置
决定**（见 4.2）。

**数据实际在哪里。**

| | 路径 |
|---|---|
| 主数据库文件 | `stores/<store-name>.db`，例如 `stores/train150.db`（185MB） |
| 向量文件 | `stores/<store-name>-index.npy` |
| 向量 id 表 | `stores/<store-name>-index.ids.json` |
| 摄入检查点 | `stores/<store-name>-ingest.json` |

**Docker 下的映射关系。** `docker-compose.yml`：

```yaml
environment:
  LLTM_STORE_DIR: /data/stores
volumes:
  - ./stores:/data/stores        # 宿主机目录 → 容器路径
  - ./configs:/app/configs:ro    # 配置只读挂载
```

也就是说容器里的 `/data/stores/train150.db` 和宿主机上的 `./stores/train150.db`
**是同一个文件**。compose 文件自己的注释指出：这样做是为了开发时容器和本地 `lltm` 命令看到
同一份数据，而这**恰恰是生产环境不该做的**——因为 SQLite 是单写的，两个进程同时写同一个
文件正是要避免的情况。

**主要的表。**

| 表 | 内容 |
|---|---|
| `sessions` / `turns` | 原文归档 |
| `memories` | 结构化记忆 |
| `memories_fts` / `turns_fts` | FTS5 虚拟表（倒排索引） |
| `entities` / `memory_entities` | 归一化实体，第五路检索信号的数据 |
| `evidence` | 合并出的记忆由哪些原始记忆支撑 |
| `meta` | store 级别的元数据（抽取器指纹等） |

### 4.2 WAL：预写日志

`schema.sql` 第一行就是：

```sql
PRAGMA journal_mode = WAL;
```

**默认模式（rollback journal）怎么工作。** 写事务开始时，SQLite 先把**要被修改的原始页**
复制到一个 journal 文件，然后直接改主 `.db` 文件。如果崩溃，用 journal 把原始页写回去。
后果：写入期间主文件处于不一致状态，**读者必须被阻塞**。

**WAL 怎么工作。** 反过来：主 `.db` 文件在事务期间**完全不动**，新的页被追加写到一个
`-wal` 文件里。读者继续读主文件（加上它们各自可见的那部分 WAL），**写者不阻塞读者，
读者也不阻塞写者**。

三个文件的关系：

| 文件 | 作用 | 生命周期 |
|---|---|---|
| `train150.db` | 主数据库 | 永久 |
| `train150.db-wal` | 尚未合并回主文件的新页 | checkpoint 后清空 |
| `train150.db-shm` | 共享内存索引，让多个连接知道 WAL 里哪一页是最新的 | 最后一个连接关闭时删除 |

**checkpoint** 就是把 `-wal` 里的页合并回 `.db` 并截断 WAL 的动作，可以自动触发也可以手动。
本项目 `stores/` 目录里能看到 `two-stage-p10.db-shm` 和大小为 0 的 `.db-wal`，那是正常
状态：WAL 已经被 checkpoint 过，共享内存文件还挂着。

**为什么 WAL 仍然不等于多写。** WAL 解决的是**读写并发**，不是**写写并发**。SQLite 在任
何时刻仍然只允许**一个**写事务。第二个写者会拿到 `SQLITE_BUSY`。

**所以本项目还需要一个跨进程锁。** `locking.py` 提供一个基于 pid 的咨询锁，摄入和评测都
必须先取得它。这个锁不是理论上的谨慎，是一次真实事故的产物：

> 两个 shell 各起了一次摄入，配额账目记了 211 次请求，实际打出去约 320 次。store 里没有
> 任何一处看起来不对——损坏的是**统计**，而不是数据。

锁是咨询式的、按 pid 的，它拦得住"从另一个终端再起一次"，而那正是实际发生过的情况。
它拦不住恶意进程，也不是分布式锁。

### 4.3 FTS5：全文检索与倒排索引

**倒排索引是什么。** 普通索引是"给定行，找它的列值"。倒排索引反过来：

```
正排： memory_42  → "The user stopped drinking coffee"
倒排： "coffee"   → [memory_42, memory_87, ...]
      "drink"    → [memory_42, memory_11, ...]
```

查 "coffee" 时不需要扫全表，直接查倒排表拿到候选行。

**FTS5 虚拟表。** SQLite 的全文检索模块。"虚拟表"意思是它看起来像表、可以用 SQL 查，
但背后不是普通的行存储，而是倒排索引结构。本项目的定义：

```sql
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    content='memories',          -- external content：不复制原文
    content_rowid='rowid',
    tokenize='porter unicode61'
);
```

三个细节：

- **`content='memories'`（外部内容表）**：FTS5 只存倒排索引，**不再复制一份 content 文本**。
  查询时通过 `rowid` 回到 `memories` 表取原文。省掉一份全文副本。
- **`tokenize='porter unicode61'`**：`unicode61` 按 Unicode 规则切词并折叠大小写；
  `porter` 是词干还原器，让 `drinking` / `drinks` / `drink` 归到同一个词元。
- **同步靠触发器**：外部内容表不会自动跟随主表变化，所以 schema 里为 `INSERT`、`DELETE`、
  `UPDATE OF content` 各定义了一个触发器（`memories_ai` / `_ad` / `_au`）来维护索引。
  `turns_fts` 有一组对应的触发器。

**BM25 排序。** FTS5 内置 `bm25()` 函数。本项目直接调用它，**没有自己实现 BM25**：

```sql
SELECT m.id AS id, bm25(memories_fts) AS score
FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid
WHERE memories_fts MATCH ? AND m.user_id = ? AND m.status = 'active'
ORDER BY score LIMIT ?          -- bm25() 是负数，越负越好
```
（`store/sqlite.py:414`）

⚠️ **实现细节：FTS5 的 `bm25()` 返回值是标准 BM25 分数的负数。** 所以"最相关"对应
**最小**值，`ORDER BY score` 用的是**升序**。这个符号约定在下游归一化时还会再出现一次
（见 5.3）。

**它用在哪两条路径上。**

| 索引 | 用途 |
|---|---|
| `memories_fts` | 混合检索的词法那一路（第二路信号） |
| `turns_fts` | 原文回退的档案级检索（`search_turns`） |

### 4.4 为什么是 SQLite + WAL + FTS5，而不是别的

不做泛泛比较，只按本项目的实际规模和部署形态比。**规模事实**：单机、单进程写入、
最大 store 18,017 条记忆 / 185MB、并发需求为零（评测是串行的）。

| 方案 | 它会带来什么 | 为什么本项目不用 |
|---|---|---|
| **SQLite + FTS5**（当前） | 一个文件、零服务依赖、BM25 免费、事务和外键齐全 | — |
| **PostgreSQL** | 真正的多写并发、更强的类型和约束、成熟运维 | 需要一个常驻服务、连接配置、迁移工具。本项目并发需求为零，付出的是每个开发者和每次 CI 都要起一个数据库。**如果做多租户生产部署，这是第一个要换的东西**（见第 12 节） |
| **pgvector** | Postgres 内的向量索引，SQL 和向量在同一事务里 | 同上，且本项目的向量规模（18k × 384 float32 ≈ 27MB）精确搜索是毫秒级，索引结构买不到东西 |
| **Elasticsearch** | 更强的全文检索、分布式、丰富的分析器 | 一个 JVM 服务、集群运维、以及一份数据副本。schema.sql 的注释直接写了：FTS5 免费给了 BM25，替代方案是搭一个 Elasticsearch，"在这个规模上不值一个服务依赖" |
| **外部向量数据库**（Pinecone/Weaviate/Qdrant 等） | 托管、可扩展、丰富的过滤 | 引入第二个持久化系统，于是"记忆行"和"它的向量"分处两地，需要跨系统一致性。本项目的删除、命名空间隔离、`status` 过滤全部依赖它们在同一个事务里 |

**一句话总结取舍：** 当前选择用"无法多写"换来了"零运维、单文件、可整体复制、可 checksum、
可在 CI 里凭空创建"。对一个每个实验都要冻结、哈希、复现的项目，最后这几条是主要收益。
一旦变成多租户服务，这个取舍就反过来了。

### 4.5 向量存储与精确搜索

**嵌入在哪里生成。** 本机，不走 API。模型 `sentence-transformers/all-MiniLM-L6-v2`，
维度 **384**（`configs/fallback.yaml` 的 `embedding_dim: 384`，也可由
`encoder.dim` 从模型读出）。设备自动选择 MPS → CUDA → CPU。

理由是配额：全语料嵌入约 **5,300 万 token**，而每日请求数正是本项目的绑定约束。把嵌入放到
API 上会和抽取抢同一个池子。

**向量存在哪里。** 不在 SQLite 里，而在两个并列的文件：

```
stores/train150-index.npy         # (n, 384) float32 矩阵，np.save
stores/train150-index.ids.json    # 长度 n 的 memory id 列表，行号对应
```

**是否做 L2 归一化——代码确认：是，而且做了两次。**

```python
# embed/encoder.py
self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, ...)

# store/vector.py :: add()
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / np.maximum(norms, 1e-12)

# store/vector.py :: search()
q = q / max(float(np.linalg.norm(q)), 1e-12)
```

编码器已经归一化了，索引在 `add` 和 `search` 里又各归一化一次。这是刻意的冗余，注释写明
理由："放在这里而不是编码器里，**这样每个后端都得到同样的保证**"——即索引不信任上游。

**精确搜索怎么做。**（`store/vector.py :: search`）

```python
scores = self._vectors @ q  # (n, 384) @ (384,) = (n,)
k = min(limit, len(self._ids))
top = np.argpartition(-scores, k - 1)[:k]  # O(n) 选出前 k，不排序
top = top[np.argsort(-scores[top])]  # 只对这 k 个排序
```

一次矩阵-向量乘法把全库分数算出来，然后用 `argpartition` 做 O(n) 的 top-k 选择，只对选出的
k 个做排序（O(k log k)），避免对 n 个元素做全排序。

**为什么不用 FAISS / HNSW / ANN。** `vector.py` 的模块注释给了两个不同层次的理由：

1. **FAISS 的 `IndexFlatIP` 就是暴力内积扫描**，跟上面那行 `@` 做的事一模一样。在这个规模
   （< 100 万向量）差别只是一个常数因子，而瓶颈根本不在这里——**瓶颈是 LLM 调用**。
   引入依赖买不到东西。
2. **近似索引（HNSW / IVF）被排除的理由不同**：它们的召回噪声**与记忆算法本身的回归不可
   区分**。本项目所有结论都建立在消融对比上，一个会随机丢候选的检索层会污染整张消融表。

| | 精确（当前） | 近似（HNSW/IVF） |
|---|---|---|
| 召回率 | 100%，确定性 | < 100%，随参数变化 |
| 延迟 | O(n·d)，18k × 384 是毫秒级 | 亚毫秒 |
| 对消融实验 | 干净 | 噪声与被测变量混淆 |

**限制。** 向量文件和 SQLite 是两个文件，一致性靠流水线保证而不是靠事务。本项目因此有一个
显式的状态检查（`scripts/check_ingest_state.py`），每次恢复摄入前联合校验
checkpoint / 原文归档 / SQLite 完整性 / 记忆数 / 索引 id 数 / 向量行数 / 抽取器指纹
是否全部一致。当前 train150 上这七个数字都吻合（18,017 条记忆 = 18,017 个 id = 18,017 行
向量，维度 384）。

---

## 5. 读取通路：一个问题如何变成答案

全程跟踪一个具体查询：

```
user_id = "alice"
query   = "What was the Mayo Clinic YouTube video you recommended?"
```

### 5.1 查询表示

查询文本走**两条**路，因为两种检索需要不同的输入形式：

| 用途 | 形式 |
|---|---|
| 语义检索 | `encoder.encode_one(query)` → 384 维 float32 向量，已 L2 归一化 |
| 词法检索 | 原始文本 → `_fts_match()` 转成 FTS5 的 MATCH 表达式 |

用的是**同一个编码器**、同一个模型。查询和记忆被映射进同一个向量空间，这是"相似度"有意义
的前提。

**嵌入是什么。** 一个把文本映射成定长实数向量的函数 $f_\theta$：

$$\mathbf{e} = f_\theta(x) \in \mathbb{R}^{384}$$

$\theta$ 是模型参数（`all-MiniLM-L6-v2` 的权重，本项目不训练它，只推理）。它的训练目标让
语义相近的句子落在相近的位置，所以"喝什么"和"戒了咖啡"即使没有共同词也会靠近——这正是词法
检索做不到的事。

### 5.2 语义检索

**公式。** 两个向量的余弦相似度：

$$\cos(\mathbf q, \mathbf m) = \frac{\mathbf q^\top \mathbf m}{\lVert \mathbf q\rVert\,\lVert \mathbf m\rVert}$$

| 变量 | 含义 |
|---|---|
| $\mathbf q$ | 查询向量，384 维 |
| $\mathbf m$ | 一条记忆 `content` 的向量 |
| $\mathbf q^\top \mathbf m$ | 内积，$\sum_{i=1}^{384} q_i m_i$ |
| $\lVert\cdot\rVert$ | L2 范数，$\sqrt{\sum_i v_i^2}$ |

**本项目的实现是归一化内积。** 4.5 已用代码确认所有向量在写入和查询时都被 L2 归一化，
因此 $\lVert\mathbf q\rVert = \lVert\mathbf m\rVert = 1$，于是

$$\lVert \mathbf q\rVert = \lVert \mathbf m\rVert = 1 \;\Longrightarrow\; \cos(\mathbf q,\mathbf m) = \mathbf q^\top \mathbf m$$

代码里那一行 `scores = self._vectors @ q` 因此**直接就是**余弦相似度，不需要再除以模长。
取值范围 $[-1, 1]$。

**归一化到 [0,1]。** 融合前要把这个分数搬到 $[0,1]$：

$$S_{\text{sem}} = \mathrm{clip}_{[0,1]}\!\left(\frac{\cos(\mathbf q,\mathbf m) + 1}{2}\right)$$

（`retrieve/hybrid.py`：`semantic=_clip((raw + 1.0) / 2.0)`）

**候选集怎么取。** 语义那一路先在**整个索引**上搜索（`limit=len(self.index)`），然后过滤
命名空间、剔除 `evicted`、（在时序模式下）只留 `active`，**过滤之后**再截断到
`candidate_limit`（默认 50）。顺序很重要：先截断再过滤会让被过滤掉的行白白占用名额。

### 5.3 BM25 / 词法检索

**BM25 是什么。** 一个经典的词袋相关性打分函数。它不理解语义，但对**精确 token 匹配**
极强——URL、型号、专有名词、数字，正是嵌入最容易糊掉的东西。

$$\mathrm{BM25}(q,d) = \sum_{t \in q} \mathrm{IDF}(t)\cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\,\dfrac{|d|}{\overline{dl}}\right)}$$

| 变量 | 含义 | 直觉 |
|---|---|---|
| $t$ | 查询里的一个词元 | 分词后的一个 token |
| $f(t,d)$ | **词频**：$t$ 在文档 $d$ 中出现的次数 | 出现越多越相关，但边际递减 |
| $\mathrm{IDF}(t)$ | **逆文档频率** | 罕见词更有区分度；"the"几乎不加分 |
| $\lvert d\rvert$ | 文档长度（词元数） | — |
| $\overline{dl}$ | 全库平均文档长度 | — |
| $k_1$ | 词频饱和参数 | 控制"出现 10 次"比"出现 2 次"强多少；越小越快饱和 |
| $b$ | 长度归一化强度，$b\in[0,1]$ | $b=1$ 完全按长度惩罚长文档，$b=0$ 不惩罚 |

IDF 常用形式：$\mathrm{IDF}(t) = \ln\!\left(\dfrac{N - n_t + 0.5}{n_t + 0.5} + 1\right)$，
$N$ 是文档总数，$n_t$ 是含 $t$ 的文档数。

**本项目调用的是 SQLite FTS5 的 `bm25()`，没有自己实现。** 两个实现细节必须知道：

1. **符号相反。** FTS5 的 `bm25()` 返回标准分数的**负数**，越负越相关。所以 SQL 里是
   `ORDER BY score`（升序）。
2. **参数不可见。** $k_1$ 和 $b$ 由 FTS5 内部固定，本项目没有配置它们，代码里也没有出现。
   **当前材料无法确认 FTS5 使用的具体 $k_1$ / $b$ 取值。**

**归一化。** BM25 的值域是负的、无上界的，而余弦是有界的。直接相加会让词法项**按实现细节
而不是按配置的权重**去支配或消失——`hybrid.py` 的模块注释明确说这不是美化，是必要的。
所以对**本次查询实际观察到的候选**做 min-max：

$$S_{\text{lex}} = \frac{\max_j r_j - r_i}{\max_j r_j - \min_j r_j}$$

其中 $r_i$ 是候选 $i$ 的原始 FTS5 分数。分子写成"最大减当前"是因为 `lower_is_better=True`
——分数越负越好，所以最负的那个得到 1.0。

⚠️ 这是一个**相对**归一化：它衡量的是"在本次返回的候选里排多好"，不是绝对相关性。
如果所有候选都很差，最差的那批里最好的一个仍然拿到 1.0。代码里还有一个特例：当
`min ≈ max`（只有一个候选或全部同分）时，全部返回 1.0，注释写的是"一个没有歧义的命中拿到
满分"。

### 5.4 五路信号与融合

**通用打分公式：**

$$S(m,q) = w_s S_{\text{sem}} + w_l S_{\text{lex}} + w_r S_{\text{rec}} + w_i S_{\text{imp}} + w_e S_{\text{ent}}$$

五路信号各自的定义（全部先 clip 到 $[0,1]$）：

| 信号 | 公式 | 说明 |
|---|---|---|
| **semantic** | $(\cos + 1)/2$ | 见 5.2 |
| **bm25**（词法） | min-max 归一化的 FTS5 BM25 | 见 5.3 |
| **recency** | $S_{\text{rec}} = \exp\!\left(-\ln 2 \cdot \dfrac{\Delta_{\text{days}}}{H}\right)$ | 指数半衰。$H$ = `recency_halflife_days`，默认 **30 天**。$\Delta_{\text{days}}$ 从 `event_time`（缺失时用 `ingested_at`）算起。$\Delta = H$ 时刚好得 0.5 |
| **importance** | 直接取 `memory.importance` | 写入时赋的 $[0,1]$ 值 |
| **entity** | $S_{\text{ent}} = \max\limits_{e \in E(m)} \dfrac{\lvert T(e)\cap T(q)\rvert}{\lvert T(e)\rvert}$ | $E(m)$ 是这条记忆关联的实体，$T(\cdot)$ 是分词（`[a-z0-9]+`，长度 > 2）。取**最大**而不是平均：一个实体完全命中就够了 |

融合后可选地乘一个强度因子：

$$S_{\text{final}} = S(m,q)\cdot \big(\text{strength}\big)^{[\,\texttt{use\_strength}\,]}$$

`use_strength` 默认 `False`，**当前生产配置未启用**。

**排序与去重。** 排序键是 `(-score, -semantic_raw, memory.id)`——分数相同时用原始余弦分
再比，还相同就用 id。这保证**同一个 store 和同一个查询永远产生同一个上下文**；一个会随
运行变化的上下文，与它试图减少的回答器方差不可区分。

#### ⚠️ 当前生产配置里，只有一路信号在起作用

`config.py:100-104` 的默认权重：

```python
class RetrievalWeights(BaseModel):
    semantic: float = 1.0
    bm25: float = 0.0  # 词法信号，代码里的字段名是 bm25
    recency: float = 0.0
    importance: float = 0.0
    entity: float = 0.0
```

`configs/fallback.yaml` 的 `retrieval:` 块**只设置了 `top_k: 20`，没有覆盖任何权重**。

所以当前冻结配置的实际打分是：

$$S(m,q) = 1.0\cdot S_{\text{sem}} + 0\cdot S_{\text{lex}} + 0\cdot S_{\text{rec}} + 0\cdot S_{\text{imp}} + 0\cdot S_{\text{ent}} = S_{\text{sem}}$$

**"架构支持"和"生产启用"是两件事，本报告不把它们混写。** 准确的表述是：

> 系统实现了五路信号的加权融合框架，五路信号在每次检索时都被**计算并记录**（出现在
> `signals` 字段里，Inspector 和 API 的 `explain` 都能看到），但**当前冻结配置只给语义
> 一路非零权重**。因此现在的排序等价于纯语义检索。另外四路是**已实现、未测量**，不是
> 已测量、被否决。

有一个后果值得指出：BM25 仍然影响**候选集**（词法命中会被并入候选），只是不影响**排序**。
所以词法路径不是完全不起作用，它的作用是召回而不是排序。

### 5.5 候选过滤与被拒记录

一次检索的漏斗：

```
全库向量搜索
   → 过滤 user_id ≠ namespace          （命名空间隔离）
   → 过滤 status = 'evicted'           （已删除）
   → 时序模式下只留 status = 'active'  （已被覆盖的不参与）
   → 截断到 candidate_limit = 50
   ∪  BM25 前 50
   → 五路打分、排序
   → （可选）重排
   → 截断到 top_k = 20
```

**被拒的记录也要返回。** API 的 `explain` 模式返回一个 `rejected` 列表，每条带理由：

| 理由 | 含义 |
|---|---|
| `superseded` | 这条事实已经不成立了 |
| `below_rank` | 仍然成立，但分数没进 top_k |

设计理由写在 README 里："**一个没有理由的'没有结果'，跟一个 bug 长得一模一样。**"
同样的原则出现在上下文组装里（`dropped_sessions`）和检索追踪里（`RetrievalTrace`）。

**分阶段召回追踪。** `RetrievalTrace` 记录截断和重排**之前**的完整候选集。理由：一个
端到端的召回数字无法区分"某一阶段没找到"和"某一阶段把找到的删掉了"。如果候选召回是 100%
而重排后是 90%，那么重排器在删除正确证据——这跟"找不到"是完全不同的问题。这个仪表正是
后面能给每个模块算 oracle 上限的基础。

### 5.6 上下文组装

**v1：平铺 top-k。** 把 20 条记忆按分数顺序拼成一个列表交给回答器。

**问题：单条相关 ≠ 连贯的推理上下文。** 三月自行车维修的一条记忆，可能夹在六月雕塑课的两条
之间。模型要回答一个跨事件的问题时，拿到的是一份被打散的时间线。

一个五题探针（每题每臂跑 3 次，共 45 次回答）测了三个臂：

| 臂 | 上下文 | 三次运行一致的单元格 |
|---|---|---:|
| `flat20` | 生产配置，top_k=20，按分数排 | 3 / 5 |
| `flatN` | 条数与 coherent 相同，仍按分数排、仍打散 | 3 / 5 |
| **`coherent`** | gold 会话的记忆，按事件顺序 | **5 / 5** |

`flatN` 是那个便宜的假设——如果只要削减 top_k 就能拿到收益，修复不花一分钱。**结果是它更差**：
在其中一题上 `flatN` 1/3 而 `flat20` 3/3，上下文只有三分之一。**变量不是给了多少条记忆，
是它的形状。** 而 `coherent` 买到的主要不是准确率，是**一致性**：它从没跟自己不一致过。

⚠️ 两个限制让这还不能算产品结论：`coherent` 是 oracle（用了 gold session id），而且五道
因为答错才被挑出来的题不是一个样本。

**v2：会话连贯上下文。** 从检索结果出发重建同样的形状，不用 gold 标签：

```
检索结果（RetrievedMemory 列表）
  → 按 source_session_id 分组
  → 每个会话聚合成一个会话分
  → 会话按分排序
  → 取前 max_sessions 个
  → 每个会话内部：取出该会话【全部】active 记忆，按事件顺序排
  → 可选：只保留最佳命中前后 window_radius 条
  → 全局硬上限 max_total_memories，超了就【整个会话】不要
  → 会话之间按时间顺序铺开
```

**会话聚合公式。** 支持四种，由 `budget.aggregate` 选择：

$$A_{\max}(s) = \max_{m\in M_s} S(m,q) \qquad A_{\text{sum}}(s) = \sum_{m\in M_s} S(m,q)$$

$$A_{\text{mean}}(s) = \frac{1}{\lvert M_s\rvert}\sum_{m\in M_s} S(m,q) \qquad A_{\text{top3}}(s) = \sum_{j=1}^{3} S_{(j)}$$

| 变量 | 含义 |
|---|---|
| $s$ | 一个来源会话 |
| $M_s$ | 检索结果中来自会话 $s$ 的记忆集合（**不是**该会话的全部记忆） |
| $S(m,q)$ | 5.4 的融合分 |
| $S_{(j)}$ | $M_s$ 中第 $j$ 高的分 |

各自的偏好：`max` 奖励单个强命中；`sum` 奖励长会话而不管相关性；`mean` 奖励整体相关的会话；
`sum_top3` 是折中。

**两条设计约束，都有明确理由：**

- **窗口按事件顺序连续，不按分数。** 注释："窗口的意义在于**周围的事情解释了这件事**，
  一个按分数拼出来的窗口只是一个更小的平铺排序。"
- **要么整个会话，要么不要。** 硬上限被触发时**跳过**整个会话而不是截断它，因为半条时间线
  正是这个机制要摆脱的形状。被跳过的会话记进 `truncated` 和 `dropped_sessions`。

**当前状态，必须说清楚。** `configs/fallback.yaml` 里的 `context:` 块是
`max_sessions=3, window_radius=null, max_total_memories=20, aggregate=sum_top3,
session_order=chronological, include_superseded=false`。**这是起点，不是结论**——配置文件
的注释自己写着"These values are the starting point, NOT a result"。

train150 上跑出的候选是 `mean` 聚合、`window_radius=1`、`max_total_memories=30`，
在 145/150 道题上达到 95.2% Top-3 会话召回和 95.2% 装配后召回、中位 140 token。
**但它还没有被写进任何配置**，因为终结器要求 7,180 个会话全部完成才允许写出 `configs/v2.yaml`。

**v2 的目标目前是一致性，不是准确率。** 预注册的决策规则明确写了：`coherent-auto` 只要
**不降低**准确率且上下文不超过 +50% 就采纳，**不要求它涨分**。理由是探针观察到的效应是
一致性，而设一个这个设计测不出来的准确率门槛，会重复批大小预注册犯过的错误。
**v2 是否提高准确率，在 dev100 跑完之前是未知的。**

### 5.7 回答器

**输入。** 一个 system prompt + 组装好的记忆上下文 + 用户问题。

**记忆上下文的格式。** 不是裸列表，而是**按 `scope` 分组加小标题**
（`evaluation/runners/memory.py :: render_grouped`）：

```
About the user:
- The user lives in Sydney.

User preferences:
- The user prefers turbinado sugar in baking.

Previously recommended by the assistant:
- The assistant recommended a Mayo Clinic video about desk posture.

Other people and things discussed:
- Andy wore a blue shirt to the meeting.
```

完整的标题映射（`_SCOPE_HEADINGS`）：`profile` → "About the user"，`preference` →
"User preferences"，`plan` → "Current plans"，`event` → "Past events"，`recommendation`
→ "Previously recommended by the assistant"，`commitment` → "The assistant agreed to"，
`shared_context` → "Other people and things discussed"，无 scope → "Other things known
about the user"。

**为什么分组。** 一条约束（"用户对坚果过敏"）和一个候选答案（"用户喜欢杏仁蛋糕"）在裸列表里
长得一样。加了小标题之后，模型知道前者是**用来约束回答**的，后者是**可以被推荐**的。
预 P10 的记忆没有 scope，代码检测到全部无 scope 时退回裸列表——不猜测、不编造。

**system prompt**（`ANSWER_PROMPT_VERSION = "memory-aware-v2"`，`runners/base.py:41`）的
四段结构：

1. 你在用长期记忆回答用户的问题；
2. **这些记忆是关于用户的上下文，不一定是问题的直接答案**——能个人化、能约束、能改进回复时
   要显式地用上它，不相关的忽略；
3. 如果问的是一个具体事实而没有任何记忆包含它，说不知道，不要猜；**但不要仅仅因为没有记忆
   一字不差地写出答案就拒答**——如果记忆足以给出一个有用的、个人化的回复，就给；
4. 直接、简洁地回答。

第 2 段和第 3 段的后半句是一次真实修复的产物，见下文。

**充分性判定：`need_source` 是什么。** 第一次调用返回的**不是散文，而是一个结构化判定**
（`AnswerVerdict`）：

```python
status: Literal["answer", "need_source", "no_evidence"]
answer: str  # status == "answer" 时的回复
reason: str  # 缺什么，一句话
source_query: str  # need_source 时：用什么关键词去搜原文
```

| status | 含义 | 后续动作 |
|---|---|---|
| `answer` | 记忆足够 | 直接返回，**总共 1 次 LLM 调用** |
| `need_source` | 有记忆切题，但问的那个具体细节（URL、精确数字、产品名）没被保留 | 触发原文回退 + **第 2 次**调用 |
| `no_evidence` | 提供的东西跟问题无关 | 触发原文回退（档案级） |

**为什么要结构化判定而不是让模型直接写散文。** 注释说得很直接：只返回散文，就只能在
"每次都附原文"（实测 3 倍上下文换不到可测收益）和"永远不恢复"之间二选一。让模型说出它
**处在哪一种情况**，第二次调用就只为 `need_source` 付费。这是把 1.4 那个取舍变成一个
运行时决策。

#### 检索成功但答案仍然错

这是本项目观测到的一类重要现象，值得单列。已经确认：**检索成功了，证据进了上下文，答案
仍然错。** 最硬的证据是 dev50 的失败分层——14 个失败里 **S4 检索阶段是 0 个**，而 S4b
（进了上下文、排在最前，仍然没被用上）1 个、S5（全部供上且正确，答案仍错）2 个。
`knowledge-update` 那五个失败全部召回了 gold session **和** gold evidence。

至少要区分五种不同的原因，它们的负责模块完全不同：

| # | 类别 | 具体样子 | 归谁 |
|---|---|---|---|
| 1 | **上下文组装失败** | 两个日期分别排第 1 和第 2，模型把它们读成了同一天 | 组装 |
| 2 | **时序推理失败** | "我收到吊灯是几周前？"——需要日期算术 | 推理层 |
| 3 | **多会话聚合失败** | "我二月去了几个博物馆？"——没有任何一轮说了这个数 | 查询分解 |
| 4 | **知识更新解读失败** | 拿到了新旧两条事实，没能判断哪条现在成立 | 更新语义 |
| 5 | **随机生成** | 同样的输入，三次运行里有一次不一样 | 采样 |

**第 5 类被单独量化过。** 同一配置把 100 道题跑三遍：6/100 的题在三次之间翻转，正确率极差
3 个点。而且噪声不是弥散的——**六类题里有四类三次运行逐位一致**，所有不一致的题都是
`temporal-reasoning` 或 `multi-session`，也就是第 2、3 类。**查找是确定性的；算术和聚合
才是采样显形的地方。**

**目前已经做了什么**（全部已实现、已测量）：

- **回答 prompt 重写**：区分"这件事从没被提过"和"上下文里有料可用"。修复前，电池那道题
  系统**检索对了**（回复里点名了充电宝和充电板）却拒答"我不知道你用的是什么手机"——旧
  prompt 是照着"事实召回 + 该拒答就拒答"优化的，同一个脾气会让它在建议类问题上也拒答。
- **scope 分组**：见上。
- **判分器按题型路由**：偏好类题的 gold 是一段评分标准而不是参考答案，按参考答案比会把
  一个照着标准做对了的回复判错。
- **会话连贯上下文**（v2 候选）：针对第 1 类。
- **重复运行评测**：把第 5 类从"未知"变成一个数字。

**未来可能的方向——以下全部是 future work，当前系统没有实现：**

- 按题型路由到不同的回答策略；
- 把日期算术交给一个确定性计算器，而不是让模型算；
- 一个显式的 current / superseded 消解器，在进 prompt 之前就把"哪条现在成立"定下来；
- 结构化证据表（把事实以表格而不是散文交给模型）；
- 对不稳定题型升级到更强的模型。

### 5.8 条件式原文回退

**这个组件解决什么问题。** 抽取丢掉的细节，怎么找回来。

**输入。** `user_id`、`source_query`（回答器自己给的关键词）、以及本次检索到的记忆列表。

**输出。** 一个 `RawEvidence`：几条原文轮次、用的哪一级、以及理由。

**两个候选来源。**

| 级别 | 来源 | 适用情况 |
|---|---|---|
| `source_local` | 检索到的记忆所锚定的那些原文轮次（靠 `source_session_id` / `source_turn_index`） | 检索找对了记忆，只是细节没保住 |
| `archive_wide` | 对整个命名空间的 `turns_fts` 做 BM25 | 抽取整个漏掉了这段对话 |
| `none` | 两边都没有 | 仍然回答"不知道" |

**机制。**（`retrieve/fallback.py :: recover`）

```python
ranked = store.search_turns(user_id, query, limit=pool)  # pool = max(15, max_turns*5) = 15
local = store.turns_for_memories(memories)
found_the_conversation = bool(local) and (not ranked or ranked[0].session_id in local_sessions)
```

关键判据是：**本次查询的最佳证据，是否落在检索到的记忆所指向的那个会话里。**

- 是 → 记忆找对了对话，只是丢了里面的一个细节 → 用 `source_local`，并且**按档案排序给
  这些本地轮次排序**再截断。
- 否 → 抽取整个漏掉了那段对话，记忆无论看起来多自信都指错了地方 → 用 `archive_wide`。

**为什么按会话判定而不是按轮次。** 注释：在同一段对话里，BM25 经常更偏爱**用户的提问**而
不是**助手的回答**，因为提问会重复查询本身的词。按轮次粒度判定会把这读成"档案赢了记忆"，
从而抛弃一个其实已经找对地方的记忆。

**一次真实事故，以及一个有说服力的错误诊断。** 在干净的 store 上，本报告开头那个 Mayo
问题在一次**改善了检索**的重建之后失败了。

- **看起来的原因**：级别分支。第一版只要检索返回了任何东西就取本地轮次，只有检索为空才去
  档案，没有任何东西检查这些记忆是不是关于这个问题的。这**是**一个真实缺陷，但**不是这个**。
- **真正的原因是截断**。检索**确实**找到了正确的对话，gold 轮次就在候选里。但
  `turns_for_memories` 返回的轮次是**按 session id 排序**的，代码取了 `[:max_turns]`，
  而 gold 轮次在字母序里排第十（共十六条）——保留的那三条全部来自一段关于现场音乐的对话。
  **召回变好意味着候选变多，候选变多意味着答案被切掉了。**
- 在更小的混合 store 上，同一个问题检索不到东西、直接落到档案、从没遇到那个切片——这就是
  这个缺陷长期没被发现的原因。

两个缺陷是同一个疏漏：**没有任何东西拿候选去和问题比较**。所以对候选排序同时修好了两个。
`single-session-assistant` 从 5/6 → **6/6**，该臂从 66.0% → **72.0%**。

**没有任何一级授权编造。** 如果档案里也没有，正确答案仍然是"我不知道"。拒答率是一个被测量
的优点（本系统 100%，全量上下文 50%），一个把未命中变成自信猜测的回退会把它换掉。

**当前配置。** `fallback: enabled=true, max_turns=3, max_chars=2400`。渲染格式：

```
[session <id> · turn <index> · <role>]
<原文，受 max_chars 预算约束>
```

**回退到底贡献了什么——三个数字放在一起看：**

| | dev50 | heldout100 |
|---|---:|---:|
| 只靠结构化记忆答对 | 54.0% | 50.0% |
| 被原文归档救回 | **+18.0pp** | **+20.0pp** |
| **最终正确率** | **72.0%** | **70.0%** |
| 回退触发率 | 32% | 36% |
| 触发后答对的比例 | 56.2% | 55.6% |

**结构化记忆单独跑，正好打平 `naive_rag`（54.0%）。归档是把产品拉到基线之上的那一半。**
把 72.0% 整个引用而不拆分，会读成好像记忆层答对了全部——本报告不这样写。

---

## 6. 模型角色

四个角色，四个独立的模型 id，全部固定版本、**从不使用 `-latest` 浮动别名**——一个浮动别名
会让第 1 周的运行和第 8 周的运行悄悄变得不可比。

| 角色 | 模型 id | 输入 | 输出 | 职责 |
|---|---|---|---|---|
| **Extractor**（Stage A） | `gemini-3.1-flash-lite` | 15 个会话的原文 | 裸事实串 | 读懂对话，说出有哪些值得记的事实 |
| **Keyer**（Stage B） | 同上（同一模型，第二次调用） | Stage A 的事实串 | `temporal_key` + `update_op` + `object` | 给事实配时序键，判定它对历史做了什么 |
| **Answerer** | `gemini-3.5-flash-lite` | system prompt + 记忆上下文 + 问题 | `AnswerVerdict` | 被测系统本身。所有变体共用，保持恒定 |
| **Judge** | `gemma-4-31b-it` | 问题 + gold + 候选答案 | 对/错 | 评分 |
| （Embedder） | `sentence-transformers/all-MiniLM-L6-v2` | 文本 | 384 维向量 | 本机运行，不占 API 配额 |

**为什么必须分开——两个独立的理由：**

1. **判分器不能给自己的文字打分。** 如果回答器和判分器是同一个模型，它对自己的措辞习惯有
   系统性偏好，评测就不再独立。这是 `docs/DECISIONS.md` D4 的内容，而且判分器与人工的一致率
   是**被测量的**，不是假设的。
2. **免费额度是按模型计的。** 服务端返回的限流信息是
   `GenerateRequestsPerDayPerProjectPerModel-FreeTier`。三个角色放在三个不同的模型上，
   每个角色拿到自己的池子——**这才是一次大规模摄入不会饿死评测的原因**。

**为什么各角色选了不同规格。** 配置文件的注释给了实测理由：完整版 flash 模型在免费层每天
只有 20 次请求，连一次 50 题的运行都不够；flash-lite 是按**分钟**限流而不是按天。抽取是
重消费方（一次 dev 子集要约 610 万 token），需要 250k TPM 才能在几分钟而不是几小时内跑完。
判分器则相反：判分输入约 200 token，TPM 无关紧要，而 gemma 的每天 1,500 次请求正是判分
需要的。

**约束。** `quota: rpm=10, tpm=250000, rpd=500`；这些是**运行时探测**得到的，不是写死的，
因为服务端返回的限额与文档不一致且按模型变化（D12）。

---

## 7. 数学公式速查

正文里公式放在各自的组件章节，这里只做汇总索引，不重复推导。**只列本项目实际用到的。**

| 名称 | 公式 | 用在哪 |
|---|---|---|
| 嵌入 | $\mathbf e = f_\theta(x)\in\mathbb R^{384}$ | 5.1 |
| 余弦 / 归一化内积 | $\cos(\mathbf q,\mathbf m)=\dfrac{\mathbf q^\top\mathbf m}{\lVert\mathbf q\rVert\lVert\mathbf m\rVert}\;\xrightarrow{\ \lVert\cdot\rVert=1\ }\;\mathbf q^\top\mathbf m$ | 5.2 |
| 语义归一化 | $S_{\text{sem}}=(\cos+1)/2$ | 5.2 |
| BM25 | $\sum_{t\in q}\mathrm{IDF}(t)\dfrac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b\lvert d\rvert/\overline{dl})}$ | 5.3 |
| 词法归一化 | $S_{\text{lex}}=\dfrac{\max_j r_j-r_i}{\max_j r_j-\min_j r_j}$ | 5.3 |
| 时近 | $S_{\text{rec}}=\exp\!\left(-\ln2\cdot\Delta_{\text{days}}/H\right)$，$H=30$ 天 | 5.4 |
| 实体重叠 | $S_{\text{ent}}=\max_{e}\lvert T(e)\cap T(q)\rvert/\lvert T(e)\rvert$ | 5.4 |
| 加权融合 | $S=w_sS_{\text{sem}}+w_lS_{\text{lex}}+w_rS_{\text{rec}}+w_iS_{\text{imp}}+w_eS_{\text{ent}}$ | 5.4 |
| 会话聚合 | $A_{\text{mean}}(s)=\frac{1}{\lvert M_s\rvert}\sum_{m\in M_s}S(m,q)$ 等四种 | 5.6 |

### 评测统计量

**准确率**

$$\text{Accuracy}=\frac{\#\{\text{judge 判对的题}\}}{N}$$

**来源会话召回 Recall@k**

$$\text{Recall@}k=\frac1N\sum_{i=1}^{N}\mathbf 1\!\left[\,g_i\in R_i^{(k)}\right]$$

$g_i$ 是第 $i$ 题的 gold 会话，$R_i^{(k)}$ 是该题排名前 $k$ 的会话（或记忆的来源会话）集合，
$\mathbf 1[\cdot]$ 是指示函数。本项目在多个阶段分别测它（候选 → 排序后 → 选中 → 装配后），
因为一个端到端数字无法定位是哪一段丢的。

**精确 McNemar 检验**

配对实验的核心：两个系统跑**同一批题**，只看它们**不一致**的那些题。

|  | B 对 | B 错 |
|---|---:|---:|
| **A 对** | $a$ | $b$ |
| **A 错** | $c$ | $d$ |

$a$ 和 $d$（两边都对、两边都错）**不携带任何关于差异的信息**，被丢弃。只剩两个不一致格：
$b$（A 对 B 错）和 $c$（A 错 B 对）。在"两个系统无差异"的零假设下，每个不一致对独立地
以概率 $1/2$ 落入任一格，于是

$$b \sim \text{Binomial}(b+c,\ 0.5)$$

双侧精确 $p$ 值就是二项分布的双尾概率。

**为什么配对实验看的是不一致而不是两个准确率。** 两个系统各自 70% 和 72%，可能是
"完全相同的 70 题都对，B 多对了 2 题"（强证据），也可能是"A 对的 70 题里 B 错了 15 题，
另外又对了 17 题"（几乎没有证据）。只有不一致格能区分这两种情况。本项目的例子：回退臂
对基线是 9 胜 1 负，$b+c=10$，$p=0.022$。

⚠️ **本项目已经明确记录这个 $p$ 值站不住。** 它是每臂**单次运行**的结果，而重复实验测出
每 100 题有 6 题会在相同配置下翻转；50 题上期望约 3 次翻转，**一次翻转**就把 9 胜 1 负变成
8 胜 2 负、$p=0.109$。效应量很大，大概率是真的，但发表出去的显著性超出了单次运行能支撑的
范围。

**Fisher 精确检验**：用于**非配对**的两组比例比较，例如同一题型在 dev50 和 heldout100 上的
正确数。本项目所有 per-type 比较的 $p \ge 0.12$，即全部不可区分。

**风险比（risk ratio）**：用于零产出分析，$RR = p_{\text{后段位置}} / p_{\text{前段位置}}$。
观测数据给出 $RR = 2.86$，批聚类 bootstrap 95% CI 为 $[2.15, 4.35]$，组内位置标签置换检验
$p = 0.00005$（$N=20{,}000$）。⚠️ 这是**观测性**的：每个会话只坐过一个位置。真正的因果证据
来自后来那个随机对照实验。

---

## 8. 评测协议

### 8.1 基准：LongMemEval-S

500 道题，每道题配一份"干草堆"——几十个会话的对话历史，其中只有少数几个含有答案。六类题：

| 题型 | 要什么 |
|---|---|
| `single-session-user` | 用户在某一次对话里说过的事 |
| `single-session-assistant` | **助手**在某一次对话里说过的事 |
| `single-session-preference` | 应用用户的偏好给出个人化回复；gold 是一段**评分标准** |
| `multi-session` | 跨多个会话聚合 |
| `temporal-reasoning` | 日期算术 |
| `knowledge-update` | 事实变过，要用**现在**成立的那个值 |

**为什么选它，不选 LoCoMo。** 前者有真实的多会话时间跨度和这六类的显式划分，后者的会话是
合成的（D1）。

### 8.2 基线

| 基线 | 作用 |
|---|---|
| `full_context` | 参考点，**不是天花板**（D16）。它在 dev50 上 56.0%，证明"全塞进去"本身不解决问题 |
| `naive_rag` | 真正的对手：对话原文上的普通向量检索 |

### 8.3 数据集分工：这是本项目最重要的一条方法论

**先说踩过的坑。** dev50 只撑了两周就不能再用了。原因不是它小，而是：

> 在普通监督学习里，训练集吸收拟合，验证集只被少数几个模型选择决策碰到——那是一条很窄的
> 通道。这里**拟合的形式是 prompt 工程**，而 prompt 是靠**逐题读 dev50 的失败**写出来的：
> 抽取 prompt 从它们重写，回退设计从它们选定，闸门和阈值在它们上面调，六个模块因为它们被砍掉。
> 50 道题同时干着训练和验证两份活，走的还是最宽的通道。

**这就是 prompt tuning 造成验证集泄漏的机制**：它不需要梯度，只需要有人反复读同一批失败。

**冻结的协议**（2026-08-20，已验证两两交集为空、并集正好 500）：

| 集合 | n | 角色 | 能逐题看失败吗 |
|---|---:|---|---|
| `dev50` | 50 | v1 开发 —— **已烧穿** | 只作历史 |
| `heldout100` | 100 | v1 最终测试 —— **已花掉**，70.0% | 已花掉，再看不额外付出 |
| `train150` | 150 | **v2 开发** | **可以，不设限** |
| `dev100` | 100 | **v2 验证** | **不行**，只看聚合指标 |
| `test100` | 100 | **v2 最终测试** | **不行**，那一次运行之前不许看 |

规则的不对称是重点：`train150` 用来消耗（错误分析是消耗数据最快的动作，所以它最大），
`dev100` 只允许读聚合数和预先声明的切片，`test100` 只跑一次。

### 8.4 预注册

每个实验在跑之前写下：臂、指标、决策规则、以及一条**注册的预测**。三份预注册文件在
`results/prereg-*.md`。作用是让"结果不好看就换个读法"变得不可能——决策规则由代码在跑完
之后自动套用，写出一个哈希绑定的决策文件，没有事后人工重新解释的余地。

一个具体的收益：heldout100 单次运行 70.0%，三次均值 71.3%。**预注册要求报告单次那一枪，
所以这个项目亏了 1.3 个点。**预注册值钱就值钱在这个方向上。

### 8.5 重复运行

`heldout100` 的三次重复给出：正确率 70 / 71 / 73，极差 3 点；三次全对 69 题、三次全错 25 题、
**不一致 6 题**。协议因此改成：

- 100+ 规模的集合，在方差被刻画过之后，单次运行可以作为标题数字；
- **几十道题规模的配对结论，必须每臂 k 次运行并报出一致率**，否则报的是噪声。
- 这条**追溯适用**——仓库里每一个早于 2026-08-20 的配对结论都是单次运行。

### 8.6 报告哪些指标，为什么

| 指标 | 为什么必须报 |
|---|---|
| 准确率 | 最终结果 |
| 分题型准确率 | 但每格只有 3–27 题，**只作描述** |
| 来源会话召回（分阶段） | 定位失败发生在哪一层 |
| 上下文 token 中位数 | 产品主张是"比转录少 75 倍"，一个吃掉这个余量的改动是另一个产品 |
| 回退触发率 + 触发后正确率 | 拆开看归档到底贡献了什么 |
| API 请求数与 token | 成本 |
| 运行间一致率 | 上面这些数字有多少是运气 |

---

## 9. 实验与消融

**按被检验的设计问题组织，不按开发日期。**每个实验固定写：假设 → 对照 → 处理 → 指标 →
结果 → 解读 → 决定。**"没有采用"也是实验结果。**

### 9.1 抽取：v1 单阶段 vs 两阶段

| | |
|---|---|
| **假设** | 抽取质量是记忆系统的瓶颈 |
| **对照** | `chronomem`（v1 抽取） |
| **处理** | 两阶段抽取 + schema 拆分 + prompt 加示例 |
| **指标** | 配对准确率（31 题先导） |
| **结果** | **+29.0pp**，11 胜 2 负，**p = 0.022** |
| **解读** | 这是全项目最大的单项增益。来源会话召回在两臂都是 93.5%，所以增益不可能来自检索 |
| **决定** | 采用。抽取成为默认路径 |

### 9.2 证据策略：纯记忆 vs 常开挂载 vs 条件回退

| | |
|---|---|
| **假设** | 给回答器附上原文证据会提高准确率 |
| **对照** | 纯记忆 |
| **处理 1** | 常开挂载（`two_stage_hydrated`）：每次都附 |
| **处理 2** | 条件回退：只在回答器声明不足时附 |
| **指标** | 配对准确率 + 中位上下文 token |
| **结果** | 常开挂载 **+3.2pp**，2 胜 1 负，p = 1.000，**3 倍上下文**；条件回退 **+18.0pp**（54.0% → 72.0%），9 胜 1 负 |
| **解读** | 挂载的收益不可区分于零而成本确定；回退把同一笔钱只花在需要的 36% 上 |
| **决定** | 挂载降级为非默认；条件回退成为产品路径 |

### 9.3 检索：交叉编码器重排

| | |
|---|---|
| **假设** | 交叉编码器重排能改善排序 |
| **对照** | 纯混合检索 |
| **处理** | `cross-encoder/ms-marco-MiniLM-L-6-v2`，50 进 20 出 |
| **指标** | 逐题答案一致性 + 来源召回 |
| **结果** | k=10 时两臂 31 题**答案完全一致**（零处分歧）；重排在**每个** k 上都**降低**来源召回（k=20 时 93.5% → 90.3%） |
| **解读** | 它不改变答案，而且在删正确证据 |
| **决定** | **不采用。** 移入可选 `rerank` extra，默认关闭，torch 移出默认依赖 |

### 9.4 检索：只砍 top_k

| | |
|---|---|
| **假设** | 干扰项太多，减少条数就能改善 |
| **对照** | `flat20` |
| **处理** | `flatN`（条数减到 coherent 相同，仍按分数打散） |
| **结果** | 在其中一题上 `flatN` **1/3** 而 `flat20` **3/3** ——**更差**，上下文只有 1/3 |
| **解读** | 变量不是数量，是形状 |
| **决定** | 不采用。这个便宜假设先测再排除，省下了直接去建复杂方案的风险 |

### 9.5 原文回退：BM25 vs 稠密检索

| | |
|---|---|
| **假设** | BM25 在改述面前会崩溃，回退需要稠密检索 |
| **证据（初）** | 一组构造查询上 BM25 的 R@1 只有 **6.9%** |
| **审计** | **人工看了那些构造查询**，发现构造过程把**问题本身**删掉了——"How long have I been collecting vintage cameras?" 变成了 `"long"` |
| **重测** | 手写改述，gold 轮次命中**第 1 位**，包括不含任何源文本特有名词的表述 |
| **决定** | 证据作废。稠密检索**推迟，不是否决**，并写下了会重启它的条件（≥50 个已验证的可检索案例且 Recall@3 有明确增益） |

### 9.6 打包：学习出的效用预测器

| | |
|---|---|
| **假设** | 学一个"这条记忆有多有用"的预测器，比按相关性打包更好 |
| **指标** | 保留集 RMSE |
| **结果** | **0.310**，而"直接预测均值"是 **0.263** ——输给了常数基线 |
| **决定** | 不采用 |

### 9.7 抽取批大小与位置效应

见 3.2 的完整表格。摘要：

| | |
|---|---|
| **假设（原）** | 零产出是抽取器的随机波动 |
| **反证** | 观测：位置 0–3 零产出 6.4%，位置 4–14 是 18.3%，$RR=2.86$ |
| **随机对照** | 同一会话在正序和逆序中分坐首尾。前 4.5 条 / 后 1.6 条，**p < 0.0001** |
| **批大小** | 15 → 2.7 条/会话；5 → 4.8；**1 → 12.7，零产出 0.0%**；请求数 400 / 1,000 / 4,800 |
| **解读** | 零产出是连续衰减的尾巴，不是独立故障模式 |
| **决定** | **暂不改。** 改了要重建全部 store。写成 v2 的已知天花板，列为 v3 的题目 |

### 9.8 上下文形状（v2 候选，尚未验证）

| | |
|---|---|
| **假设** | 会话连贯的上下文提高**一致性**（不一定提高准确率） |
| **臂** | `flat20`（基线）/ `coherent-auto`（从检索重建）/ `coherent-oracle`（用 gold 会话 id，天花板） |
| **免费前置门** | train150 上会话召回@Top-3 必须 ≥80%，否则不花验证配额。当前 145/150 题上是 **95.2%** |
| **状态** | **dev100 尚未运行。准确率影响未知。** |
| **决策规则（已注册）** | 不降准确率且上下文不超 +50% 即采纳；若 auto 降而 oracle 不降，瓶颈在会话选择；若两者都降，关闭该方向 |

---

## 10. 按流水线阶段的失败分析

先定义分类法，再把事故作为案例放进去。这套分类法的价值在于**它反过来决定了工程优先级**。

### 10.1 分类法

| 阶段 | 含义 | dev50 失败数 |
|---|---|---:|
| **S0 来源** | 这个事实压根不在对话里 | 0 |
| **S1 抽取** | 它从来没变成一条记忆 | **10** |
| **S2 生命周期** | 变成了记忆，然后被合并或覆盖掉了 | 1 |
| **S3 资格** | 排序之前被过滤掉了 | 0 |
| **S4 检索** | 有资格，但没进上下文 | **0** |
| **S4b 组装** | 进了上下文而且排最前，仍没被用上 | 1 |
| **S5 推理** | 全部供上且正确，答案还是错 | 2 |
| **评测基础设施** | 系统没错，测量错了 | 见 10.4 |

**十四个失败里十个死在抽取，检索一个都没有。**

### 10.2 模块上限：在建之前先量它最多能修好几道题

| 模块 | 上限 | 实测 |
|---|---:|---|
| 抽取 | 10 | 试了 9 道，**修好 4 道** |
| **检索** | **0** | **没跑**——没有失败死在这里 |
| 组装 | 1 | 4 次运行里 3 次修好 |
| 推理 | 2 | 0 |

抽取的 oracle 是"把丢掉的事实直接注入，然后跑真实流水线"。九道只修好四道，说明**丢事实之外
还有第二个缺陷**——这个发现比"修好四道"更有价值，因为它说明光把抽取做好也到不了顶。

### 10.3 案例：S1 阶段

- **`single-session-assistant` 全类 0/4。**（3.2）schema 把 `subject` 当说话人开关，
  第三方事实无处安放。修复：拆成 `subject` / `source_role` 两个字段 + prompt 加示例。
- **批位置衰减。**（3.2 / 9.7）同一会话挪到批次后面产出只剩三分之一。**未修复，已量化。**
- **零产出。** 干净 store 上 14.5% 的实质会话一条记忆都没产出（304/2,096）。原计划的
  "重跑零产出会话"补救会给出**自信的错误答案**，因为重新组批等于把它们挪到了前面。

### 10.4 案例：评测基础设施

这一类不是系统错了，是**测量错了**，而它们和真实缺陷一样能让人做出错误决定。

- **一个 store 被两代抽取器写了，而没有任何东西发现得了。** checkpoint 记了九个进度计数器，
  **一个抽取器版本都没记**，而 `set_meta("extractor_version", …)` 是每次运行**覆盖写**、
  不是比对。结果：一个被 pre-P10 抽取器写到 63% 的 store 被 P10 抽取器续跑完成，两次运行
  都成功、所有结构性检查全过。两代的 `source_role` 分布是 93.3%/6.7% 对 44.8%/54.4%。
  **修复不是加版本号比对**——版本号是手填的，它抓不住"prompt 改了个词"这种更常见的漂移。
  改成从输入本身算指纹（两段 prompt 原文 + `schema.sql` + 模型 id + 每请求会话数 + 去重
  阈值），并报出**哪个部件动了**。那个 store 整个重建，旧的保留为诊断记录。
- **闸门打印"关闭"然后退出 0。** 人看着没问题，任何看返回码的自动化都看不见。现在闸门关闭
  一律非零退出。
- **陈旧产物不长得像错误。** 从 `temporal-gate.json` 读到 `gate_open: true`，不能证明**这次**
  运行的闸门过了——如果它在写文件前崩了，磁盘上是上一次的判决。现在按运行开始时间校验新鲜度。
- **一条报错猜了一个没验证过的原因。** `2348/2400 sessions — quota probably ran out`：
  配额根本没停过。这条报错被**明确撤回**并记在报告里，理由是"一条猜原因的报错比一条只报观测
  的报错更有害，因为它会把下一个人带向错误的方向"。同一处的完整性检查还比错了两个数——语料
  有 2,400 个会话**条目**但只有 2,348 个不重复**ID**，因为同一会话会作为多道题的证据。
- **两个进程同时摄入，配额账目被覆盖。** 记了 211 次、实际约 320 次。加了跨进程锁。
- **一个指标错了三次，比抽取器错的次数还多。** 保真度指标在抽取器被改对之前自己先错了两次。
- **会话 id 没做用户作用域**，导致会话召回算出 64%（把未完成的题算进分母，又拿内部 id 直接
  比公开 id）。修正后是 95%+，那个 64% 被明确作废。

### 10.5 这套分析对工程优先级的影响

**结论很直接：检索侧的改进天花板是 0 道题，而抽取和推理各有 10 道和 2 道。**

这解释了为什么本项目会：拒绝一个已经实现好的重排器；在 BM25 的负面证据上先做人工审计而不是
直接上稠密检索；把力气放在抽取 schema 和 prompt 上；以及为什么 v2 选的是**上下文组装**
（S4b/S5 那一侧）而不是继续调检索。

---

## 11. 工程与可复现性

这些机制放在这里而不是打断前面的架构说明。它们共同回答一个问题：**怎么保证一个数字是它
声称的那个系统产生的。**

| 机制 | 解决什么 | 实现 |
|---|---|---|
| **摄入指纹** | 续跑时代码已经变了 | 从 Stage A/B 的 prompt 原文、`schema.sql`、模型 id、每请求会话数、去重阈值算哈希；不一致时报出**哪个部件动了** |
| **结果版本化** | 一行结果不知道是哪个系统产生的 | 每行评测记录都带回答 prompt 版本、判分 prompt 版本、抽取器版本——**抽取器版本取自 store，不是取自代码检出**，因为它描述的是被评测的数据 |
| **冻结清单** | "n=50" 不是一个实验身份 | 报告的运行必须指名一个冻结的问题清单文件；`--limit` 只用于探索 |
| **内容寻址冻结** | 事后偷偷改配置 | 冻结记录把代码、配置、数据、预注册规则一起哈希；摄入后的冻结只有在这些哈希全部没变时才被接受 |
| **一次性 ledger** | 最终测试被跑第二次 | 状态变 `complete` 后运行器拒绝再执行 |
| **闸门非零退出** | 打印"关闭"却退出 0 | 所有闸门关闭时返回非零 |
| **产物新鲜度** | 读到上一次运行的判决 | 按本次运行的开始时间校验 |
| **跨进程锁** | 两个摄入互相覆盖 | `locking.py`，pid 咨询锁 |
| **配额中断安全** | 在途批次被误判成"零产出" | 配额/网络失败不给在途批次写终态、也不拿它做分析；内容策略拒绝保留原文并写显式终态；有自动化重放测试证明续跑不产生重复 |
| **只读状态检查** | 续跑前不知道 store 是否一致 | `scripts/check_ingest_state.py` 联合校验 checkpoint / 原文归档 / SQLite 完整性 / 记忆数 / 索引 id / 向量行数 / 抽取器指纹 |
| **测试与 CI** | — | **529 个测试**，ubuntu / windows / macos 三平台，行覆盖 80%（关键实验路径 88–100%） |
| **结构化日志** | — | 每请求一条 JSON：request id、延迟、配置指纹；**不含记忆内容、不含凭据** |

**对外的三个接口共用同一个服务对象**，所以它们不可能与被测量的行为漂移：

| 接口 | 形态 |
|---|---|
| REST | FastAPI，9 组端点；每次读都要 `user_id`；**跨命名空间读返回 404 而不是 403**——403 会确认这个 id 存在 |
| MCP | `search_memory` / `remember` / `search_conversations` / `get_timeline` / `forget`；每个工具都要显式 `user_id`，没有隐式会话身份 |
| Memory Inspector | 网页，显示每条记忆为什么被选中、跳过、覆盖，URL 可分享；四个 demo 是**录制的真实运行**，带 store 和 prompt 指纹，任一变化就显示 `stale` |

**Docker**：2.95GB，大部分是 PyTorch。默认 PyTorch 安装会拉 CUDA wheel（24.4GB 的 GPU
运行时，装在一个永远见不到 GPU 的容器里），所以 Dockerfile 装 CPU wheel 并删掉被替换后
孤立的 CUDA 包。镜像不烘焙任何数据集、模型、凭据或数据库——store 从挂载卷进来。

---

## 12. v2 现状与下一步

### 12.1 已经由实验支持的事实

| 结论 | 证据 |
|---|---|
| 结构化记忆是一种可用的压缩 | 上下文小 75 倍，开发集上准确率更高 |
| 增益主要来自抽取重写 | +29.0pp，11 胜 2 负，p = 0.022；两臂来源召回相同 |
| 常开挂载不值它的 token 成本 | +3.2pp，2 胜 1 负，p = 1.000，3 倍上下文 |
| 条件回退贡献显著 | 开发集 +18.0pp，保留集 +20.0pp |
| 结果泛化 | 保留集 70.0%（三次均值 71.3），与开发集差 -0.7pp |
| 交叉编码器重排在这里没用 | k=10 时逐题答案完全一致，且降低来源召回 |
| 批量抽取按位置丢信息 | 随机对照，p < 0.0001 |
| 单次运行噪声约 3 个点 | 三次重复，6/100 题翻转 |
| 噪声集中在两类题 | 六类里四类三次运行逐位一致 |

### 12.2 当前候选（尚未验证）

**会话连贯上下文。** train150 上（145/150 题）：Top-3 会话召回 **95.2%**、装配后召回
**95.2%**、中位上下文 **140 token**、1 题被截断。远高于注册的 80% 门槛。

⚠️ **这是代理指标。"找对了源会话"不等于"答对了题"。v2 是否提高准确率，在 dev100 跑完之前
是未知的。**候选配置（`mean` / `radius=1` / `cap=30`）**还没有被写进任何配置文件**——
终结器要求 7,180 个会话全部终态才允许写出 `configs/v2.yaml`。当前是 **6,983 / 7,180**。

### 12.3 开放问题（有数据支持，但没有结论）

| 问题 | 已知的 | 不知道的 |
|---|---|---|
| **抽取批大小衰减** | 批 1 是批 15 的 4.7 倍产出，因果已确立 | 机制未定：输出预算耗尽 / 枚举漂移 / 输入位置效应 / schema 长度压力，本实验一个都没分离 |
| **检索正确之后仍答错** | S4=0，S4b=1，S5=2；knowledge-update 五个失败全部召回了 gold | 五类原因（5.7）各占多少 |
| **谓语命名空间** | 全库 55.6% 落在永远无法覆盖的多值键上；88% 的替换信号独占一个键，其中 96% 的 `(user,subject)` 下还有别的谓语 | 机制有证据，但**尚未证明**它导致了已观测的那些失败 |
| **精确细节保真** | 反复出现的失败形状是"留下大意、丢掉标识符"（Mayo URL、`Garmin Forerunner`、`2-3 eggs`） | 带身份信息的片段是否应该原样保留而不是被改写 |
| ~~检索权重~~ **2026-08-25 关闭** | 离线实测：每加一路都掉分（-5.3pp 到 -17.3pp）；`recency` 是空操作，30 天半衰期对上 932–1,727 天的语料 | 一个**调过参**的混合是否可行。没有做权重网格，也没有交互搜索 |
| **回退深度** | `max_turns=3` 从未扫过；已知一个案例的答案在 BM25 第 6 位 | 取多深合适 |
| **保留集上没有基线** | `full_context` / `naive_rag` 从未在 heldout100 上跑过 | "结构化记忆打平普通 RAG"在未见数据上是否成立 |

### 12.4 未来假设——以下全部尚未实现、尚未验证

明确标记，不能读成当前系统已有的能力：

- **确定性时序计算器**：把日期算术从模型手里拿走，交给代码。针对 S5 里那一类。
- **按题型路由**：不同题型走不同的上下文组装和回答策略。
- **显式的 current/superseded 消解器**：在进 prompt 之前就把"哪条现在成立"定下来，
  而不是指望模型从一堆事实里推断。
- **结构化证据表**：把事实以表格而非散文交给模型。
- **更强模型升级**：对不稳定题型使用更强的回答器。
- **图表示**：把记忆之间的关系显式化。

### 12.5 接下来的执行顺序

| 步 | 动作 | 完成证据 | 是否花配额 |
|---:|---|---|---|
| 1 | 跑完 train150 最后 197 个会话 | checkpoint 与 store 均为 7,180/7,180 | 约 70 次调用 |
| 2 | `finalize_train150.py` | 最终零产出审计、七组固定网格、选型记录、`configs/v2.yaml` | **否** |
| 3 | 冻结并摄入 dev100 | 摄入前后两个哈希，后者是前者的严格扩展 | 约 1,400 次 |
| 4 | dev100 五臂 × 三重复 | 十五个组合全完成后出分，写出 `dev100-decision.json` | 约 2,050 次 |
| 5 | 冻结并摄入 test100 | 两个哈希，且绑定 dev100 的聚合与决策 | 约 1,400 次 |
| 6 | **一次性**跑完 test100 | 一份 ledger + 一张含全部保留臂的表 | 约 550 次 |
| 7 | 产品化 | 见 `docs/PRODUCTIZATION_V2_PLAN.md` | — |

第 2 步是分水岭。如果它判定 STOP（Top-3 或装配召回低于 80%，或上下文超过 flat20 的 1.5 倍），
按预注册，工作转向会话选择，**一分 dev100 配额都不花**。

### 12.6 产品化状态：这仍然是原型

| 项 | 现状 |
|---|---|
| 身份认证与账号隔离 | `user_id` 直接取自请求正文，不是可信令牌 |
| 完整删除与导出 | 只有软删除的 memory 状态，不等于数据真的没了 |
| 备份与恢复 | 有备份文件，没做过恢复演练 |
| 费用与额度监控 | 免费额度，没有可信价目表，因此不记美元金额 |
| 并发与幂等 | SQLite 单写 + 咨询式进程锁；多进程写入不承诺 |
| 隐私与安全 | 日志脱敏、密钥管理、保存期限均未定 |
| 上线运维 | 健康检查、告警、灰度、回滚均无 |

### 12.7 不承诺什么

- 不宣称打赢任何第三方系统。没有在同一协议下跑过别人的系统。
- 不宣称保留集上有基线对比。
- 不宣称单机 SQLite 支持多进程写入。
- 不把"软删除 memory 状态"说成用户数据已彻底删除。
- 不在没做过恢复演练的情况下说备份可靠。
- 不根据免费额度或过期价格猜真实费用。
- 不把仅靠请求正文里的 `user_id` 说成鉴权。
- 不把"架构支持五路信号"说成"五路都在生产中参与排序"。

---

## 附录 A：关键配置速查

全部来自 `configs/fallback.yaml`（当前冻结配置）。

```yaml
models:
  extractor:  gemini-3.1-flash-lite      # Stage A + Stage B
  answerer:   gemini-3.5-flash-lite      # 被测系统
  judge:      gemma-4-31b-it             # 独立池，不给自己打分
  embedder:   sentence-transformers/all-MiniLM-L6-v2
  embedding_dim: 384

quota:        rpm: 10   tpm: 250000   rpd: 500      # 运行时探测，非写死

retrieval:
  top_k: 20
  # 未覆盖权重 → 取 config.py 默认：semantic 1.0，其余四路 0.0
  # candidate_limit 默认 50，recency_halflife_days 默认 30

ingest:
  sessions_per_request: 15    # 已知有损，见 3.2
  two_stage: true
  checkpoint_every: 5
  dedupe_similarity_threshold: 0.92

temporal_resolution: true

fallback:
  enabled: true
  max_turns: 3
  max_chars: 2400

context:                      # 仅 two_stage_coherent 变体读取
  max_sessions: 3             # ⚠️ 起点，不是结论
  window_radius: null
  max_total_memories: 20
  aggregate: sum_top3
  session_order: chronological
  include_superseded: false
```

## 附录 B：代码位置索引

| 组件 | 文件 |
|---|---|
| Schema、索引、FTS5 触发器 | `src/llm_long_term_memory/store/schema.sql` |
| SQLite 存储、BM25 查询、原文检索 | `store/sqlite.py` |
| 精确向量索引 | `store/vector.py` |
| 本机编码器 | `embed/encoder.py` |
| 两阶段抽取 | `ingest/two_stage.py`、`extract_facts.py`、`keying.py`、`structure.py` |
| 出处锚点 | `ingest/provenance.py` |
| 摄入流水线、检查点 | `ingest/pipeline.py` |
| 抽取器指纹 | `ingest/fingerprint.py` |
| 零产出审计 | `ingest/zero_yield.py`、`zero_yield_report.py` |
| 时序消解与覆盖 | `temporal/resolve.py` |
| 混合检索与五路信号 | `retrieve/hybrid.py` |
| 会话连贯上下文 | `retrieve/coherent.py` |
| 条件原文回退 | `retrieve/fallback.py` |
| 交叉编码器重排（默认关闭） | `retrieve/rerank.py` |
| 回答器 prompt 与判定 | `evaluation/runners/base.py`、`runners/memory.py` |
| 基线 | `evaluation/runners/full_context.py`、`naive_rag.py` |
| 冻结与血统校验 | `evaluation/reproducibility.py`、`scripts/freeze_v2.py` |
| 跨进程锁 | `locking.py` |
| 配置定义与默认权重 | `config.py` |

## 附录 C：实验记录索引

| 内容 | 文件 |
|---|---|
| 30 条设计决策及理由 | `docs/DECISIONS.md` |
| 英文工程报告 | `docs/ENGINEERING_REPORT.md` |
| 数据集分工协议 | `results/data-protocol.md` |
| 批大小/位置的随机对照 | `results/batch-position-pilot.md` |
| 运行噪声的三次重复 | `results/heldout-variance.md` |
| 失败分层与模块上限 | `results/failure-stages.md` |
| 上下文形状的五题探针 | `results/context-arms.md` |
| 重排帕累托 | `results/rerank-pareto.md` |
| 原文检索诊断 | `results/raw-recall-diagnostic.md` |
| 三份预注册 | `results/prereg-batch-size.md`、`prereg-context-shape.md`、`prereg-v2-final.md` |
| v2 执行顺序与命令 | `results/v2-runbook.md` |
| v2 进度日志 | `results/v2-progress.md` |
| 产品化验收标准 | `docs/PRODUCTIZATION_V2_PLAN.md` |
