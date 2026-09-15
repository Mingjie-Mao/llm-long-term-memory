[![English](docs/badges/lang-en-idle.svg)](README.md)[![中文](docs/badges/lang-zh-active.svg)](README.zh-CN.md)

# llm-long-term-memory

为 LLM 应用提供持久记忆：从对话提取结构化事实，跟踪事实随时间的变化，并在抽取丢失关键细节时找回原始对话。

**它买到了什么，在一百道未见过的题上测过一次。** 冻结的 v2 在 `test100` 上答对 **72%**，中位上下文 **574 token**；整段历史答对 86%，用了 109,059 token——更准，但上下文是 **190 倍**。反过来的地方是时间：问「什么变了」的题，v2 拿 **74.1%**，整段历史只有 **23.1%**；知识更新 **75.0%**。原因是被推翻的事实在模型看到之前就已经解析成了时间线，而不是全都塞给它自己猜。

它**没有**被证明比朴素检索更准：对 naive RAG +7 个点，p = 0.3368。[结果与计量口径](#实验结果)。

[交互演示](https://lltm-memory.pages.dev) · [架构图集](docs/PROJECT_REPORT.zh-CN.md#系统的架构) · [实验沿革](docs/PROJECT_REPORT.zh-CN.md#实验历史) · [当前研究状态](docs/PROJECT_REPORT.zh-CN.md#现在在哪)

示例会把虚构事实发送到演示后端；输入框会发送用户输入的受支持陈述与问题。陈述由浏览器句式规则转成显式事实，后端执行本地嵌入、跨会话检索与时间更新；不展示 LLM 自动抽取或生成答案。

![系统总览：对话抽取与存储、独立问题查询、混合召回、候选记忆和条件原文回退](docs/figures/overview.zh-CN.svg)

[中文矢量图](docs/figures/overview.zh-CN.svg) · [English diagram](docs/figures/overview.svg) · [实现细节与可选分支](docs/PROJECT_REPORT.zh-CN.md#系统的架构)

## 快速开始

本地运行需要 Python 3.11+ 和 `uv`。在仓库根目录执行：

```bash
uv sync --group dev --extra api --extra llm --extra embed --extra mcp
uv run uvicorn llm_long_term_memory.api.app:app --host 127.0.0.1 --port 8000
```

打开 [Inspector](http://localhost:8000) 或 [API 文档](http://localhost:8000/docs)。本地嵌入模型若尚未缓存，会在首次使用时下载。在另一个终端验证：

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/v1/config
curl -X POST http://localhost:8000/v1/memories/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"alice","query":"where do I live?","explain":true}'
```

服务默认用 `configs/fallback.yaml` 打开 `stores/two-stage-p10.db`。**全新 checkout 不含对话数据**，没有填充 store 时搜索返回空列表。搜索需要嵌入模型，不需要 Gemini API key；调用 `/v1/answer` 在线回答，还需在环境变量或 `.env` 中配置 `GEMINI_API_KEY`。

**默认服务写入已接通。** 安装 `llm`、`embed` 依赖并配置 `GEMINI_API_KEY` 后，`/v1/messages` 和 MCP `remember` 会按需创建 `LiveTurnExtractor`，复用真实抽取、去重和时序消解。无 key 时仍可运行只读服务。批量导入入口见 `uv run lltm ingest run --help`；无需 key 的体验见[独立 playground](public-demo/README.md#run-it-locally)。

<details>
<summary>Docker 方式</summary>

```bash
docker compose up --build
curl http://localhost:8000/healthz
```

Compose 挂载 `./stores` 与 `./configs`。镜像包含代码和依赖，不内置数据集、数据库、凭据或嵌入模型权重。镜像内置默认配置，Compose 挂载可以覆盖。

</details>

## 工作原理

- **事实带出处。** 每条记忆包含 subject、predicate、object、scope 和说话人。`subject` 是事实关于谁，`source_role` 是谁说的；源会话引用和可选句子范围可用于回溯原文。
- **更新不抹掉历史。** 替换意图只关闭较早事实的有效区间，旧记忆行保留；时序消解按事件顺序重建受影响的键，并存意图下两条事实同时有效。当前日期列采用会话日期，`ingested_at` 记录写入时间。
- **先读记忆，必要时找原文。** 检索排序后组装事实上下文，由模型返回结构化判定。`need_source` 与 `no_evidence` 都可能触发恢复，只有找到原文才执行第二次回答。

| 默认路径 | 当前代码行为 |
|---|---|
| 抽取 | Stage A 事实 → Stage B key/更新意图 → 规则/出处；近邻去重可能增加模型请求 |
| 存储 | SQLite + FTS5；向量保存在 `<store>-index.npy` 和 `<store>-index.ids.json` |
| 检索 | 语义与词法候选取并集；计算五路信号，默认仅语义评分权重非零 |
| 候选预算 | 每路最多 50 条，合并后最多 100 条；打分后再取 top-k |
| 上下文 | 按排名选事实；服务默认 top-k 10，v2 评测 top-k 20 |
| 原文恢复 | source-local 或 archive-wide BM25；最多 3 轮、2,400 个原文字符 |
| 接口 | REST/MCP 共用 MemoryService；批处理与 playground 有各自写入路径 |

<details>
<summary>写入图：抽取、去重与时序更新</summary>

![批量写入路径：抽取、来源注册、模型判重、持久化、时序消解和检查点](docs/figures/write-path.svg)

非空的两阶段抽取通常需要两次模型请求，判重可能再增加若干次。0.92 相似度阈值用于找近邻，不会直接删除事实。[完整图注与代码定位](docs/PROJECT_REPORT.zh-CN.md#系统的架构)。

</details>

<details>
<summary>存储图：关系模型、出处与向量文件</summary>

![SQLite 表与单独保存的 NumPy 向量文件](docs/figures/storage.svg)

源会话外键与启发式轮次/字符锚点提供不同层次的保证。向量文件与 SQLite 分别保存。[完整图注与代码定位](docs/PROJECT_REPORT.zh-CN.md#系统的架构)。

</details>

<details>
<summary>读取图：候选排序与三种判定分支</summary>

![检索与条件回退：answer、need_source、no_evidence 分别进入直接回答或原文恢复](docs/figures/read-path.svg)

[完整六图图集](docs/PROJECT_REPORT.zh-CN.md#系统的架构) 还包含乱序写入时的时间更新，以及运行入口、演示和研究评测的边界。

</details>

重排、会话连贯上下文、无条件附加原文、衰减、合并和效用打包在代码里都有实现，但都不在默认回答路径上。v3 推理/证据恢复与 v4 合成/扫描是显式研究变体。见[分支对应表](docs/PROJECT_REPORT.zh-CN.md#系统的架构)和[关闭功能的证据](results/shipped-but-disabled.md)。

## 实验结果

**冻结 v2 终测：100 道 LongMemEval-S 题，每臂一次。** 这些是归档系统的结果，不代表当前所有研究变体。

| 臂 | 正确率 | 中位上下文 token | 回答 + 评分 token |
|---|---:|---:|---:|
| 整段历史 | **86.0%** | 109,059 | 10,932,294 |
| **v2 记忆 + 条件原文回退** | 72.0% | **574** | **159,458** |
| 原始会话 naive RAG | 65.0% | 12,763 | 1,352,847 |

v2 比 naive RAG 高 7 个点，但配对检验尚不足以判定优劣（`p=0.3368`）；整段历史比 v2 高 14 个点（`p=0.0043`）。[完整终测报告](results/final/test100-aggregate.md)。

上下文沿用当时评测脚本的口径：memory/RAG 是按字符估算，整段历史是 provider 返回的输入 token。最后一列来自实际 API usage，包含评分，不含共享抽取阶段的 **13,757,713 token**。上下文比值是近似量，不等于账单节省；终测每臂只跑一次，无方差可报。

**独立研究验证：v3.3 / dev60。** 每臂三次的多数票正确率为 55.0%，同期 v2 对照为 46.7%；中位上下文为 1,115 与 573 token，`p=0.1797`。这是刻意偏难的另一套题，不能把 55.0% 与上表 72.0% 直接比较。后续 schema 审计发现置信度恒为默认值，相应门限空转通过；准确率不受影响，另外八项门限仍可解释。[聚合结果](results/validation/v3-dev60.md)、[schema 审计](results/audit/v3-verdict-schema-never-sent-20260906.json)。

源会话召回率表示**至少命中一个 gold 会话**，不证明答案所需事实全部保留。开发结果、负结果以及已经测量的 v4 探针集中在[实验沿革](docs/PROJECT_REPORT.zh-CN.md#实验历史)。

## 接口

REST 配置 `LLTM_API_TOKENS` 后，从 bearer token 映射得到租户；请求中的 `user_id` 若与凭据不符，返回 403。未配置令牌时为开放模式，调用方自行指定命名空间；认证配置非空但格式错误时，服务拒绝启动。令牌目前由环境配置管理，没有自动过期或完整的轮换体系。

| REST 端点 | 用途 / 可用条件 |
|---|---|
| `GET /healthz` · `GET /v1/config` | 就绪状态与实际服务配置 |
| `POST /v1/memories/search` | 排序结果、信号、出处与可选被拒记录 |
| `GET /v1/memories` · `GET /v1/memories/{id}` | 浏览/检查 memory 与可用的来源证据 |
| `GET /v1/timeline` | 某个 subject/predicate 的历史 |
| `POST /v1/raw/search` | 搜索原始对话档案 |
| `POST /v1/answer` | 在线回答；需要 API key 与 LLM 依赖 |
| `POST /v1/messages` | 抽取并保存一轮消息；需 llm/embed 依赖与 API key |
| `DELETE /v1/memories/{id}` | 将 active memory 标为 evicted |
| `GET /v1/export` | 导出该命名空间的数据与来源 |
| `DELETE /v1/data` | 硬删除该命名空间在线数据库与向量数据；不清除备份 |

<details>
<summary>MCP 配置</summary>

```bash
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

工具：`search_memory`、`search_conversations`、`get_timeline`、`forget`、`remember`。全部需要 `user_id`；`remember` 与 REST 共用已接通的写入路径。MCP 暴露记忆工具，没有另外的生成答案工具。

MCP stdio 面向可信本地客户端；HTTP transport 尚无等同 REST 的凭据身份边界，应限制在本机使用。

</details>

## 记忆设计（精简）

只有一层，不是三层。**没有工作记忆**（任务状态、调过什么工具、第几步），**也没有短期记忆**——当前轮次由调用方自己放进 prompt。这里全部是长期记忆：用户是谁、说过什么、什么时候变的。

| 问题 | 已建成的 | 实测 |
|---|---|---|
| **存什么** | 5 种类型 x 7 种范围；范围由模型判定，未经人工核验 | — |
| **即时写入** | `POST /v1/messages` 同步抽取，说完就进库，不等总结；幂等键保护重试 | 没有优先级信号：救命事实和闲聊走同一条路 |
| **后台归并** | 相似度 0.84 以上、至少 3 条聚成一簇，保留证据链接 | **v2 关闭** |
| **冲突与状态** | 时间线按事件顺序从该键全部记忆重建，不打补丁；重述折叠到最早区间；不调模型 | 终测 knowledge-update **75.0%**、时序 **74.1%** |
| **历史保留** | 被推翻的事实留在库里并带有效区间，`as_of()` 与 `/v1/timeline` 可回放 | 「以前住哪、现在住哪」是一次查询，不是丢失 |
| **检索** | 五路信号都实现；v2 **只开语义** | 每加一路都更差：重要性 −5.3、BM25 −6.0、实体 −17.3 个点 |
| **时间衰减** | 半衰期 30 天，而语料里最新的记忆已 932 天 | 死信号：18,519 条里 0 条得分超过 0.01 |
| **上下文** | 从不填满，所以不需要压缩：抽取本身就是压缩，原文逐字留库 | 中位 **574 token**，对比 naive RAG 12,763、整段历史 109,059 |
| **细节丢失** | 答题器报告缺具体值时，原文回退取回逐字轮次 | 触发 **35.0%**，其中 **18.0%** 是回退之后才对的 |
| **降权与淘汰** | 指数衰减与容量淘汰都已实现 | **两个都关着**，从未在会增长的语料上标定 |
| **能存多久** | 没有时间上限，也没有容量上限，只有显式删除才会消失 | 按设计无界 |

**最要紧的两个缺口。** 抽取保真度 36.6%，是 14 个已分析失败里 10 个的第一丢失点——下面每一层都继承这个损失。以及**没有隔天考卷**：没有任何测试在第二天换个问法再问同一件事。`knowledge-update` 是同一次运行里、用同一种问法、只问一次。

完整版与每个数字的证据：[项目报告](docs/PROJECT_REPORT.zh-CN.md)。

### 定时运行

没人执行的备份策略不是备份，是计划。`tools/operate.py` 是给调度器调的一条命令，做的是持有真实数据的部署必须有的三件事：

```bash
python3 tools/operate.py \
  --store stores/live.db --backups /var/backups/lltm \
  --journal-copy /mnt/offsite/lltm --keep 7 --alert-at 0.8 --log /var/log/lltm-ops.jsonl
```

它**先快照再裁剪**，所以保留策略不会删掉新快照还没替代的那一份。它把删除日志复制到库自己那块盘之外——删除记录和库一起丢失，会让恢复什么都不重放，然后报告"已删除的命名空间已正确移除"。它还会在账号接近日上限、**还来得及处理的时候**点名，而不是让对方以 429 的形式发现——包括那些花了配额却一条记忆都没写成的账号，这些在命名空间列表里根本看不到。

退出码 0 正常、1 需要关注、2 没能完成；每次运行往 `--log` 追加一行 JSON，所以"昨晚的备份跑了没有"不用翻邮箱就能回答。恢复演练是另一条命令：`tools/backup_restore.py verify`。

部署它，以及部署前必须先成立的几件事：[DEPLOY.md](DEPLOY.md)。

## 局限

- 抽取有损；日期列采用会话日期，出处范围是启发式匹配。命中源会话弱于命中完整答案证据。
- 冻结终测上整段历史更准确。回答上下文减少不代表没有写入成本。
- 服务仍是原型：令牌留空即等于开放访问；进程内串行写入，SQLite 单写，向量另行保存，跨资源恢复与多进程写入尚未闭环。
- 常规 `forget` 保留行；REST 命名空间硬删除清理在线数据，并在 store 旁记录这次删除；恢复演练会重放备份之后发生的删除。定时备份、删除记录的异地副本和生产环境恢复演练仍待完成。

## 文档

| 文档 | 内容 |
|---|---|
| [架构图集](docs/PROJECT_REPORT.zh-CN.md#系统的架构) | 六张矢量图、代码定位、默认与可选分支 |
| [实验沿革](docs/PROJECT_REPORT.zh-CN.md#实验历史) | 合并后的阶段记录及结果限定 |
| [完整项目报告](docs/PROJECT_REPORT.zh-CN.md) · [可视化阅读版](docs/PROJECT_REPORT.zh-CN.html) | 当前实现、研究结果、问题与完整计划 |
| [早期报告](docs/PROJECT_REPORT.zh-CN.md) · [English](docs/PROJECT_REPORT.zh-CN.md) | 历史设计与当时测量 |
| [设计决策](docs/PROJECT_REPORT.zh-CN.md#决策记录) | 测量支持的决策 |
| [当前状态](docs/PROJECT_REPORT.zh-CN.md#现在在哪) · [路线图](docs/PROJECT_REPORT.zh-CN.md#后续完整计划) | 持续研究记录与后续计划 |
| [分层计划](docs/PROJECT_REPORT.zh-CN.md#系统的架构) | 拟议 core/research 拆分，不是当前模块布局 |
| [数据协议](results/data-protocol.md) | 各题集允许的使用方式 |

[MIT 许可](LICENSE)
