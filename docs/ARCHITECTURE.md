# Architecture atlas / 架构图集

本图集描述**工作区实现**，不表示冻结实验运行时的源码快照，也不把分层计划画成已经实现的系统。总览于 2026-09-13 按当前代码重画，提供中英文版本；其余五张细节图保留英文标签，中文图注解释边界与实现细节。

六张图使用白底、细线与可编辑文字。新总览使用横向彩色分区：蓝色接入、绿色抽取/回答、橙色存储、红色召回/回退、紫色客户端/候选。其余细节图沿用蓝色处理、绿色记忆状态、赭色原文证据。实线表示流程或引用，虚线表示条件路径；图 3 的虚线特指应用层的来源查找。盒子表示逻辑职责，不表示独立进程。

| 图 | 关注的问题 | 可编辑矢量图 |
|---|---|---|
| 1 | 系统如何连接记忆与原文？ | [中文总览](figures/overview.zh-CN.svg) · [English](figures/overview.svg) |
| 2 | 会话如何被抽取、去重、写入和更新？ | [批量写入](figures/write-path.svg) |
| 3 | 哪些数据在 SQLite，哪些在向量文件？ | [存储模型](figures/storage.svg) |
| 4 | 候选如何排序，何时回退原文？ | [读取与回退](figures/read-path.svg) |
| 5 | 旧事实晚到时如何修正时间线？ | [时间更新](figures/temporal.svg) |
| 6 | REST、MCP、批处理、演示和评测是什么关系？ | [入口与评测](figures/runtime-evaluation.svg) |

## 1. 系统总览

![系统总览：对话抽取与存储、独立问题查询、混合召回、候选记忆和条件原文回退](figures/overview.zh-CN.svg)

**图 1.** 结构化记忆与原文承担不同职责：前者用于压缩、排序和状态更新，后者用于恢复被抽取丢掉的链接、数字或措辞。原始会话不是只能写入而无法查询的日志，`turns_fts` 使它可以被回退路径搜索。

横向主图区分写入与读取：对话进入抽取和持久化，顶部问题路径绕过写入直接进入召回；存储到检索的箭头表示读取关系，不表示每次写入会自动生成回答。REST/MCP 通过单轮 adapter 复用抽取与去重组件，CLI 使用批处理入口，实际注册原文与时序消解顺序见图 2。

租户/状态过滤发生在召回阶段；`temporal=true` 只保留 active 状态，不是自动推断问题日期。候选并集后计算五路信号，默认只有语义权重为 1，其余为 0；重排可选。服务默认 K=10，冻结 v2 评测 K=20，图中明确区分。

`answer` 直接返回；`need_source` 和 `no_evidence` 都可能进入原文回退，后者不带所选记忆的来源锚点。下方回退区表示同一命名空间内的 source-local / archive-wide 选择，只有找到原文才再次调用回答器；无命中保留判定或返回未知。输出画为答案与来源/恢复摘要，不保证每次自然语言回答都有完整、正确的引用。

REST 配置令牌后获得凭据绑定的租户身份；空配置是开放模式，错误的非空配置拒绝服务。MCP stdio 是可信本地入口，不能推断其 HTTP transport 已获得等同 REST 的认证。可选研究策略与运维边界单独标注，未实现的外部 Document RAG 不画入当前架构。

代码：[IngestionPipeline](../src/llm_long_term_memory/ingest/pipeline.py)、[MemoryRunner](../src/llm_long_term_memory/evaluation/runners/memory.py)、[RawFallback](../src/llm_long_term_memory/retrieve/fallback.py)。

## 2. 批量写入

![批量写入：Stage A、Stage B、规则与出处，然后归档、去重、持久化、时序消解和检查点](figures/write-path.svg)

**图 2.** `IngestionPipeline.run()` 先检查抽取指纹，再按命名空间分批，跳过检查点中已完成或已被内容策略拒绝的会话。`_ingest_batch()` 的主路径是：

1. `TwoStageExtractor` 调用 Stage A 提取事实、说话人、主体和 scope；有事实时再调用 Stage B 分配 key、object、update operation。
2. 规则补充类型、实体、importance 等属性。`event_time` 和 `valid_from` 来自**会话日期**。内容里的事件日期可能被保留为文字，但当前代码没有通用的事件日期解析器把它写入这两列。
3. `source_span_for()` 根据句子词项重合和数字加权选择出处。找不到锚点时可以为空，因此不能声称每条记忆都有经过验证的精确证据。
4. 抽取返回后，先注册原始会话和轮次，再将 memory 的 `source_session_id` 转为带命名空间的内部 ID。
5. 对候选做嵌入，找到近邻并调用 LLM 判重。`0.92` 是近邻候选阈值；**不是达到阈值就直接丢弃**。同键事实也可能进入判重，只有 `DUPLICATE` 才丢弃新候选。
6. 将保留的 memory 写入 SQLite，并追加向量；之后 `TemporalResolver` 重建本批触及的 key。
7. 按配置间隔保存向量文件和检查点，在运行结束或配额停止时也保存。它们不是与 SQLite 共享的原子事务。

因此，一批非空事实通常需要 **Stage A + Stage B 两次抽取请求，另加可能发生的去重裁决请求**。Stage A 无事实时跳过 Stage B。内容策略拒绝的批次仍归档原文并记录拒绝状态；其他失败批次保留为待处理。

代码：[two_stage.py](../src/llm_long_term_memory/ingest/two_stage.py)、[extract_facts.py](../src/llm_long_term_memory/ingest/extract_facts.py)、[keying.py](../src/llm_long_term_memory/ingest/keying.py)、[structure.py](../src/llm_long_term_memory/ingest/structure.py)、[provenance.py](../src/llm_long_term_memory/ingest/provenance.py)、[dedup.py](../src/llm_long_term_memory/ingest/dedup.py)、[fingerprint.py](../src/llm_long_term_memory/ingest/fingerprint.py)。

对应测试：[pipeline](../tests/test_pipeline.py)、[two-stage](../tests/test_two_stage.py)、[ingest](../tests/test_ingest.py)、[fingerprint](../tests/test_fingerprint.py)。

## 3. 存储模型

![存储模型：SQLite 会话、轮次、记忆与 FTS5 索引，以及独立保存的 NumPy 向量和 ID 映射](figures/storage.svg)

**图 3.** 正常服务与 CLI 使用同一命名约定：

```text
stores/<store>.db
stores/<store>.db-wal             # SQLite 运行时可能存在
stores/<store>.db-shm             # SQLite 运行时可能存在
stores/<store>-index.npy
stores/<store>-index.ids.json
```

SQLite 保存源数据、状态、词法索引以及辅助关系。向量索引是 `N × 384` 的归一化 `float32` 矩阵；`.ids.json` 按行记录 memory ID。这是应用维护的对应关系，不是 SQLite 外键，也不是外部向量数据库。

`sessions.id → turns.session_id` 和 `memories.source_session_id → sessions.id` 有外键约束。`source_turn_index` 与字符偏移没有直接的外键约束；`superseded_by` 是 memory 对自身的引用。图中 `memories` 的字段按职责分组，完整字段以 [schema.sql](../src/llm_long_term_memory/store/schema.sql) 为准。

辅助表的用途：

| 表 | 用途 |
|---|---|
| `entities`, `memory_entities` | 命名空间内的实体归一化及 memory–entity 多对多关系 |
| `evidence` | 合并后记忆与其源 memory 的多对多关系；不是原文轮次表 |
| `meta` | 抽取版本、指纹和迁移标记等元数据 |
| `memories_fts`, `turns_fts` | 由触发器维护的内容索引；状态/命名空间过滤由查询完成 |

常规 supersession 和 `forget` 保留 memory 行并更新状态；REST `DELETE /v1/data`、会话迁移和演示命名空间清理会物理删除在线数据，备份生命周期另行处理。原始轮次的 namespace 检查通过 `sessions` 联表完成，语义检索则在向量扫描后过滤 memory。底层 `get(id)` 不自带用户鉴权，服务入口补做所属用户检查。

代码：[SQLiteMemoryStore](../src/llm_long_term_memory/store/sqlite.py)、[NumpyFlatIndex](../src/llm_long_term_memory/store/vector.py)、[session_keys.py](../src/llm_long_term_memory/store/session_keys.py)、[session_migration.py](../src/llm_long_term_memory/ingest/session_migration.py)。

## 4. 读取与条件回退

![读取路径：语义与词法各取候选，合并评分并组装上下文，三种判定分别进入答案或原文恢复](figures/read-path.svg)

**图 4.** `HybridRetriever.retrieve_with_trace()` 在整个向量索引上做精确相似度检索，再过滤用户和状态，取最多 `candidate_limit` 条语义候选；词法路径在 SQL 中过滤后也取最多 `candidate_limit` 条。二者按 memory ID 取并集，**不是合并后再截成 50 条**。默认每路 50，合并后至多 100。

五路信号都被计算。默认 `semantic=1.0`，其余四路权重为零；BM25 仍参与候选召回。语义信号映射为 `(cosine + 1) / 2`，BM25 在当前候选中归一化，时间、importance、实体信号再进入加权公式。

服务默认 `service.top_k=10`，v2 评测配置的 `retrieval.top_k=20`。`temporal=True` 表示排除 superseded 行；不是自动理解问题时间并执行 `as_of` 查询。历史问题仍可能需要专门的时间线访问或原文恢复。

启用 fallback 后，第一次回答是一个结构化 verdict：

| 状态 | 当前实际行为 |
|---|---|
| `answer` | 直接返回答案，无原文搜索、无第二次回答 |
| `need_source` | 同时考虑 selected memories 的来源和 BM25 原文排名；选择 source-local 或 archive-wide |
| `no_evidence` | 仍然可以搜索原文，但不传入 selected memory 锚点，即 archive-wide |

`RawFallback.recover()` **先搜索并排名原文，再截断**。有本地出处，且最佳 BM25 命中的会话属于这些本地出处时，优先返回本地轮次；没有 BM25 命中但有本地轮次时，也使用本地轮次。否则选 archive-wide 命中。只有实际找到原文才调用第二次回答器；没找到时保留 verdict 中的文本或返回不知道。

默认恢复最多 3 个轮次、2,400 个原文字符；来源标签等格式开销不计入这个字符上限。v2 第二次回答主要接收恢复的原文；`reasoned_v3` 第二次回答还带上第一轮结构化上下文。结构化输出解析异常走防泄漏保护；无法恢复可读答案时不会把原始结构作为回复展示。

`/v1/memories/search` 的 `explain` 返回候选信号和部分被拒原因；`/v1/answer` 返回答案及选择/回退摘要。它们不是同一个响应结构，不应在图中将所有输出都画在 answer 响应上。

代码：[hybrid.py](../src/llm_long_term_memory/retrieve/hybrid.py)、[fallback.py](../src/llm_long_term_memory/retrieve/fallback.py)、[MemoryRunner](../src/llm_long_term_memory/evaluation/runners/memory.py)、[API models](../src/llm_long_term_memory/api/models.py)。

对应测试：[hybrid retrieval](../tests/test_hybrid_retrieval.py)、[raw fallback](../tests/test_raw_fallback.py)、[memory runner](../tests/test_memory_runner.py)、[API](../tests/test_api.py)。

## 5. 时间更新

![时间更新示例：事实按八月、一月、三月顺序到达，仍重建为一月到三月、三月到八月、八月起的有效区间](figures/temporal.svg)

**图 5.** 同一 `(user_id, subject, predicate)` 上，后到达的旧事实不会直接成为当前值。resolver 读取包括 superseded 在内的全链，按 `(event_time, id)` 排序，重算状态和有效期；因此三月事实晚到时可以把一月事实的 `valid_to` 从八月改为三月。

这个示例显式假设每个新值都是 replacement。实际代码不会因为“同 key 有一个 replacement”就把所有事实排成互斥链：是否关闭前一条由**下一条事实**的 `replaces_previous` 决定。`coexists` 允许并存；`removes` 记录为结束操作并设置关闭标记。重复值可能折叠到最早的区间 owner，新增数字会阻止部分有损折叠。无日期事实被跳过，evicted 行不参与重建。

`event_time`/有效区间与 `ingested_at` 是两种时间信息，但当前实现没有保存每次数据库状态变更的完整事务时间历史。文档宜写“有效期 + 写入时间”，不宜让“bi-temporal”暗示已经具备任意事务时间快照查询。

代码：[TemporalResolver 和 as_of](../src/llm_long_term_memory/temporal/resolve.py)、[时序测试](../tests/test_temporal.py)。

## 6. 入口与研究评测

![入口与评测：批处理、REST/MCP、无需 LLM 的演示写入，以及冻结、运行、评分和聚合工作流](figures/runtime-evaluation.svg)

**图 6.** 仓库包含三种运行入口，不能把它们简单画成一条统一写入流水线：

| 入口 | 真实连接 | 写入/回答边界 |
|---|---|---|
| `lltm ingest run` | 配置 → Extractor/TwoStageExtractor → IngestionPipeline | 完整批量抽取、去重、持久化、可选时序消解与检查点 |
| REST、Inspector、MCP | `MemoryService` | 共享搜索、读取、时间线、软遗忘和单轮抽取写入；REST 回答复用 MemoryRunner |
| `demo-api/app.py` | `Playground` | 接收显式结构化事实，直接写入、嵌入和 resolve；原始 turns 单独写入；没有 LLM 抽取或自然语言回答 |

`MemoryService.add_message()` 在配置 API key 后按需创建 `LiveTurnExtractor`，适配真实批量抽取器，并执行去重、出处关联、索引保存和可选时序消解；测试仍可注入 adapter。服务使用进程内 `RLock` 串行化共享资源访问，回答的 `limit` 作为参数传递。`/healthz` 在嵌入器不可用时返回 503，`/livez` 单独报告进程存活。上述修复于 2026-09-12 从未合入的审计分支接入当前工作区。

REST 另有凭据身份边界、数据导出与命名空间硬删除，不能由“共享 MemoryService”推断这些 HTTP 功能已经暴露为 MCP 工具。参见 [identity.py](../src/llm_long_term_memory/api/identity.py) 和 [app.py](../src/llm_long_term_memory/api/app.py)。

服务目前从 `evaluation/` 导入 `MemoryRunner`、`Instance` 和 prompt 版本，批处理也使用 LongMemEval 的类型。因此 [架构分层计划](ARCHITECTURE_SEPARATION_PLAN.md) 中独立的 core/research 边界仍是目标，不是现状。

评测由 `Runner.prepare()/answer()` 与 `Judge.grade()` 协作。full-context 基线使用完整会话；naive RAG 以**会话为块**取 top-5；MemoryRunner 复用已抽取的 store。普通回答不使用 gold labels；显式 oracle arm 是上限诊断。冻结、运行锁、检查点、usage 和 aggregate 由不同模块及脚本协作，不是独立在线服务。

### 可选分支的真实状态

| 模块 | 接入点与默认状态 |
|---|---|
| `retrieve/rerank.py` | 配置启用时对预选候选 cross-encode；默认关闭 |
| `retrieve/coherent.py` | 只有 `two_stage_coherent` / oracle 变体传入 session budget；v2 最终选择 flat20 |
| `retrieve/hydrate.py` | 根据出处恢复句子；无条件 hydration 与 v3 条件 hydration 是不同选项 |
| `runners/reasoning.py` | `reasoned_v3` 通过问题措辞分类；时间/聚合/当前状态可触发 v3 hydration |
| `runners/synthesis.py` | `synthesis_v4` 的模型输出操作数，Python 计算 count/duration；comparison 保留模型措辞 |
| `retrieve/relation_router.py`, `scan.py` | v4.1 scan 变体，按问题路由后追加关系匹配事实；需要外部 predicate map，默认不注入 |
| `lifecycle.py` | 可选衰减、访问强化、容量淘汰；默认关闭 |
| `consolidate/runner.py` | CLI 显式触发聚合，保留源 memory 证据；默认路径不执行 |
| `influence/`, `pack/` | 离线消融产生效用数据，拟合预测器，再供可选预算打包；默认关闭 |

注意当前 wiring 的一个区别：v4 变体虽然传了 `adaptive_reasoning_hydration=True`，但 `MemoryRunner.answer()` 只在 `answer_policy == 'reasoned_v3'` 时计算 `reasoning_kind`，所以不能据这个 flag 就在图中声称 v4 也执行了 v3 的条件 hydration。这里仅记录现状，没有修改研究代码。

代码：[cli.py](../src/llm_long_term_memory/cli.py)、[service.py](../src/llm_long_term_memory/api/service.py)、[mcp_server.py](../src/llm_long_term_memory/mcp_server.py)、[demo-api](../demo-api/app.py)、[harness.py](../src/llm_long_term_memory/evaluation/harness.py)、[reproducibility.py](../src/llm_long_term_memory/evaluation/reproducibility.py)、[validation.py](../src/llm_long_term_memory/evaluation/validation.py)。

## 重建与论文使用

```bash
python3 docs/figures/generate.py         # 六张图 + 中文总览，共七份 SVG；仅需 Python 标准库
python3 docs/figures/generate.py --pdf   # 加导出六页英文矢量 PDF，需要 reportlab
```

源文件：[generate.py](figures/generate.py)、[双语总览布局](figures/overview.py)。PDF 输出到 `output/pdf/architecture-atlas.pdf`；横向总览页使用 2600×1040 画布，其余页保留原尺寸。SVG 保留可选中的文字、形状和路径；图标由路径绘制，不依赖模型品牌标志或外部图片。PDF 同样使用矢量图形，不是截图。论文排版建议每次使用一张图并配对应图注；主文使用总览与回退图，存储细节和实验流程可以放附录。长图适合通栏或单独横页，避免缩小到单栏后文字难读。

此生成器只处理文档，不导入项目运行代码，不读数据库、题目或密钥，也不发出模型请求。未向受冻结的 `scripts/`、`src/`、配置或证据目录增加文件。
