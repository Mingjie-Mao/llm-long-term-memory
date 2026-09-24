# 仓库瘦身与代码结构审计

审计日期：2026-09-24（Australia/Sydney）

本报告统计当前工作树中“Git 已跟踪 + 未被 `.gitignore` 排除的未跟踪文件”。因此它包含本轮尚未
提交的 v2b/v2c/v2d 代码和实验结果，但不把 `.git/`、`.venv/`、`data/`、`stores/`、缓存和本地
密钥算作仓库源码。统计命令：

```bash
uv run python scripts/repo_stats.py
```

本轮没有移动核心代码，没有删除任何实验结果，也没有改 benchmark 数字。`results/frozen/`、
`results/sealed/` 和历史归档保持只读。

## 一、结论先行

- 仓库约 50.5 万行，主要因为 **JSON/JSONL 实验产物按文本行被计入**。
- `results/` 占约 41.8 万行，约为仓库物理行的 83%；其中约 95.6% 是机器生成或冻结产物。
- `src/` 一共 23,021 Python 行，不是几十万行。
- 按文件职责进一步拆分，明确的产品 runtime 约 **11,185 行**；另有 6,729 行 evaluation、
  883 行 influence 研究代码、1,591 行摄取诊断/迁移代码，以及 841 行职责混合的 CLI。
- 真正的问题不是“50 万行业务代码”，而是**研究证据、一次性脚本、benchmark 代码和产品代码在
  同一搜索面内**，导致 GitHub 语言统计、Codex 搜索和新开发者阅读成本偏高。

## 二、仓库统计

下表的 generated 比例按行数计算。这里只把能明确识别的机器产物标为 generated；手写的实验协议、
结论 Markdown 和配置不会为了让数字好看而被标记。

| 范围 | 文件数 | 大小 | 物理行 | Python 行 | 数据/文档行 | generated 行占比 | 主要类别 |
|---|---:|---:|---:|---:|---:|---:|---|
| `src/` | 114 | 967.9 KiB | 23,925 | 23,021 | 651 | 0% | A + C |
| `tests/` | 135 | 849.1 KiB | 23,992 | 23,965 | 27 | 0% | B |
| `scripts/` | 30 | 316.6 KiB | 8,039 | 8,039 | 0 | 0% | C + D |
| `tools/` | 56 | 585.6 KiB | 14,119 | 14,119 | 0 | 0% | C + D |
| `results/` | 521 | 32.3 MiB | 418,435 | 0 | 418,435 | 95.6% | E + F + 手写协议 |
| `docs/` | 27 | 390.1 KiB | 4,598 | 1,653 | 2,945 | 60.8% | G + 生成图 |
| `configs/` | 16 | 37.5 KiB | 1,254 | 0 | 1,254 | 0% | A + C |
| `public-demo/` | 14 | 84.6 KiB | 2,027 | 517 | 1,510 | 0% | H |
| `demo-api/` | 9 | 144.5 KiB | 4,310 | 1,513 | 2,781 | 0% | H |
| 根目录 | 13 | 571.2 KiB | 3,797 | 0 | 704 | 73.7% | G + H + lock/config |
| 其他（CI、skills、research 索引等） | 13 | 47.5 KiB | 774 | 0 | 774 | 0% | G + 开发配置 |

即时总计为 **948 个文件、36.2 MiB、505,270 物理行**。文件继续变化时，应以
`scripts/repo_stats.py` 的即时输出为准。

### `src/` 的职责快照（2026-09-19）

以下职责拆分是上一轮逐文件分类的快照，不冒充 2026-09-24 的实时逐文件统计；实时宽口径以
上表的 23,021 Python 行为准。新增文件应在下一次职责重分类时归入对应类别。

| 组成 | Python 行 | 判断 |
|---|---:|---|
| 产品 runtime（API、store、retrieve、LLM、temporal、核心 ingest、低耦合 CLI 等） | 11,185 | A |
| `evaluation/` 及其 CLI | 6,729 | C，不应称为业务 runtime |
| `influence/` 及其 CLI | 883 | C，研究/效用评估 |
| ingest 诊断、覆盖、迁移和内置 benchmark | 1,591 | C，部分已被 package 化 |
| `cli.py` | 841 | A/C 混合，无法在文件级准确拆分 |
| **合计** | **21,229** | 全部 shipped Python，不等于核心业务代码 |

因此，对“真正 src 核心程序占多少”的可审计回答是：

- 当前宽口径（整个 `src/`）：23,021 行，占当前仓库物理行约 4.6%；
- 2026-09-19 严格文件职责口径：11,185 行；
- 2026-09-19 非 evaluation 上限：14,500 行，但这里仍含研究诊断和混合 CLI，不能全算 core。

### `results/` 的组成

| 子目录 | 文件数 | 大小 | 行数 | 处理意见 |
|---|---:|---:|---:|---|
| `results/raw/` | 139 | 12.4 MiB | 162,324 | 生成证据；按实验 ID 局部读取 |
| `results/analysis/` | 83 | 4.0 MiB | 132,991 | JSON 多为生成物，Markdown 是摘要入口 |
| `results/sealed/` | 34 | 1.2 MiB | 55,621 | E，保留且只读 |
| `results/frozen/` | 85 | 9.6 MiB | 42,495 | E，保留且只读 |
| `results/archive/` | 56 | 4.4 MiB | 6,801 | E/D 的归档证据，保留 |
| 其余 audit/review/validation/manifests/协议 | 124 | 0.6 MiB | 18,203 | E、C 与手写协议混合 |

## 三、最大文件

### 按体积最大的 20 个文件

| 文件 | 大小 | 行数 |
|---|---:|---:|
| `results/analysis/train150-zero-yield.final.json` | 985.0 KiB | 35,986 |
| `results/raw/extraction-batch-size.extractions.json` | 715.3 KiB | 21,226 |
| `results/analysis/train150-zero-yield.partial.json` | 654.4 KiB | 23,982 |
| `results/raw/two_stage_hydrated.heldout100.jsonl` | 645.7 KiB | 100 |
| `results/raw/two_stage_hydrated.heldout100-rep2.jsonl` | 644.1 KiB | 100 |
| `results/raw/two_stage_hydrated.heldout100-rep3.jsonl` | 643.9 KiB | 100 |
| `results/frozen/v4.2-development-20260912e/live/rows.jsonl` | 640.4 KiB | 404 |
| `results/frozen/v4.2-development-20260912/rehearsal/rows.jsonl` | 603.4 KiB | 404 |
| `results/frozen/v4.2-development-20260912b/rehearsal/rows.jsonl` | 603.4 KiB | 404 |
| `results/frozen/v4.2-development-20260912c/rehearsal/rows.jsonl` | 603.4 KiB | 404 |
| `results/frozen/v4.2-development-20260912d/rehearsal/rows.jsonl` | 603.4 KiB | 404 |
| `results/frozen/v4.2-development-20260912e/rehearsal/rows.jsonl` | 603.4 KiB | 404 |
| `results/raw/train150.ingest.usage.json` | 542.4 KiB | 23,349 |
| `uv.lock` | 524.7 KiB | 2,800 |
| `results/frozen/v4.2-development-20260912e/source.tar.gz` | 521.8 KiB | 0 |
| `results/frozen/v4.2-development-20260912d/source.tar.gz` | 521.5 KiB | 0 |
| `results/frozen/v4.2-development-20260912c/source.tar.gz` | 521.5 KiB | 0 |
| `results/frozen/v4.2-development-20260912b/source.tar.gz` | 521.1 KiB | 0 |
| `results/frozen/v4.2-development-20260912/source.tar.gz` | 518.5 KiB | 0 |
| `results/frozen/v4.2-development/source.tar.gz` | 518.3 KiB | 0 |

### 按行数最大的 20 个文件

前 20 名全部是实验 JSON/CSV，而不是 Python：

| 文件 | 行数 |
|---|---:|
| `results/analysis/train150-zero-yield.final.json` | 35,986 |
| `results/analysis/train150-zero-yield.partial.json` | 23,982 |
| `results/raw/train150.ingest.usage.json` | 23,349 |
| `results/raw/extraction-batch-size.extractions.json` | 21,226 |
| `results/raw/dev100.ingest.usage.json` | 12,837 |
| `results/raw/heldout100.ingest.usage.json` | 12,657 |
| `results/raw/test100.ingest.usage.json` | 12,495 |
| `results/raw/batch-position-pilot.json` | 8,008 |
| `results/raw/two-stage-hydrated.ingest.usage.json` | 5,511 |
| `results/raw/two-stage-p10.ingest.usage.json` | 5,439 |
| `results/analysis/predicate-map.csv` | 5,055 |
| `results/audit/workspace-review-20260912.json` | 4,666 |
| `results/raw/v2c-reasoning48.ingest.usage.json` | 4,287 |
| `results/raw/influence-chronomem.usage.json` | 3,865 |
| `results/frozen/v4.2-development/freeze.json` | 3,817 |
| 五个 `v4.2-development-20260912* /freeze.json` | 各 3,807 |

### 最大 Python 文件

`scripts/` + `tools/`：

| 文件 | 行数 |
|---|---:|
| `scripts/a2_preflight.py` | 659 |
| `tools/run_synthesis_probes.py` | 645 |
| `scripts/run_v3_dev60.py` | 562 |
| `tools/backup_restore.py` | 560 |
| `tools/run_v42_comparison.py` | 541 |
| `tools/entity_count_probes.py` | 540 |
| `tools/count_review.py` | 486 |
| `tools/synthesis_probes.py` | 471 |
| `scripts/run_v3_phase5_tune3.py` | 426 |
| `scripts/finalize_train150.py` | 407 |

`src/`：

| 文件 | 行数 | 主要风险 |
|---|---:|---|
| `src/llm_long_term_memory/api/service.py` | 1,258 | 服务、回答、评测适配混合 |
| `src/llm_long_term_memory/store/sqlite.py` | 1,000 | 多存储职责集中 |
| `src/llm_long_term_memory/evaluation/runners/memory.py` | 991 | 多代候选策略共存 |
| `src/llm_long_term_memory/cli.py` | 841 | 主装配、runner 构造、ingest run/fidelity 仍混合 |
| `src/llm_long_term_memory/evaluation/validation.py` | 704 | 研究验证集中，但边界合理 |
| `src/llm_long_term_memory/evaluation/reproducibility.py` | 528 | 冻结/哈希职责 |
| `src/llm_long_term_memory/api/app.py` | 505 | HTTP 路由较集中 |
| `src/llm_long_term_memory/ingest/pipeline.py` | 467 | core 与 benchmark DTO 耦合 |
| `src/llm_long_term_memory/evaluation/runners/synthesis.py` | 466 | 实验策略 |
| `src/llm_long_term_memory/evaluation/runners/v2d.py` | 442 | 开发候选策略 |

## 四、文件分类

### A. Core runtime

保留：`api/`、`store/`、`retrieve/`、`llm/`、`temporal/`、`consolidate/`、`runtime/`、
`pack/`、`embed/`、配置、锁、生命周期、MCP，以及 ingest 的抽取/去重/时间/指纹主路径。

风险：这些目录不是完全独立。`api/service.py` 直接导入 evaluation 的 judge、answer prompt、
`MemoryRunner` 和 LongMemEval `Instance`；多个 ingest 模块也直接使用 `HaystackSession`。所以
11,185 行是“按职责归类的 core”，不是已经完成解耦的 deploy-only 包。

### B. Tests

`tests/` 共 115 文件、21,274 行。测试规模与 shipped source 接近，原因是项目有大量冻结、跨平台
编码、配额、租户隔离、恢复和实验协议不变量。不能因为行数与 `src/` 接近就把它当作 code bloat。

### C. Active research tooling

暂定 38 个通用或仍有明确入口的脚本，约 9,526 行，包括：

- `scripts/{aggregate_validation,audit_zero_yield,check_ingest_state,finalize_train150,freeze,`
  `migrate_scoped_sessions,record_golden,repo_stats,run_final_test,run_validation,session_recall}.py`；
- `tools/` 下的 gold/probe 审计、backup/restore、count review、failure taxonomy、operate、
  retrieval replay、probe safety、hidden-set registration、trial simulation 等通用工具。

其中 `backup_restore.py`、`operate.py`、`synthesis_probes.py` 等已被测试或其他工具引用；
`repo_stats.py` 是本轮新增的稳定统计入口。

### D. Historical research tooling

暂定 38 个版本绑定或一次性脚本，约 10,321 行。证据是文件名、固定结果路径、冻结 verifier、
对应历史测试和只服务某次实验的 manifest/runner：

- `scripts/`：`a2_preflight.py`、`batch_position_*`、`context_arms.py`、
  `counterfactual_qa.py`、`fact_lineage.py`、`freeze_v2.py`、`heldout_*`、`ku_oracle.py`、
  `project_v3_phase5_context.py`、`run_frozen_ingest.py`、全部 `run_v3_*`、`stage_oracle.py`、
  `targeted_batch1.py`；
- `tools/`：v2b/v2c/v2d analyzer/gate、v3 manifest builder、v4.2 comparison/protocol、
  `diagnose_v4_probes.py` 和全部 `verify_v2*` / `verify_v3*`。

“历史”不等于“可删除”：verifier 和 runner 是冻结结论的复现链，很多还有专门测试。应先建立归档
索引，再决定是否移动；本轮不移动。实验的状态、证据入口和重跑规则已集中在
[`research/EXPERIMENT_INDEX.md`](../research/EXPERIMENT_INDEX.md)。

### E. Frozen evidence

必须保留：`results/frozen/`、`results/sealed/`、`results/archive/`、正式 manifest、freeze/conclusion、
支撑 README/项目报告/EVALUATION 数字的 raw rows 与 usage。冻结目录即使内容重复，也不能按普通
generated 文件去重。

### F. Generated/reproducible artifact

主要是 `results/raw/*.usage.json`、多数 `results/analysis/*.json`、JSONL 行、CSV、SVG、Numpy
features、`docs/figures/*.svg`、badge SVG、`uv.lock` 和 demo 的锁定 requirements。它们可以生成，
但“可生成”不自动等于“可删除”：外部 API 结果重跑有配额和非确定性，发布数字还依赖旧产物。

### G. Documentation

`README.md`、`docs/PROJECT_REPORT.md`、`ARCHITECTURE.md`、`EVALUATION.md`、本审计、部署说明、
schemas 及手写实验协议。README 和项目报告现在只保留中文 canonical 文件。

### H. Demo/deployment

`public-demo/`、`demo-api/`、`render.yaml`、Docker 文件和相关 CI。它们约 6,316 物理行，其中
demo 锁文件/参考 embedding 属于生成数据，服务与检查脚本属于代码。

### I. Dead / duplicated / superseded candidate（仅候选）

没有发现可以在无人工判断下直接称为 dead 的已跟踪 Python。以下只进入 review 清单：

- `train150-zero-yield.partial.json` 与 final 并存，看起来被 final 取代，但可能记录中断恢复证据；
- v2b/v2c 的多轮 gate analyzer 与结果已结束，但仍是负结果和决策证据；
- 多个 v3 phase runner 结构高度相似，但各自冻结 protocol/路径不同；
- `docs/badges/` 的中英文切换 badge 与生成器在中文单版本后已无页面引用；
- 若干无 repo 内调用者的 probe 脚本可能仍是手动 CLI，不能用“未 import”作为删除证据。

## 五、`scripts/` 与 `tools/` 审计

### 职责重叠

当前命名边界不稳定：`scripts/` 既有正式冻结 runner，也有分析器；`tools/` 既有运维工具，也有一次性
实验 runner。以下族存在明显重叠：

1. evaluation orchestration：`run_validation.py`、`run_final_test.py`、`run_v3_*.py`、
   `run_v42_comparison.py`；
2. failure analysis：`failure_taxonomy.py`、`diagnose_v4_probes.py`、`analyze_v2*.py`、
   `*_offline_gate.py`；
3. result/protocol validation：`aggregate_validation.py`、`check_arm_invariant.py`、
   `check_gate_is_resolvable.py`、`verify_v*.py`；
4. benchmark processing：manifest builders、holdout splitter、gold correction/review、probe builders；
5. release validation：`public-demo/check_release_figures.py`、`check_site.py` 与 frozen verifiers
   各有不同权威来源，不能简单合并。

### 应 package 化的稳定能力

- JSONL/usage 加载、paired comparison、McNemar、artifact fingerprint 和 report rendering 仍在脚本间
  重复；应逐步收敛到 `evaluation/` 的稳定模块。
- ingest 状态检查、freeze inventory、manifest 验证已经接近稳定 API，应由薄 CLI 调用，而不是每个
  phase runner 复制 orchestration。
- `backup_restore.py` 和 `operate.py` 服务产品运维，长期应进入 package/CLI，而不是 research tools。

### 应保留为 research utility

probe 生成、oracle/stage decomposition、retrieval replay、failure taxonomy 和离线 gate 分析。它们
依赖 benchmark gold 或实验路径，不应进入产品 runtime。

### 无调用者不等于无用途

静态搜索发现多个脚本无 import/CI/docs 调用，例如 `heldout_report.py`、`dedup_namespace_probe.py`、
`exhaustive_scan_probe.py`、`predicate_vocabulary.py`。这些都有可执行入口或生成历史 artifact 的语义，
只能标记为“需要所有者确认”，不能直接删除。

## 六、`src/` 结构审计

### 是否真的过度膨胀

总量不算异常：2.1 万 Python 行包含产品、REST/MCP、完整评测框架和研究候选。但边界有五个明显问题：

1. **CLI 已明显收敛，但仍混域。** 本轮已把 doctor、data、eval、lifecycle、influence、root service、
   ingest coverage 与 temporal gate 拆到 `commands/`，并通过显式注入避免评测/影响力命令反向依赖
   入口；`cli.py` 从 1,968 行降到 344 行，只保留 Typer 装配与 `_build` 这一个共享构造器。
2. **产品服务依赖 evaluation。** `api/service.py` 直接依赖 judge、answer prompt、MemoryRunner 和
   LongMemEval DTO。产品回答能力应依赖通用 answer request/context 类型，benchmark adapter 留在
   evaluation。
3. **ingest 依赖 benchmark DTO。** `pipeline.py`、`extract.py`、`two_stage.py` 等直接接收
   `HaystackSession`/`Instance`。应定义 package-owned conversation/session DTO，并由 LongMemEval
   adapter 转换。
4. **多代 answer policy 共处单 runner。** `evaluation/runners/memory.py` 991 行同时处理 v2/v3/v4、
   v2c/v2d、fallback、hydration、notes。它属于 research，不影响核心 runtime 体积，但增加实验间
   回归面。
5. **研究诊断被 shipped。** `evaluation/`、`influence/`、zero-yield/fidelity/probe 支持代码都进入
   wheel。当前体积不危险，但使“安装产品”和“复现实验”边界不清。

`store/sqlite.py`（1,000 行）和 `api/service.py`（1,258 行）也较大，但尚没有证据表明拆分能减少
缺陷；应先按职责和测试边界拆，不为减少行数而合并或机械拆文件。

## 七、建议的目标结构

不建议现在大规模移动 frozen 路径。更安全的渐进结构是：

```text
src/llm_long_term_memory/       # 产品 runtime + 稳定公共能力
src/llm_long_term_memory/evaluation/  # 先保留，逐步解除产品反向依赖
tests/
research/
  active/                       # 当前 probe / analysis / experiment runner
  archived/                     # 只读历史 runner，带索引和对应 artifact
results/
  frozen/                       # 路径不动
  sealed/                       # 路径不动
  archive/                      # 路径不动
  raw/ analysis/ manifests/     # 先加索引，不机械迁移
docs/
configs/
```

迁移前必须建立 path-reference 清单并更新测试、CI、README、EVALUATION、freeze manifest；任何被
哈希记录的历史源码或 verifier 都不能移动。第一步应只是新建索引/约定，让**未来**实验进入清晰位置。

## 八、Codex 与 GitHub 减噪

已在 `AGENTS.md` 增加默认搜索范围：普通开发先读 `src/` 和对应 `tests/`，只有 benchmark、failure、
reproduction、release audit 或用户明确要求时才默认进入 `results/`；大型 artifact 必须先按 ID/key
定位，先读摘要再读 raw。

`.gitattributes` 只把真正的 JSON/JSONL/CSV/SVG/NPY、frozen source bundle 和生成图标为
`linguist-generated=true`。Python、手写 Markdown、协议和配置没有被隐藏。

### 发布一致性审计

- README 与项目报告的公开 benchmark 数字未改，仍可追溯到 EVALUATION 和 frozen/sealed 证据；
- `public-demo/check_site.py` 通过，部署文件、labels、翻译字段和 release manifest 结构一致；
- `public-demo/check_release_figures.py` 一度**未通过**：manifest 记录 1,287 个测试，
  measured tree SHA 也已变化。这是发布准备阻塞项，不是 benchmark 结果错误；
- 阻塞的真实原因是当时还有新增 source/test 文件未被 Git 跟踪。`measured_tree_sha256` 由
  `git ls-files` 计算，测试数却由工作树实跑得出，所以在纳入版本控制之前 `--update` 会写出一条
  别的 checkout 无法复现的声明；
- **已解决**：新增代码、测试、工具、配置与 `results/` 证据已全部提交，随后运行
  `public-demo/check_release_figures.py --update`，measured tree SHA 与
  `source_commit_for_reference` 同批刷新；
- 刷新时暴露出第二个问题：测试数从 1,318 跳到 1,356，多出的 38 项全部来自
  `tests/test_markdown_renders.py`——它按 `git ls-files` 参数化，本轮提交的 prereg 与 analysis
  Markdown 现在都被逐篇渲染检查。也就是说这个对外数字同时随「代码质量」和「提交了几篇文档」
  变动，再提交一批预注册它就会涨，而引擎什么也没被多测一次；
- **已修正**：release manifest 升到 schema 2，`tests` 只计代码测试，文档渲染检查单列为
  `document_checks`。2026-09-24 实测 **1,583 项工程测试 · 152 篇文档核查 · 86% 行覆盖率**。
  schema 版本一并提升，因为 `tests` 的含义变了——拿着旧 manifest 的消费者不应把两个数直接相比。
  被排除的是具名的那一个参数化测试，改名会让
  `tests/test_release_manifest.py` 当场失败，而不是悄悄把文档用例算回工程测试。

## 九、清理清单（本轮不执行）

### SAFE TO CLEAN

这些都已被忽略且可重建；删除只影响本机缓存，不影响 Git 历史：

| 路径 | 当前大小 | 理由 | 风险 |
|---|---:|---|---|
| `.pytest_cache/` | 152 KiB | pytest 缓存 | 下次测试稍慢 |
| `.ruff_cache/` | 340 KiB | Ruff 缓存 | 下次 lint 稍慢 |
| `.coverage` | 76 KiB | 可重新跑 coverage | 丢失当前本地覆盖快照 |
| 根目录与 `stores/` 的 `.DS_Store` | 约 12 KiB | macOS 元数据，已 ignore | 无产品风险 |

`.venv/`（约 1.0 GiB）也可由 lock 重建，但它是当前开发环境，不建议为“仓库瘦身”而删。

### REVIEW BEFORE CLEAN

| 候选 | 大小/规模 | 理由 | 风险 |
|---|---:|---|---|
| `docs/badges/` | 约 5 KiB | 中文单版本后已无页面引用 | 生成器和历史设计说明会一起失效 |
| `results/analysis/train150-zero-yield.partial.json` | 654 KiB | 有 final 同类文件 | 可能丢失中断/恢复证据 |
| 38 个历史 research scripts/tools | 10,321 行 | 日常开发不再需要 | 可能破坏 frozen 复现和 verifier 测试 |
| v2b/v2c/v2d 未跟踪 raw/analysis | 多文件 | 当前阶段已结束 | 是本轮负结果与决策证据，应先冻结/索引 |
| 本地 `stores/` | 2.9 GiB | Git 已忽略，可重建 | API 重跑昂贵；WAL 不能单独删除 |
| 本地 `data/` | 354 MiB | Git 已忽略，可重下 | 下载与许可/网络成本 |

### KEEP

- 全部 core runtime、tests、配置和 CI；
- `results/frozen/`、`results/sealed/`、`results/archive/`；
- 支撑 README、PROJECT_REPORT、EVALUATION 和 public demo 数字的 raw/usage/manifest；
- 仍有测试或运行入口的 research tooling；
- `docs/figures/`（README/报告直接引用）；
- `uv.lock`、demo requirements、reference embeddings（可复现部署所需）；
- 当前 v2d gate16 的失败结果与 `STOP_NO_48` 分析。

## 十、建议执行顺序

### P0：立即保持

- **已完成**：使用 `scripts/repo_stats.py`，不再用仓库总行数代表代码量；
- **已完成**：保持 AGENTS 搜索边界和 Linguist generated 标记；
- **已完成**：用 `research/EXPERIMENT_INDEX.md` 记录 current/historical experiments 的维护责任、
  状态、入口、输出和权威结论；
- **已保持**：不动 frozen/sealed 路径。

### P1：低风险结构整理

- **已完成**：doctor、data、eval、lifecycle、influence、root service、两项 ingest 诊断以及
  `ingest run` / `ingest fidelity` 全部拆入 `src/llm_long_term_memory/commands/`，
  `cli.py` 从 1,968 行降到 344 行；命令、参数与帮助文本不变，由
  `tests/cli_surface.txt` 与 `tests/test_cli_surface.py` 固定；
- **已完成**：精确 McNemar 的四份实现收敛到 `llm_long_term_memory/stats.py`，
  证据读写收敛到 `tools/analysis_io.py`；八个 v2 分析器中七个重跑逐字节一致，
  第八个暴露出输出随哈希种子变化的缺陷并已修复；
- **已完成**：`research/README.md` 写明 active/archived 的进入条件，以及 `tools/`
  （可复现证据生成器）与 `scripts/`（依赖本机状态的人工/运维脚本）的判据；本轮不批量移动历史文件。

### P2：解除产品与 benchmark 耦合

- **已完成**：`llm_long_term_memory/conversation.py` 提供 `ConversationTurn`、
  `ConversationSession`、`AnswerRequest` 与 `ConversationSource`；LongMemEval 的类型改为继承它们；
- **已完成**：`ingest/` 全线不再 import 基准类型（`coverage.py` 因按金标打分已移入
  `evaluation/extraction_coverage.py`，`lltm ingest coverage` 命令不变）；API 不再构造
  `answer=""` 的假 `Instance`；
- **已完成**：`MemoryRunner` 在 `answer(instance)`（评测适配器）与 `answer_request(request)`
  （引擎）之间分开，回答契约移入 `llm_long_term_memory/answering.py`，prompt 文本与版本逐字节不变；
- **待办**：引擎在物理上仍位于 `evaluation/runners/`；`influence/runner.py` 仍按金标打分，
  属于放错包的评测代码；
- 每一步都跑了 API、ingest、租户隔离、fallback 与完整测试。

### P3：经人工确认后归档/清理

- 把确认结束的 runner 移到 `research/archived/`，但先更新所有哈希、测试和文档引用；
- 决定 partial/superseded generated artifacts 是否保留；
- 清理无引用语言 badge；
- 只在证据链完整且可重建成本可接受时删除 generated 产物。

## 十一、预期效果

- **core code 不会因本轮减噪而减少**；重构后也应以边界清晰为目标，不追求机械降 LOC。
- 若只清缓存，Git 仓库总行数基本不变；若未来经确认移出/删除可重建 raw/partial artifact，总行数可
  明显下降，但 frozen evidence 必须保留，所以不会变成一个只有几万行的纯产品仓库。
- Codex context 会明显减少：默认搜索不再进入占 83% 行数的 `results/`，普通改动的有效搜索面从
  约 46 万行降到 `src/ + 对应 tests` 的约 4.2 万行。
- GitHub 展示会更准确：generated experiment data 不再冒充主要语言源码，而 Python 不被隐藏。

本报告到此暂停。任何历史脚本移动、artifact 删除或大规模目录调整，都需要下一步明确确认。
