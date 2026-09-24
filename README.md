# llm-long-term-memory

跨会话保存用户事实，追踪变更，并在记忆缺少细节时找回原始对话。

[在线演示](https://lltm-memory.pages.dev) · [架构](docs/ARCHITECTURE.md) · [评测](docs/EVALUATION.md) · [项目报告](docs/PROJECT_REPORT.md)

## 结果

冻结 v2 在 100 道 LongMemEval-S 终测题上，每种方法运行一次：

| 方法 | 答对 | 回答＋评分 token |
|---|---:|---:|
| LLTM v2 | 72/100 | 159,458 |
| naive RAG | 65/100 | 1,352,847 |
| 整段历史 | 86/100 | 10,932,294 |

v2 比 naive RAG 多答对 7 题，配对检验 p=0.3368，不能证明稳定优势；整段历史多答对
14 题，p=0.0043。token 数来自同一次运行的 provider 用量，均不含三组共用的抽取阶段。
这些是**冻结 v2 的成绩，不是当前开发版本的成绩**。[原始汇总](results/final/test100-aggregate.md)

## 实现

```text
历史对话 → 两阶段事实抽取 → 去重 → 时间状态更新 → SQLite＋向量索引
                                                    ↓
问题 → 检索相关事实 → 必要时找回原始轮次 → 回答上下文
```

事实保留来源和时间；替换旧值时保留历史，`as_of()` 可查过去的状态。原始对话另存，
用于恢复抽取时丢掉的数字、日期或原话。提供 REST 和 MCP 接口。

在线演示展示真实的写入、检索和状态变化，但用浏览器句式规则生成事实，**不运行 LLM 抽取或生成回答**。

## 运行

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync --group dev --extra api --extra llm --extra embed --extra mcp
uv run uvicorn llm_long_term_memory.api.app:app --host 127.0.0.1 --port 8000
```

打开 <http://localhost:8000/docs>。新 checkout 没有实验数据库；检索使用本地嵌入模型，
`/v1/messages` 和 `/v1/answer` 还需要 `GEMINI_API_KEY`。

## 当前边界

- 写入保真度是主要风险之一。同批 60 会话把每请求会话数从 15 改为 8，具体信息召回从
  37.3% 升到 64.9%；另一个 780 会话库的 batch 8 基线只有 47.5%。加入条件修复后是
  60.5%，但调用约增至三倍。**这些是抽取指标，未证明端到端答题变好。**
  [批次实验](results/batch-size-result.md) · [修复结果](results/v2b-gate16-repair-decision.md)
- 全部 500 道 LongMemEval-S 题都已用于开发或终测。最新改动没有新的未见题集成绩，
  不能把开发结果当成新终测。[评测口径](docs/EVALUATION.md)
- SQLite 与 NumPy 索引适合单进程研究和集成；两者不是同一事务，不承诺多进程写入。
  [架构与恢复方式](docs/ARCHITECTURE.md)

更完整的设计、失败分析和未解决问题见[项目报告](docs/PROJECT_REPORT.md)。

[MIT 许可](LICENSE)
