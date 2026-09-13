# 运维手册 —— 安装、认证、轮换、备份恢复

只写已经实际跑通并有测试钉住的内容。没有验证过的能力不写在这里。

## 干净环境安装

```bash
git clone <repo> && cd llm-long-term-memory
uv sync --group dev --extra llm --extra api --extra mcp
uv run pytest
```

**已验证**（2026-09-13，从干净克隆执行）：安装成功，全套测试通过，1 项跳过（依赖本机私有
store 的产物复放）。`embed` extra 刻意不装 —— 它带约 2GB 的 torch，而 `Encoder` 在属性里惰性
加载，收集测试时不需要它。

启动服务：

```bash
uv run uvicorn llm_long_term_memory.api.app:app --host 127.0.0.1 --port 8000
```

### Docker

```bash
docker build -t lltm .                       # 默认含 embed，约 2GB torch
docker build --build-arg EXTRAS="api" -t lltm-lite .   # 只读部署，无语义检索
docker run -p 8000:8000 -v /data:/data lltm
```

**已验证**（2026-09-13）：轻量变体构建成功并启动。它的 `/healthz` 返回 **503** 并说明
`semantic search unavailable: No module named 'sentence_transformers'` —— 这是正确行为，
不是故障：不能检索的实例不应该被调度器当成就绪。Dockerfile 的注释也这样声明。

完整变体（含 `embed`）未在本次演练中构建，两者只差安装的包。

镜像里不含数据、模型权重或凭据；store 从挂载卷来。

## 认证：默认是开放的，这是刻意的，也必须是知情的

**没有配置 token 时，任何调用方可以指定任何命名空间。** 研究 CLI、inspector 和离线测试套件
都这样驱动服务，所以开放模式保留了下来。它现在是一个被声明的状态而不是未经审视的默认：

```bash
curl -s localhost:8000/healthz | jq .authenticated_access   # false = 开放模式
```

启用认证：

```bash
export LLTM_API_TOKENS="secret-a:tenant-a,secret-b:tenant-b"
```

调用方带 `Authorization: Bearer secret-a`。命名空间由凭据决定；请求里写了不同的 `user_id`
会得到 **403**，而不是被静默改写 —— 静默改写会让越权尝试看起来像一次对空命名空间的成功调用。

**配置写错不会退回开放模式。** `LLTM_API_TOKENS=mytoken`（漏了 `:tenant`）会让服务在
`/v1/*` 上返回 503，而不是静默地关掉全部认证。这个行为有测试。

### 凭据轮换（无停机）

同一租户可以同时挂多个 token，所以轮换是三步：

```bash
# 1. 加新的，旧的仍然有效
export LLTM_API_TOKENS="old-secret:tenant-a,new-secret:tenant-a"
# 2. 客户端切到 new-secret
# 3. 撤掉旧的
export LLTM_API_TOKENS="new-secret:tenant-a"
```

**已验证**：两个 token 都解析到同一租户、看到同一份数据；撤销后旧 token 立即失效。

**一个会被拒绝的错误**：同一个 secret 指向两个租户（`shared:tenant-a,shared:tenant-b`）会让
配置整体失败，因为那会让任一租户读到另一个的数据。

### 这不是什么

Bearer token 是刻意选的弱方案，为了能现在落地并被测试。它们**不过期、不带 claim、撤销必须改
环境变量**。`principal_from_token` 是 OIDC/JWT 验证器替换进来的接缝。

MCP 的 HTTP transport **没有**等价机制，不得暴露到 localhost 之外。

## 备份与恢复演练

```bash
uv run python tools/backup_restore.py backup --store stores/two-stage-p10.db --into /backups/$(date +%F)
uv run python tools/backup_restore.py verify --backup /backups/2026-09-13 --into /tmp/drill
```

`verify` 恢复到一个**空目录**并检查三件事：行数、SQLite 自身的完整性检查、以及向量索引与恢复
出来的记忆是否一致。

**第三项是关键。** 数据库和索引是两个文件，没有任何机制让它们原子。一次在写入中途拿到的备份
可能包含索引从未见过的记忆，而这种不对称在检索静默漏掉它们之前是看不见的。需要修复时：

```bash
uv run python tools/backup_restore.py verify --backup ... --into ... --rebuild-index
```

重建用本地编码器，**恢复演练永远不需要 provider**。

**已验证**（2026-09-13）：对真实的 6,233 条记忆 / 24,590 轮对话的 store 演练，行数一致、完整性
ok、索引一致。十项测试钉住这条路径：按真实故障的方式破坏备份（文件损坏、文件缺失、索引缺行），
检查恢复流程**报告**问题而不是返回一个看起来干净的目录；并在每个平台上按 Windows 的规则检查，
重建不会写入仍被内存映射的索引文件。

## 重复写入

写入路径先落盘 turn 再做抽取，所以 provider 失败后的普通重试会追加一条重复的 turn。带上
幂等键即可避免：

```bash
curl -X POST localhost:8000/v1/messages \
  -H "Authorization: Bearer secret-a" \
  -H "Idempotency-Key: $(uuidgen)" \
  -d '{"role":"user","content":"I live in Canberra"}'
```

重放返回第一次的回复并带 `idempotent_replay: true`，不会再写、也不会再付抽取的钱。键按命名
空间隔离，删除数据时一并删除。

**409** 表示另一次调用已认领该键且尚未完成 —— 重试，而不是改请求。

## 还没有的东西

诚实起见列在这里，因为运维手册最容易变成能力清单：

- **没有跨进程并发保障。** 服务用进程内的 `RLock` 串行化 SQLite 连接和向量索引。多进程部署
  会绕过它。
- **锁跨 provider 调用持有。** 一次回答或写入期间，整个服务实际一次只处理一个请求。
- **没有账号额度与告警。** 用量被计量并记录，但没有上限和通知。
- **没有自动备份调度。** 上面的备份是手动命令。
- **删除不跨资源原子。** SQLite 行先删、向量后删，中途崩溃留下孤立向量而不是可被检索到的
  幽灵记忆 —— 这是较安全的方向，但不是原子的。
- **备份不含**：删除只作用于在线数据面，已有备份不受影响。
