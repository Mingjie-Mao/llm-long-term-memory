# v2d gate16 开发集结果

> **DEVELOPMENT RESULT**：16 题均来自已暴露开发集，不能代表 unseen final 表现，
> 也不用于宣称统计显著性。

## 结论

- candidate：**8/16（50.0%）**；
- 冻结 v2 baseline：**8/16（50.0%）**；
- 14 道机制题：2 wins / 2 losses / 10 ties，净值 +0；
- source-session recall：**16/16**；
- 决策：**`STOP_NO_48`**。

未晋级的直接原因是预注册规则要求机制题 wins > losses，而本次为 2 = 2。不扩到 48 题。

## 预注册门槛

| 门槛 | 结果 |
|---|---|
| 总正确数不低于 baseline | 通过 |
| 14 道机制题 wins > losses | 失败 |
| 两道 source-detail 无正确→错误退化 | 通过 |
| 所有代码计算均有有效 operand citation | 通过 |
| frozen/sealed 未修改 | 通过 |

- wins：1192316e, 0db4c65d
- losses：d23cf73b, gpt4_d12ceb0e

## 计算与拒算审计

真正由代码产出数值的题：**2**；拒绝不完整/无效操作数后保留模型文本的题：**9**。
- `gpt4_2f91af09`：sum，citation M1, M2, M4，有效。
- `0db4c65d`：duration，citation M1, M2，有效。

无效标签、数组错位、缺失日期来源、单位冲突和零百分比基线的拒算规则均由单元测试覆盖。

## 路由与成本

- operation 分布：`{'average': 1, 'count': 4, 'difference': 1, 'duration': 2, 'lookup': 5, 'sum': 3}`；
- fallback 分布：`{'archive_wide': 1, 'none': 14, 'source_local': 1}`；
- 摄取：243 次成功调用 / 253 次尝试，2,407,918 tokens；
- QA（含 provider 重试和 7 次零 token schema 拒绝）：34 次成功调用 / 58 次尝试，37,502 tokens。

## 协议偏差

The registered v2d.3 nested/expanded response schema was rejected before any row was written. Provider-compatibility-only revisions produced v2d.4 compact schema; the first effective run then wrote all 16 rows under the registered result label.
没有任何失败请求写入答案行；v2d.4 的 16 行是该标签第一次有效运行。
