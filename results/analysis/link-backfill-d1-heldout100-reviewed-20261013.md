# D 回填审计：heldout100

> 由 tools/link_backfill_audit.py 生成；零模型调用，开发证据。
> 关系与关闭复查均为助手草稿，待维护者确认；不是端到端答题成绩。

- 完成用户：18/100；仅审计完成用户。
- 累计判定：110；replace 应用次数：4。
- 相对原库实际改变的关闭关系：4；它与 replace 应用次数不同。
- 关闭复查：合理 1、错误 2、不确定 1、未复查 0。
- 已明确标注的关闭精确率：1/3；排除了不确定／未复查，不能当作全库真实精确率。
- 已有草稿替换对（含旧时点／新时点探针）通过：1/1；并存对状态保持：1/1。
- 旧草稿替换对与本次复查冲突：1；排除争议后的草稿对子集通过 0/0。
- 跨用户后继错误：0；非预期存储字段变化：0。
- 缓存正反遍历比较：2226/2226 条状态一致（18 个用户）。
- 正向重放与实际 d1 结果不一致用户：0。

批内遍历顺序检查不等于任意乱序到达实验。全库召回率与误替换率需要完整正负例标注；
当前仅能给出已有草稿对子集的结果和实际关闭复查。TERMINATE 判定尚未应用，不能宣称
全操作类型的时间状态正确。缓存缺失的用户不进入顺序对比。

## 实际关闭逐对复查

| 用户 | 旧事实 | 后继事实 | 草稿结论 | 原因 |
|---|---|---|---|---|
| 0977f2af | The user ran a charity 5K on May 6, 2023, in 27 minutes and 12 seconds. | The user finished a charity 5K run two weeks ago (around 2023-05-07) with a time of 24 minutes and 12 seconds. | incorrect | 两条是带日期的比赛事件及完赛时间，不是同一个可变当前属性；个人最好成绩的变化不应关闭过去比赛事实。 |
| 099778bb | The user turned 30 on February 12th, 2023. | The user turned 32 in April 2023. | uncertain | 同年2月满30与4月满32的生日陈述互相矛盾，且都指历史事件；没有明确纠正原话，不把矛盾生日简单当作正常年龄增长。 |
| 0f05491a | The user needs 125 stars to reach the Gold level on the Starbucks Rewards app. | The user needs 120 stars to reach the Gold level on the Starbucks Rewards app. | correct | 同一Rewards app达到Gold所需星数的当前陈述从125改为120。 |
| 18bc8abd | The user intends to serve Sweet Baby Ray's BBQ sauce. | The user likes Kansas City Masterpiece BBQ sauce. | incorrect | 早先聚会计划供应Sweet Baby Ray与后来喜欢KC酱可以共存；未说明早先聚会取消或换酱。 |

## 时间线重建（仅临时副本）

- 对实际 d1 副本完整重建一次，改变 7 条状态。
- 再重建一次，改变 0 条状态；第二次为零才说明该重建结果幂等。

原库对照重建改变 7 条；与 d1 共有 7 条，仅 d1 出现的改变为 0 条。
共有变化不归因于本次关系判定回填。

## 标注争议（保留旧标注）

- `18bc8abd`：旧草稿 `replace`；本次关闭复查 `incorrect`：早先聚会计划供应Sweet Baby Ray与后来喜欢KC酱可以共存；未说明早先聚会取消或换酱。
