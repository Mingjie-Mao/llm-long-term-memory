# D 回填审计：dev100

> 由 tools/link_backfill_audit.py 生成；零模型调用，开发证据。
> 关系与关闭复查均为助手草稿，待维护者确认；不是端到端答题成绩。

- 完成用户：100/100；仅审计完成用户。
- 累计判定：497；replace 应用次数：36。
- 相对原库实际改变的关闭关系：27；它与 replace 应用次数不同。
- 关闭复查：合理 11、错误 12、不确定 4、未复查 0。
- 已明确标注的关闭精确率：11/23；排除了不确定／未复查，不能当作全库真实精确率。
- 已有草稿替换对（含旧时点／新时点探针）通过：6/6；并存对状态保持：4/4。
- 旧草稿替换对与本次复查冲突：1；排除争议后的草稿对子集通过 5/5。
- 跨用户后继错误：0；非预期存储字段变化：0。
- 缓存正反遍历比较：12434/12436 条状态一致（100 个用户）。
- 正向重放与实际 d1 结果不一致用户：0。

批内遍历顺序检查不等于任意乱序到达实验。全库召回率与误替换率需要完整正负例标注；
当前仅能给出已有草稿对子集的结果和实际关闭复查。TERMINATE 判定尚未应用，不能宣称
全操作类型的时间状态正确。缓存缺失的用户不进入顺序对比。

## 实际关闭逐对复查

| 用户 | 旧事实 | 后继事实 | 草稿结论 | 原因 |
|---|---|---|---|---|
| 078150f1 | The user is planning to join a triathlon in June. | The user is considering signing up for a triathlon in June. | incorrect | 5月28日计划参加被5月20日考虑参加关闭；原话未表示取消计划，实际时间方向也反了。 缓存判定的A/B时间方向正确（考虑→计划），错误出现在应用/重述折叠，不能全归因关系判定。 |
| 078150f1 | The user aimed to raise $200 in donations for a local children's hospital during a charity cycling event. | The user raised $250 in donations for a charity cycling event. | incorrect | 目标金额与实际募集金额是不同属性；完成结果不改写历史目标。 |
| 07b6f563 | The user plans to wake up at 9:00 am on Saturdays to plan their week, do meal prep, and exercise. | The user usually likes to sleep in on Saturdays. | incorrect | 同会话turn6的新计划被turn4的通常睡懒觉关闭；没有放弃计划的陈述，时间戳相同不能用ID推断反向更新。 |
| 07b6f563 | The user's goal was to complete 20 workout classes. | The user completed 16 workout classes and earned a 10% discount on their next month's membership. | incorrect | 同一句原话明确完成16节、比20节目标少4节；目标与进展应并存。 |
| 2698e78f | The user has a therapy session with Dr. Smith every two weeks. | The user sees Dr. Smith every week for help with setting healthy boundaries. | correct | 同一医生会面频率由每两周改成每周；全会话turn4明确每周，v2锚点邻域未定位到频率原句。 |
| 352ab8bd | The user has a gray and beige rug in their living room. | The user has a new coffee table and rug in their living room. | uncertain | 新地毯未声明取代旧灰米色地毯；也可能是同一地毯的较笼统重述，证据不够确认关闭。 |
| 3e321797 | The user plans to purchase travel-sized versions of their hair dryer and straightening iron. | The user decided to pack their hair dryer and straightening iron in their carry-on luggage. | incorrect | 同一轮次同时说随身携带并寻找旅行装；购买计划和行李位置可以同时成立。 |
| 41698283 | The user is considering getting Nigerian Dwarf goats as pets. | The user acquired two Nigerian Dwarf goats about three weeks before May 23, 2023. | correct | 考虑养两只山羊的待办意图在已获得两只山羊后完成；来源明确已获得，按当前计划状态关闭合理。 |
| 41698283 | The user is considering the Lowepro ProTactic 450 AW and the Think Tank Urban Disguise 40L camera bags. | The user decided to purchase a Think Tank Urban Disguise 40L camera bag. | correct | 同一相机包的待决选择变为明确选择Think Tank；全会话turn8确认，锚点邻域未包含该轮。 |
| 4d6b87c8 | The user has a to-watch list of 20 titles. | The user has 25 titles on their to-watch list. | correct | 同一待看片单的当前标题数由20增加到25。 |
| 4d6b87c8 | The user is looking for a dance studio in their city that offers bachata classes. | The user takes salsa classes at a local studio that also offers bachata classes. | correct | 后续原话明确已经找到喜欢的舞蹈工作室，找店状态结束。 |
| 5d3d2817 | The user is planning a dinner party with a seafood theme. | The user plans to host a winter-themed dinner party. | uncertain | 未说明是同一晚宴或取消海鲜主题；冬季主题与海鲜菜单也可能共存。 |
| 72e3ee87 | The user is on episode 10 of the Crash Course Science series. | The user has completed 50 episodes of Crash Course's Science series. | correct | 同一Science系列的观看进度由第10集更新为完成50集。 |
| 7a87bd0c | The user has been sticking to a daily tidying routine for 3 weeks as of 2023-06-11. | The user has been following a daily tidying routine for 4 weeks as of 2023-09-30. | correct | 同一每日整理习惯的当前持续时长由3周变成4周；但会话相隔数月，不能据此推断连续未中断。 |
| 945e3d21 | The user prefers a traditional style TV stand with ornate details like carved wooden legs or decorative trim. | The user prefers a wooden TV stand with storage bins. | incorrect | 传统雕花风格和木质带储物箱可兼容；原话没有改掉风格偏好。 |
| b01defab | The user started taking yoga classes at Zen Den 6 weeks ago. | The user has been taking yoga classes since July. | incorrect | 特定Zen Den课程开始时间与整体瑜伽练习开始时间不同，后文未否定Zen Den事实；不能当成同一状态替换。 |
| b46e15ee | The user owns a KitchenAid stand mixer. | The user has a new stand mixer they got two months ago. | uncertain | 两个月前的新搅拌机可能就是同一KitchenAid，也可能另外购买；未明确旧机器停止拥有。 |
| c4ea545c | The user goes to the gym on Tuesdays, Thursdays, and Saturdays. | The user exercises at the gym four times a week. | correct | 按原知识更新题标注草稿，旧三天周安排视为完整安排，由每周四次替换；新原话并未给出四个具体星期几。 |
| ce6d2d27 | The user has a cocktail-making class on the upcoming Thursday. | The user attends a cocktail-making class on Fridays. | incorrect | 一次即将到来的周四课程与每周五课程的时间范围不同，原话未表明同一课程改期或取消；独立事件应保留。 |
| d851d5ba | The user is considering participating in a charity bike ride. | The user is planning to participate in a charity cycling event next month. | correct | 同日相隔44分钟，从考虑慈善骑行变为计划下月参加，按同一意图的状态推进关闭合理；题目评分歧义另行处理。 |
| e25c3b8d | The user runs 20-25 miles per week. | The user runs 3 times a week for 3-4 miles on roads. | correct | 新原话每周3次、每次平均3-4英里，对应约9-12英里，作为同一用户当前跑量的后续状态与20-25不一致；推导依据保留。 |
| e831120c | The user wants to go to bed at 11:30 pm and get 7-8 hours of sleep. | The user wakes up at 7 am and goes to bed at 10:30 pm. | incorrect | 希望11:30入睡的目标与通常10:30入睡的习惯是不同属性；未明确取消或修改目标。 |
| gpt4_1a1dc16d | The user plans to set workout reminders on Google Calendar and within MyFitnessPal. | The user decided to use MyFitnessPal to track their workouts. | incorrect | 同一句原话同时承诺MyFitnessPal追踪和Calendar/应用内提醒；追踪与提醒用途应并存。 |
| gpt4_213fd887 | The user is planning a trip to Maui. | The user decided to add an extra day to their Maui trip. | incorrect | 同一会话给Maui行程增加一天，并未取消旅行计划；存在行程与行程时长属于不同信息。 |
| gpt4_59c863d7 | The user listens to indie folk artists including Bon Iver, Sufjan Stevens, and Fleet Foxes. | The user enjoys indie folk artists Phoebe Bridgers, Fleet Foxes, and Bon Iver. | incorrect | 新列表增加Phoebe Bridgers，未说不再听Sufjan Stevens；非排他的艺人喜好列表不能因漏提某成员关闭旧事实。 |
| gpt4_65aabe59 | The user upgraded their home Wi-Fi router 3 weeks before May 25, 2023. | The user has had a new dual-band router since mid-January 2023. | uncertain | 5月25日前三周升级与1月中旬开始使用路由器可能为不同设备或历史事件；未确认同一升级，显式事件时间也不同。 |
| gpt4_76048e76 | The user is located in the 10021 zip code in New York City. | The user's zip code is 30309. | correct | 后续明确当前所在邮编30309，更新先前10021的位置；这只支持当前位置，不推断永久搬家。 |

## 时间线重建（仅临时副本）

- 对实际 d1 副本完整重建一次，改变 66 条状态。
- 再重建一次，改变 0 条状态；第二次为零才说明该重建结果幂等。

原库对照重建改变 66 条；与 d1 共有 66 条，仅 d1 出现的改变为 0 条。
共有变化不归因于本次关系判定回填。

## 标注争议（保留旧标注）

- `ce6d2d27`：旧草稿 `replace`；本次关闭复查 `incorrect`：一次即将到来的周四课程与每周五课程的时间范围不同，原话未表明同一课程改期或取消；独立事件应保留。

## 正反遍历差异（保留失败）

- `7024f17c` / `mem_42bc11f020cd57df`：正向 `{'status': 'active', 'superseded_by': None, 'valid_from': datetime.datetime(2023, 5, 23, 18, 3), 'valid_to': None, 'subject': 'user', 'predicate': 'yoga_practice', 'update_op': 'replaces', 'target_object': 'three times a week for 2 hours', 'replaces_previous': True}`；反向 `{'status': 'active', 'superseded_by': None, 'valid_from': datetime.datetime(2023, 5, 23, 18, 3), 'valid_to': None, 'subject': 'user', 'predicate': 'yoga_practice', 'update_op': 'replaces', 'target_object': 'Monday, Wednesday, Friday at 7:00 am', 'replaces_previous': True}`。
- `7024f17c` / `mem_c05d28c658297822`：正向 `{'status': 'superseded', 'superseded_by': 'mem_42bc11f020cd57df', 'valid_from': datetime.datetime(2023, 5, 22, 23, 10), 'valid_to': datetime.datetime(2023, 5, 23, 18, 3), 'subject': 'user', 'predicate': 'yoga_practice', 'update_op': 'replaces', 'target_object': None, 'replaces_previous': True}`；反向 `{'status': 'active', 'superseded_by': None, 'valid_from': datetime.datetime(2023, 5, 22, 23, 10), 'valid_to': None, 'subject': 'user', 'predicate': 'yoga_practice', 'update_op': 'replaces', 'target_object': None, 'replaces_previous': True}`。
