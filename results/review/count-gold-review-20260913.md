# 计数金标人工复核表（2026-09-13）

状态：待人工填写。38 条既有裁定来自模型，本文件没有批准或改变任何裁定。

复核目标是“该提取记忆是否属于问句要求的集合”。不展示候选/对照答案、分数或裁定影响；原模型建议仍列出供核对，因此这不是完全盲审。

对每条填写：保留 / 剔除 / 问句或计数单位需重写，并给出理由。整批确认之前，修正分数只作探索性分析。即使全部确认，仍需解决一条记忆多个成员、重复成员及谓词范围与动词冲突的问题。

对应可编辑决策文件：[decisions.json](count-gold-review-20260913.decisions.json)。该文件所有人工字段均为空。

计数单位尚为记忆行，不保证等同于 distinct 实体。隐藏探针未纳入本复核表。

## 01 · count_0005 · mem_c6c3e5ed8012c3be

问题：How many distinct dishes has the user cooked? Count only facts recorded under: cooking ingredients, cooking skills, recipe modifications.

提取记忆：The user decided to add cherry tomatoes and mushrooms to their Spaghetti Carbonara.

主体 / 谓词：`user` / `recipe_modifications`
来源：session `scoped-session-v1:13:gpt4_a2d1d1f6a1bfb382_1`，turn `2`。

模型暂定：**exclude**；规则 `R3`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 02 · count_0005 · mem_e47b0804100a5c57

问题：How many distinct dishes has the user cooked? Count only facts recorded under: cooking ingredients, cooking skills, recipe modifications.

提取记忆：The user plans to use avocado oil for roasting asparagus.

主体 / 谓词：`user` / `cooking_ingredients`
来源：session `scoped-session-v1:13:gpt4_a2d1d1f6a1bfb382_1`，turn `9`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 03 · count_0007 · mem_dd8eab07beb913d4

问题：How many distinct possessions does the user have on record? Count only facts recorded under: owned items, photography gear, stamp collection.

提取记忆：The user plans to use an Epson Perfection V550 scanner and Adobe Lightroom software to digitize negatives.

主体 / 谓词：`user` / `photography_gear`
来源：session `scoped-session-v1:13:gpt4_a2d1d1f6f5cf8815_4`，turn `10`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 04 · count_0011 · mem_6bd7f8c08d2d72cf

问题：How many distinct financial activities are on record for the user? Count only facts recorded under: expense tracking fields, expense tracking method, interests, renovation budget.

提取记忆：The user is interested in exploring gender identity and expression through stream-of-consciousness writing.

主体 / 谓词：`user` / `interests`
来源：session `scoped-session-v1:8:8ebdbe5039911f94`，turn `3`。

模型暂定：**exclude**；规则 `R3`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 05 · count_0011 · mem_9567d36ae03780a0

问题：How many distinct financial activities are on record for the user? Count only facts recorded under: expense tracking fields, expense tracking method, interests, renovation budget.

提取记忆：The user wants to track date of purchase, item description, cost, and store/website for their expenses.

主体 / 谓词：`user` / `expense_tracking_fields`
来源：session `scoped-session-v1:8:8ebdbe5012cff183_1`，turn `2`。

模型暂定：**exclude**；规则 `R1`；置信标记 `disputed`。

模型争议说明：两个方向都成立。问题措辞是 `are on record for the user`，是全部题目里最宽松的一种，且范围明确含 `expense tracking fields`；但记忆本身是意图（`wants to track`）。维持 exclude 不改数字，标为争议交人裁定。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 06 · count_0012 · mem_100a159f47e3286a

问题：How many distinct films or shows has the user watched? Count only facts recorded under: tv shows.

提取记忆：The user plans to watch Breaking Bad.

主体 / 谓词：`user` / `tv_shows`
来源：session `scoped-session-v1:13:gpt4_8e165409aee13015_3`，turn `1`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 07 · count_0012 · mem_4593f716b042d96f

问题：How many distinct films or shows has the user watched? Count only facts recorded under: tv shows.

提取记忆：The user plans to watch Ozark.

主体 / 谓词：`user` / `tv_shows`
来源：session `scoped-session-v1:13:gpt4_8e16540937067e75_1`，turn `5`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 08 · count_0012 · mem_c4d5417b8c50a3f1

问题：How many distinct films or shows has the user watched? Count only facts recorded under: tv shows.

提取记忆：The user watches the Daily Dose of Laughter playlist on Peacock.

主体 / 谓词：`user` / `tv_shows`
来源：session `scoped-session-v1:13:gpt4_8e16540937067e75_1`，turn `0`。

模型暂定：**keep**；规则 `R4`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 09 · count_0012 · mem_fa51858f617410ba

问题：How many distinct films or shows has the user watched? Count only facts recorded under: tv shows.

提取记忆：The user plans to watch Narcos: Mexico.

主体 / 谓词：`user` / `tv_shows`
来源：session `scoped-session-v1:13:gpt4_8e16540937067e75_1`，turn `9`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 10 · count_0015 · mem_2bf4d65c6a390121

问题：How many distinct hobbies or activities does the user practise? Count only facts recorded under: business activities, current activities.

提取记忆：The user is planning to participate in the Summer Fest event in June 2023.

主体 / 谓词：`user` / `business_activities`
来源：session `scoped-session-v1:8:561fabcd3be19aed_1`，turn `11`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 11 · count_0016 · mem_0950c5aa6a7994e6

问题：How many distinct books or articles has the user read? Count only facts recorded under: beta reading, book club reading, books read, reading interests, reading tracker.

提取记忆：The user's book club is reading 'The Power' by Naomi Alderman.

主体 / 谓词：`user` / `book_club_reading`
来源：session `scoped-session-v1:13:gpt4_78cf46a37fd52e1c_2`，turn `9`。

模型暂定：**keep**；规则 `R4`；置信标记 `high`。

模型争议说明：问题范围明确含 `book club reading`，所以范围声明本身把书友会在读的书纳入成员。原判 keep 依据动词，现在有范围声明这一独立依据。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 12 · count_0016 · mem_724d251d6c2c0cc3

问题：How many distinct books or articles has the user read? Count only facts recorded under: beta reading, book club reading, books read, reading interests, reading tracker.

提取记忆：The user wants recommendations for thrillers and mysteries by authors from diverse backgrounds.

主体 / 谓词：`user` / `reading_interests`
来源：session `scoped-session-v1:13:gpt4_78cf46a3e8e65a1f`，turn `3`。

模型暂定：**exclude**；规则 `R2`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 13 · count_0016 · mem_a3ba4ae3193da343

问题：How many distinct books or articles has the user read? Count only facts recorded under: beta reading, book club reading, books read, reading interests, reading tracker.

提取记忆：The user is looking for thriller and mystery book recommendations.

主体 / 谓词：`user` / `reading_interests`
来源：session `scoped-session-v1:13:gpt4_78cf46a3e8e65a1f`，turn `3`。

模型暂定：**exclude**；规则 `R2`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 14 · count_0018 · mem_fc8c00f5a7db4811

问题：How many distinct plants does the user grow? Count only facts recorded under: garden plants, gardening tasks.

提取记忆：The user is hardening off seedlings 7-10 days before transplanting them into a raised bed.

主体 / 谓词：`user` / `gardening_tasks`
来源：session `scoped-session-v1:8:6456829ec44d1533_2`，turn `1`。

模型暂定：**exclude**；规则 `R3`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 15 · count_0022 · mem_38bc79b0f37a9b67

问题：How many distinct books or articles has the user read? Count only facts recorded under: books read, books reading, reading interests.

提取记忆：The user is interested in historical fiction, specifically World War II stories focusing on women's experiences.

主体 / 谓词：`user` / `reading_interests`
来源：session `scoped-session-v1:13:gpt4_7a0daae17c08d74b_2`，turn `2`。

模型暂定：**exclude**；规则 `R2`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 16 · count_0022 · mem_a7b311f7b3a3f085

问题：How many distinct books or articles has the user read? Count only facts recorded under: books read, books reading, reading interests.

提取记忆：The user is reading The Seven Husbands of Evelyn Hugo by Taylor Jenkins Reid.

主体 / 谓词：`user` / `books_reading`
来源：session `scoped-session-v1:13:gpt4_7a0daae1445e6a7a_2`，turn `0`。

模型暂定：**keep**；规则 `R4`；置信标记 `high`。

模型争议说明：问题范围明确含 `books reading`，即正在读的书。范围声明独立支持 keep。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 17 · count_0024 · mem_0822e99c062b1730

问题：How many distinct tasks or projects has the user completed? Count only facts recorded under: project development, projects, task.

提取记忆：The user is planning a community garden project in their neighborhood.

主体 / 谓词：`user` / `projects`
来源：session `scoped-session-v1:8:7527f7e218a06652_3`，turn `2`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 18 · count_0024 · mem_0970283ff75f8606

问题：How many distinct tasks or projects has the user completed? Count only facts recorded under: project development, projects, task.

提取记忆：The user is creating an AI-generated mind map tool for their website.

主体 / 谓词：`user` / `project_development`
来源：session `scoped-session-v1:8:7527f7e2sharegpt_1i8PAcL_0`，turn `0`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 19 · count_0024 · mem_4f028e5f132df883

问题：How many distinct tasks or projects has the user completed? Count only facts recorded under: project development, projects, task.

提取记忆：The user needs to reorder Luna's medication tomorrow during their lunch break.

主体 / 谓词：`user` / `task`
来源：session `scoped-session-v1:8:7527f7e2d81d9846`，turn `6`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 20 · count_0025 · mem_5b7eb955bbb3888c

问题：How many distinct hobbies or activities does the user practise? Count only facts recorded under: hobbies, volunteering interests.

提取记忆：The user is interested in protecting marine ecosystems.

主体 / 谓词：`user` / `volunteering_interests`
来源：session `scoped-session-v1:8:0bc8ad92717d1aea_1`，turn `8`。

模型暂定：**exclude**；规则 `R2`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 21 · count_0025 · mem_d2571c170df1890d

问题：How many distinct hobbies or activities does the user practise? Count only facts recorded under: hobbies, volunteering interests.

提取记忆：The user is interested in volunteering for conservation projects or wildlife conservation efforts.

主体 / 谓词：`user` / `volunteering_interests`
来源：session `scoped-session-v1:8:0bc8ad92717d1aea_1`，turn `5`。

模型暂定：**exclude**；规则 `R2`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 22 · count_0026 · mem_b04bd195cda70394

问题：How many distinct possessions does the user have on record? Count only facts recorded under: birding equipment, camera gear, home ownership.

提取记忆：The user listed a Canon Powershot G7 X Mark II on Craigslist for $400.

主体 / 谓词：`user` / `camera_gear`
来源：session `scoped-session-v1:8:1a1907b465b01f7c`，turn `5`。

模型暂定：**keep**；规则 `R5`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 23 · count_0030 · mem_0c15f854c875f544

问题：How many distinct books or articles has the user read? Count only facts recorded under: books read, reading material, reading speed.

提取记忆：The user averages 55 pages of reading per day.

主体 / 谓词：`user` / `reading_speed`
来源：session `scoped-session-v1:8:8a2466dbb67748d1_1`，turn `4`。

模型暂定：**exclude**；规则 `R3`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 24 · count_0030 · mem_9077370dd750ac1a

问题：How many distinct books or articles has the user read? Count only facts recorded under: books read, reading material, reading speed.

提取记忆：The user has been reading The Huffington Post and Politico during their lunch break for the last three months.

主体 / 谓词：`user` / `reading_material`
来源：session `scoped-session-v1:8:8a2466db6be54739_3`，turn `0`。

模型暂定：**keep**；规则 `R4`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 25 · count_0032 · mem_07eb659cca3861c6

问题：How many distinct hobbies or activities does the user practise? Count only facts recorded under: personal projects, physical activities, religious activity, volunteer interests, volunteering interests.

提取记忆：The user is considering volunteering at a local charity event.

主体 / 谓词：`user` / `volunteering_interests`
来源：session `scoped-session-v1:8:cc5ded98bd8708e2_2`，turn `6`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 26 · count_0032 · mem_318f016f3e9f49c4

问题：How many distinct hobbies or activities does the user practise? Count only facts recorded under: personal projects, physical activities, religious activity, volunteer interests, volunteering interests.

提取记忆：The user is interested in volunteering as a legal observer for the ACLU.

主体 / 谓词：`user` / `volunteer_interests`
来源：session `scoped-session-v1:8:cc5ded98ultrachat_323437`，turn `7`。

模型暂定：**exclude**；规则 `R2`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 27 · count_0032 · mem_9cd6c5d077811c0c

问题：How many distinct hobbies or activities does the user practise? Count only facts recorded under: personal projects, physical activities, religious activity, volunteer interests, volunteering interests.

提取记忆：The user plans to build a web scraper or a chatbot as personal projects.

主体 / 谓词：`user` / `personal_projects`
来源：session `scoped-session-v1:8:cc5ded98answer_a5b68517_2`，turn `8`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 28 · count_0035 · mem_c756f3abc723123a

问题：How many distinct events has the user attended? Count only facts recorded under: events attended, medical appointment, photography workshop.

提取记忆：The user has a dental crown procedure scheduled for Friday, May 26, 2023.

主体 / 谓词：`user` / `medical_appointment`
来源：session `scoped-session-v1:8:3b6f954b28209b6a_3`，turn `0`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 29 · count_0039 · mem_26ccf38f83833ce2

问题：How many distinct dishes has the user cooked? Count only facts recorded under: recipe interest, recipe to make, recipes tried.

提取记忆：The user is interested in vegan dessert recipes.

主体 / 谓词：`user` / `recipe_interest`
来源：session `scoped-session-v1:8:45dc21b6answer_07664d43_1`，turn `3`。

模型暂定：**exclude**；规则 `R2`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 30 · count_0039 · mem_28238b261bfdcaba

问题：How many distinct dishes has the user cooked? Count only facts recorded under: recipe interest, recipe to make, recipes tried.

提取记忆：The user decided to make the Chocolate Chia Pudding.

主体 / 谓词：`user` / `recipe_to_make`
来源：session `scoped-session-v1:8:45dc21b6answer_07664d43_1`，turn `3`。

模型暂定：**exclude**；规则 `R1`；置信标记 `medium`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 31 · count_0039 · mem_3b88ea2468932982

问题：How many distinct dishes has the user cooked? Count only facts recorded under: recipe interest, recipe to make, recipes tried.

提取记忆：The user is interested in brisket recipes.

主体 / 谓词：`user` / `recipe_interest`
来源：session `scoped-session-v1:8:45dc21b6eb34144a_2`，turn `3`。

模型暂定：**exclude**；规则 `R2`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 32 · count_0039 · mem_886cbee092d96877

问题：How many distinct dishes has the user cooked? Count only facts recorded under: recipe interest, recipe to make, recipes tried.

提取记忆：The user has already tried the vegan cheesecake recipe.

主体 / 谓词：`user` / `recipes_tried`
来源：session `scoped-session-v1:8:45dc21b6answer_07664d43_1`，turn `2`。

模型暂定：**keep**；规则 `R4`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 33 · count_0044 · mem_990ce959d4425acd

问题：How many distinct possessions does the user have on record? Count only facts recorded under: baseball collection, bike accessories, bikes owned, camera collection, coin collection, motorcycles owned, watch collection.

提取记忆：The user decided to purchase the Pro Bike Tool Waterproof Bike Cover.

主体 / 谓词：`user` / `bike_accessories`
来源：session `scoped-session-v1:12:0ddfec37_abse92792b8_2`，turn `4`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 34 · count_0048 · mem_0c5c366b6e325e6f

问题：How many distinct places has the user visited? Count only facts recorded under: commute route, past travel, travel destination, travel interests.

提取记忆：The user is planning to stay in Kailua-Kona.

主体 / 谓词：`user` / `travel_destination`
来源：session `scoped-session-v1:13:gpt4_74aed68ea4e89c45`，turn `4`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 35 · count_0048 · mem_80e30da43d28c645

问题：How many distinct places has the user visited? Count only facts recorded under: commute route, past travel, travel destination, travel interests.

提取记忆：The user needs to sort through photos from a family trip to Maui in April 2022.

主体 / 谓词：`user` / `past_travel`
来源：session `scoped-session-v1:13:gpt4_74aed68e5c0574b1`，turn `0`。

模型暂定：**keep**；规则 `R4`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 36 · count_0048 · mem_ecf50bd17e05d185

问题：How many distinct places has the user visited? Count only facts recorded under: commute route, past travel, travel destination, travel interests.

提取记忆：The user is interested in visiting Taiwan and Vietnam.

主体 / 谓词：`user` / `travel_interests`
来源：session `scoped-session-v1:13:gpt4_74aed68e5c0574b1`，turn `4`。

模型暂定：**exclude**；规则 `R2`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 37 · count_0052 · mem_641b61c027628568

问题：How many distinct financial activities are on record for the user? Count only facts recorded under: car expenses, home budget.

提取记忆：The user is aiming for a home budget of $320,000 with a 20% down payment.

主体 / 谓词：`user` / `home_budget`
来源：session `scoped-session-v1:8:a3838d2be02ddb14_1`，turn `2`。

模型暂定：**exclude**；规则 `R1`；置信标记 `disputed`。

模型争议说明：同上：措辞 `on record` 且范围明确含 `home budget`，但 `aiming for` 是意图。维持 exclude。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

## 38 · count_0056 · mem_d42bd370e9171a14

问题：How many distinct events has the user attended? Count only facts recorded under: church attendance, medical appointments.

提取记忆：The user is scheduled for an endoscopy on April 15th.

主体 / 谓词：`user` / `medical_appointments`
来源：session `scoped-session-v1:13:gpt4_4edbafa2f25cfdd1_3`，turn `10`。

模型暂定：**exclude**；规则 `R1`；置信标记 `high`。

- [ ] 保留
- [ ] 剔除
- [ ] 问句或计数单位需重写

人工理由：

