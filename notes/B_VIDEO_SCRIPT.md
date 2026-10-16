# B 视频讲稿与全组页序

本稿基于固定训练集发现；模型数值以 `notes/RESULTS.md` 为准。建议 B 主段约 2 分 40 秒，按实际朗读速度彩排。不要把本稿当作已录制或已人工复核。

## B 英文讲稿（约 320 词）

My part investigates why these preparation choices are worth testing. All exploratory results here use only the fixed training partition: 27,126 records, including 3,173 subscriptions.

Our main finding is that an unknown value cannot be interpreted in isolation. We cross-tabulated three historical fields: pdays, previous, and poutcome. In the training data, every record with nonnegative pdays also had a positive previous-contact count, while every record with pdays equal to minus one had zero previous contacts.

However, an unknown previous outcome did not always mean no previous contact. There were 22,235 records with no previous contact and an unknown outcome, but also five records with previous contact and an unknown outcome. We retained those five records and their row identifiers. The public data cannot tell us whether their outcomes were unavailable, unrecorded, or unknown for another reason.

This supports separating contact status from elapsed days, while keeping poutcome outside our job-and-education imputation experiment. It also suggests why the new contact flag might add limited predictive information: other historical variables already contain overlapping information. That is a possible explanation, not proof of redundancy.

Our second finding concerns information completeness. Overall, records with unknown job or education had a subscription rate of 13.52 percent, compared with 11.61 percent when both fields were known. But this association was not uniform across age groups. In the group aged sixty or above, the rates were 28.87 percent for unknown records and 33.67 percent for known records. The unknown subgroup had only 97 observations, so we flag it as a small sample.

We do not call this Simpson's paradox or a causal effect. Customer composition, recording practices, and sampling variation are alternative explanations. We therefore compare retaining unknown categories with training-only mode imputation, and ask D to report both overall and subgroup errors.

Our contribution is the evidence linking data semantics to preparation decisions. Clearer representation is valuable, but predictive improvement must be measured rather than assumed.

## 中文理解与应答

- 主发现不是复述“-1 表示未联系”，而是实算三个字段的对应关系与 5 条例外，解释为何不能统一处理 unknown。
- previous 的零/正状态与联系标记对应，不等于作为连续输入的 previous 与新二元特征在线性模型里完全等价。
- 分层方向不一致不自动构成辛普森悖论；未知组不必然随机，但不能判定 MAR/MNAR。
- 97 是 ≥60 岁 unknown 训练组的分母，28 是其正例数；图表需要同时给出，不能只报百分比。
- 没有依据删除负余额或极端 previous；保留数据并记录尾部限制，四组比较不夹带额外处理。
- 模型结果小、混合或变差也要保留。测试与验证的拟合规模不同，不比较两阶段分差来宣称 S/U 效果。

## 全组建议页序（最多 8 页）

| 页 | 主讲人 | 内容 | 图/证据 | 建议时间 |
| --- | --- | --- | --- | --- |
| 1 | A | 问题、数据来源与联系前场景 | 数据来源与白名单 | 00:00–01:10 |
| 2 | A | 质量治理、固定 split、泄漏边界 | 数据审计与 split | 01:10–02:30 |
| 3 | B | 历史字段的重叠与 5 条例外 | `02_history.png` | 02:30–03:50 |
| 4 | B | unknown 总体与年龄差异 | `03_unknown_age.png` | 03:50–05:10 |
| 5 | C | S/U 转换、四方案、训练拟合边界 | 四方案矩阵与边界示例 | 05:10–07:50 |
| 6 | D | 四方案测试与五项增量 | 当前 `outputs/D/test/` | 07:50–09:15 |
| 7 | D | 分组代价与指标限制 | `04_model_tradeoffs.png` | 09:15–10:35 |
| 8 | 全员 | 各自贡献、反思与限制 | 四句个人总结 | 10:35–11:20 |

四人均须出镜并实际讲解。视频总时长不超过12分钟。`01_overview.png` 可作备用证据，不必为了凑图增加页数。公开视频、组号及课程提交仍由成员完成。
