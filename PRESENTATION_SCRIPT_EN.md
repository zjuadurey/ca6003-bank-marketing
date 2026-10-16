# Four-person English presentation script

用途：四人连续出镜视频，课程上限 **12:00**。本稿四段正文共约 **1248 words**（程序按单词边界计数；数字读法会略增时长）。按每分钟120词正文约 10.4 分钟，另留45秒停顿/换页，预计 11.2 分钟；必须实际计时彩排，不能保证所有语速都低于12分钟。建议目标11:15左右，硬截止12:00。

数据版本：本次冻结后的 `outputs/D/test/`；B 段只引用训练集。请先理解内容并复核数字，再录制；本稿不代表已经人工审核。

| Speaker | Words | Suggested budget | Visual |
| --- | --- | --- | --- |
| A | 296 | 00:00–02:45 | Problem, provenance, split and whitelist |
| B | 308 | 02:45–05:30 | B history cross-tab and unknown-by-age chart |
| C | 283 | 05:30–08:15 | S/U definitions and four-experiment matrix |
| D | 361 | 08:15–11:30 | Final metrics, all contrasts and subgroup tradeoffs |

下方只有英文正文需要朗读；标题和中文提示不读。D 可将三个年龄组标签读作“under thirty”, “thirty to fifty-nine”, “sixty and above”。若超时，先精简衔接或指标定义，不删除负面结果和限制。

## A

Hello everyone. Our project asks how the meaning of special values and the treatment of unknown information affect the reliability of bank marketing predictions.

We use the legacy UCI Bank Marketing dataset, bank-full.csv. It contains 45,211 records, sixteen original input fields, and a binary target indicating whether the customer subscribed to a term deposit. Our objective is to investigate data preparation, not to compete for the highest accuracy.

We frame the task as an offline study of contact-before-call screening. We retain eleven customer and historical features. We exclude duration because the completed call length would not be available before the call. We also conservatively exclude campaign, contact, day, and month because they describe the current campaign. This is our study design, not a claim that every excluded variable is always unusable.

For governance, we preserve the original data bytes, record their source, license, and hash, and keep a field dictionary. A row identifier tracks records but is never a predictor. It is not a customer identifier.

We use one fixed, stratified sixty-twenty-twenty split, with random state forty-two. Open-ended exploration uses only the training partition. Learned preprocessing is fitted only on the current training data. Validation supports the predefined comparison, and the final test follows protocol freezing.

Our initial audit distinguishes actual missing cells from literal unknown categories and the special value minus one in pdays. We do not automatically delete negative balances, long tails, or apparently repeated feature records.

There are important boundaries. We lack unique customer identifiers and complete feature timestamps. Therefore, random record splitting cannot establish customer independence or future performance. Predicting subscription also does not estimate the causal benefit of making a call.

My contribution is this reproducible data foundation. B will now explain the evidence that motivates our preparation experiments.

## B

My investigation uses the fixed training partition: 27,126 records, including 3,173 subscriptions. The main finding is that an unknown value cannot be interpreted correctly by inspecting one field alone.

We cross-tabulated pdays, previous, and poutcome. Every training record with nonnegative pdays had a positive previous-contact count. Every record with pdays equal to minus one had zero previous contacts. These fields therefore contain overlapping information about contact history.

However, an unknown previous outcome did not always mean no previous contact. We found 22,235 records with no previous contact and an unknown outcome, but also five records with previous contact and an unknown outcome. Their row identifiers are preserved for inspection. The public fields cannot tell us why those five outcomes are unknown, so we retain the records rather than inventing a correction.

This supports distinguishing contact status from elapsed days, and treating poutcome differently from unknown job or education. The overlap also suggests why explicitly adding a contact flag might produce only a small predictive change. It does not prove mathematical redundancy in logistic regression.

Our second finding is that information completeness has different associations across age groups. Overall, records with unknown job or education had a subscription rate of 13.52 percent, compared with 11.61 percent when both were known. Among customers aged sixty or above, the direction differed: 28.87 percent versus 33.67 percent. The unknown subgroup contained only 97 observations, so we flag it as small.

This is not sufficient evidence of Simpson's paradox, causality, or a particular missing-data mechanism. Customer composition, recording practices, and random variation are alternative explanations. We also inspected balance and previous-contact tails without deleting valid-looking extremes on assumption alone.

My contribution is the link from observed data to explicit preparation choices. We now test those choices rather than assuming that more preparation must improve every result.

## C

I implemented two preparation switches and kept all other experimental conditions fixed.

The first switch, S, changes the representation of pdays. When S is off, the original value, including minus one, enters as a numeric feature. When S is on, we replace it with a previous-contact indicator and a conditional interval. A never-contacted record receives indicator zero and interval zero. A record contacted zero days earlier receives indicator one and interval zero. The indicator is essential: the placeholder zero does not mean that a never-contacted customer was contacted today.

The second switch, U, affects only job and education. When U is off, their literal unknown values are replaced using modes learned from the fitting partition. When U is on, unknown remains a separate category for one-hot encoding. We do not apply this switch to poutcome.

These switches produce four experiments: E-zero, with both off; E-S, with only S on; E-U, with only U on; and E-S-U, with both on.

All four use the same raw records, eleven-feature policy, logistic regression settings, and probability threshold of zero point five. Numeric features are standardized, while the binary history flag remains zero or one. One-hot encoding handles unseen inference categories without learning from validation or test data.

The complete pipeline learns imputation, scaling, and encoding only during fit. This prevents the common error of preparing the full dataset before splitting it. Boundary checks cover minus one, zero, positive intervals, forbidden fields, unexpected missing values, and row alignment.

My contribution is the shared, switchable preparation implementation. The two switches can change representation, feature count, and regularization behavior, so the score differences are not isolated causal effects. D will report all four configurations, including deterioration and unchanged results.

## D

I evaluated the same basic model under all four preparation configurations. We froze the protocol and then refitted each configuration on the combined training and validation partitions, totaling 36,168 records. The final test contains 9,043 records. We did not tune thresholds or select new features after seeing the test results.

We report average precision for ranking, recall and F-one at the fixed threshold, and Brier score for probability error. Accuracy is background information. Brier and false-positive rate are better when lower; the other main metrics are better when higher.

On the final test, average precision was 0.3472 for the baseline and 0.3469 for full preparation. The full-minus-baseline change was -0.00026. Recall decreased by 0.09 percentage points, and Brier score increased by 0.000004. These are descriptive changes, not claims of statistical significance.

To separate the two factors, our tables include both S comparisons, both U comparisons, and the complete preparation comparison. We preserve every result rather than reporting only the most favorable configuration.

Here, S slightly improved average precision and Brier under both U settings. U worsened both metrics under both S settings. Preserving unknown categories was worth investigating, but did not deliver a predictive gain in this particular comparison.

We also report subgroup sample sizes, positive and negative counts, recall, false-positive rate, and Brier score. For full preparation versus baseline, age-group recall changes are: <30: +0.00 percentage points; 30-59: +0.00 percentage points; >=60: -0.89 percentage points. Information groups use the original job and education values, before imputation. Small groups are explicitly flagged, and group differences alone do not establish discrimination.

Baseline test accuracy was 89.34 percent, while predicting no for everyone achieved 88.30 percent. Baseline recall was only 16.73 percent. This illustrates why accuracy alone would give an incomplete picture of subscription identification.

Our conclusion is that semantic clarity and predictive improvement are separate questions. Historical information overlaps, unknown groups are heterogeneous, and preparation must be evaluated with explicit tradeoffs. We acknowledge Codex assistance and retain reproducible evidence for personal verification. Together, A provided governance, B the investigation, C the preparation pipeline, and D the evaluation. Thank you.
