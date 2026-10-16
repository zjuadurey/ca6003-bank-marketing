# LIANG SHUANG · D4 对照评估交付与交接

> **历史交接文档。** 本文记录原 D 包验证阶段的交接流程。后续用户已授权完成 B、冻结和最终测试，实际记录见 `../outputs/D/freeze.json` 与 `test_completed.json`。当前入口为 [PROGRESS](../PROGRESS.md) 和 [最终结果](RESULTS.md)。人工复核仍 pending；不要按本文旧流程重复执行已完成的测试。

依据：本地 `Bank_Marketing_Group_Execution_Manual_Condensed.pdf` 第 6-10 页；A 仓库基线提交 `b4c4ff1be75a6e48baacd920f405982b20399352`。更新：2026-10-16。

## 当前交付状态（2026-10-16）

C正式管线`src/preprocessing.py`已收到并接入，原始管线源码字节未修改。D已用C正式管线重跑四组验证、全部增量、分组误差、逐行概率、特征名、系数、图表和Notebook。A10项+C14项+D12项技术检查均通过。B发现尚未收到，实际人工复核和最终测试待完成。

旧D结果和报告已归档至`outputs/D/history/reference_20261015/`。C原包及原机证据在`outputs/C/received/`，本机复跑在`outputs/C/`。本次统一沿用D环境（Python3.12.14、numpy2.5.4、pandas3.0.6、sklearn1.8.0），未混用C原机分数。

独立核对：规范特征名对齐后，C与旧D输入矩阵差异为0，概率最大差异<1e-12，阈值分类变化0条。C类别块顺序不同，系数已按C特征名重新生成，不能按旧表位置拼接。证据见`outputs/C/migration_comparison.csv`。

当前只用train/val计算模型结果；B仍需交核心发现并确认分组和解释，之后冻结协议并一次性测试。默认分组来自手册，在验证运行前已固定，未根据验证结果重分组。测试代码已备好，不填造测试分数、不代填人工确认。

## 向 B 获取什么

| 材料 | 必需内容与格式 | D 如何使用 / 验收 |
| --- | --- | --- |
| 核心发现及证据 | 仅 train 的可复现代码/Notebook；至少一张 CSV 交叉表；列明分区、n、正例数、分母、分组；最多 4 张主图 | 将模型变化与观察对应；不能拿 A 的审计计数当作 B 已完成的独立发现 |
| 历史联系交叉表 | `pdays=-1 / >=0 × previous=0 / >0 × poutcome`，异常组合的原始 row_id、解释及未知原因 | 核对 S 的必要性及已有 previous/poutcome 可能带来的冗余；不据此声称因果 |
| unknown 的分层证据 | 按手册年龄组或联系历史分层的 job/education unknown 比例、认购率与分母；分层前后比较、替代解释 | 解释 U 与组内误差；没有关系反转证据时不写“辛普森悖论” |
| 分组冻结确认 | 年龄 `<30 / 30-59 / >=60`；原始 job 或 education 任一 unknown vs 两列已知；若需修改须在测试前提出 | 必须用原始值定义信息完整性组，不能用填补后的值；当前程序只接受手册分组 |
| 解释复核 | 对 D 的全部结果，特别是变差或无变化，提供真实姓名、日期和书面意见 | 形成最终讨论与视频讲稿；小样本规则为 n<100 或正/负例<20 |

建议 B 交付：`B_discovery.ipynb`、`B_history_crosstab.csv`、`B_unknown_strata.csv`、`B_findings.md`、`B_groups_review.md`。文件名是建议，内容尚未收到。请索取代码和原始表，不能只要截图或口头结论。

## C已经收到什么、还需什么

| 材料 | 本次状态 |
|---|---|
| 正式build_pipeline和四组配置 | 已收到、接入；配置指向src.preprocessing，四组验证重跑 |
| S/U转换、防泄漏、未见类别、白名单、序列化测试 | C14项和适配后的D12项本机通过 |
| 环境锁、特征名、拟合统计和原机运行证据 | 已收到；本机证据重新生成，原机材料单独保留 |
| C Notebook | 已收到，将在本机新内核运行；完整评价以D章节为准 |
| 整合Notebook | 原ZIP未含说明中的analysis.ipynb；本次补A+C+D草稿，B明确留空 |
| C姓名和实际理解/复核 | 原包为pending，需C本人阅读并确认；不需重复索取已有代码 |

原包提及的build_c_delivery.py、execute_c_notebooks.py也未提供，但当前D不依赖它们。本次已补整合与复现脚本。若C有更新全项目，可再提供对照；C最后仍负责核对单一Notebook的装配和贡献。完整清单见`D_MISSING_ITEMS.md`。

## D 文件地图与复现

- `notebooks/D_evaluation.ipynb`：LIANG SHUANG 的独立 D 章节，包含源代码、实际验证输出、指标定义、结果解释与后续测试入口；并非全组最终 Notebook。
- `src/preprocessing.py`：当前使用的C正式管线；`src/d_reference_pipeline.py`仅作历史迁移核对。
- `analysis.ipynb`：A+C+D整合草稿，B留空，不是完整课程成品。
- `outputs/C/local_integration.json`：原包SHA-256、来源文件和接入记录。
- `src/d_evaluation.py`：统一指标、原始分组、五项对比。
- `scripts/run_d.py`：validate / freeze / test；验证拟合 27,126 行，评价 9,042 行；最终程序拟合 train+val 36,168 行，评价 test 9,043 行。
- `outputs/D/validation/`：metrics、references、contrasts、subgroups、subgroup_contrasts、predictions、coefficients、run_manifest。
- `outputs/D/qa.json`、`outputs/D/notebook_validation.json`：自动化检查；不能替代真实人工复核。
- `requirements.D.lock.txt`：本次 D 实测环境锁；A 的原锁保留。Python 3.12；具体补丁版本见运行 manifest。

PowerShell，在仓库根目录使用已准备好的环境：

```powershell
.\.venv\Scripts\python.exe scripts/run_d.py validate
.\.venv\Scripts\python.exe C_preprocessing_tests.py
.\.venv\Scripts\python.exe scripts/validate_d.py
.\.venv\Scripts\python.exe scripts/validate_c_integration.py
.\.venv\Scripts\python.exe scripts/execute_d_notebook.py
.\.venv\Scripts\python.exe scripts/build_acd_draft.py
.\.venv\Scripts\python.exe scripts/execute_acd_notebooks.py
```

新机器先 `py -3.12 -m venv .venv`，再 `.\.venv\Scripts\python.exe -m pip install -r requirements.D.lock.txt`。所有路径相对项目；不需要复制当前机器的绝对路径。

## 收齐 B/C 后的最后步骤

1. C正式代码的技术接入已完成。收B核心发现和分层表，核对分组与解释，成员实际理解并确认。如果改任何规则，重跑验证并记录依据。禁止先看测试分数再修改策略。
2. 将模板 `config/d_handoff_receipt.template.json` 复制成 `config/d_handoff_receipt.json`，填入真实 B/C/D 复核姓名、日期、两个确认项和实际证据文件相对路径。不要把 pending 自动改成人名。
3. 执行下列命令。freeze 保存数据/划分/代码/环境/验证输出/交接证据的 SHA-256；test 核查未改变后，用 train+val 重拟合全部四组。独占标记阻止重复运行测试。若中途故障，保留标记和故障原因供全组审查，不能自动删标记重试。

```powershell
.\.venv\Scripts\python.exe scripts/run_d.py freeze --receipt config/d_handoff_receipt.json
.\.venv\Scripts\python.exe scripts/run_d.py test
```

4. 结果在 `outputs/D/test/`，结构与 validation 相同。把测试表与分组差值填入最终 Notebook，重新写最终结论并与 B 复核。验证分数与测试分数对应不同训练样本规模，不能把两者差值解释成 S/U 的效果。
5. C 完成单一 Notebook；A 新内核运行；D 核视频、图、表、讲稿中的每个数值。四人出镜录制并完成实际提交，不能由自动化代替。

## 指标、解释及 GenAI 声明

正类 yes=1；预测 p>=0.5 为 yes。AP 使用 average_precision_score（不是梯形 PR-AUC）；Recall=TP/(TP+FN)，FPR=FP/(FP+TN)，F1=2TP/(2TP+FP+FN)，Brier=mean((p-y)^2)，Accuracy=(TP+TN)/n。分母为零则 NaN/无定义；无正例时 AP 也为 NaN。Recall/FPR/Accuracy 差值乘100才是百分点，Brier/AP 用原始分数差；Brier/FPR 越小越好。未计算置信区间，不使用“显著改善”。

全部预测 no 为阈值识别背景参考；额外保留仅用当前训练正例率的恒定概率参考。概率质量与排序能力分别解释；固定阈值下 Recall 很低时，高 Accuracy 不能代表营销筛选有效。

S 同时改变表示、维数及正则化几何；previous/poutcome 可能冗余。单个 OHE 系数受相关特征、完整独热与正则化影响，不能作为因果效应或跨管线直接等价解释。Brier 同时受校准与区分度影响，不能单凭 Brier 称为校准检验。分组误差不等于歧视判定；没有客户 ID/完整时间戳，不证明客户独立性、未来泛化或收益。未调阈值、未加入复杂模型。

本包由 OpenAI Codex 辅助编写代码、运行实验、检查指标和组织解释。LIANG SHUANG 是用户声明的负责人 D；本包没有声称其已经完成代码人工审阅、录制或签字。B材料尚缺、C技术交付已收到、人工核验待完成的状态如实保留；最终应由成员阅读代码、复算指标、理解结论并补上真实复核记录。

参考：手册第 6-10 页；A 的 DATA_CONTRACT.md / HANDOFF.md；[AP 官方文档](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html)、[Brier 官方文档](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.brier_score_loss.html)、[拟合泄漏官方说明](https://scikit-learn.org/1.8/common_pitfalls.html)。
