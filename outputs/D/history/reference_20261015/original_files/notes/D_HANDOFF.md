# LIANG SHUANG · D4 对照评估交付与交接

依据：本地 `Bank_Marketing_Group_Execution_Manual_Condensed.pdf` 第 6-10 页；A 仓库基线提交 `b4c4ff1be75a6e48baacd920f405982b20399352`。日期：2026-10-15。

## 当前交付状态

D 已完成统一评估器、四组验证实验、五项增量、分组误差、逐行概率、模型系数、可执行 Notebook、边界与泄漏检查，以及封版后的最终测试程序。当前结果来自 D 按手册补写的参考管线，**是验证集结果，等待 B/C 人工交接核验，不是全组最终结论**。不能将参考实现计入 C 的已完成贡献。

手册第 7 页要求先接收 B 的预先分组及 C 的统一 API，再冻结规则并一次性测试。由于 B/C 内容尚未提供，当前不打开测试集进行预测或指标分析，不填造测试分数，也不代签人工复核。测试代码已经写好。默认分组来自手册，在验证运行前写入配置，未根据验证结果重新分组。

## 向 B 获取什么

| 材料 | 必需内容与格式 | D 如何使用 / 验收 |
| --- | --- | --- |
| 核心发现及证据 | 仅 train 的可复现代码/Notebook；至少一张 CSV 交叉表；列明分区、n、正例数、分母、分组；最多 4 张主图 | 将模型变化与观察对应；不能拿 A 的审计计数当作 B 已完成的独立发现 |
| 历史联系交叉表 | `pdays=-1 / >=0 × previous=0 / >0 × poutcome`，异常组合的原始 row_id、解释及未知原因 | 核对 S 的必要性及已有 previous/poutcome 可能带来的冗余；不据此声称因果 |
| unknown 的分层证据 | 按手册年龄组或联系历史分层的 job/education unknown 比例、认购率与分母；分层前后比较、替代解释 | 解释 U 与组内误差；没有关系反转证据时不写“辛普森悖论” |
| 分组冻结确认 | 年龄 `<30 / 30-59 / >=60`；原始 job 或 education 任一 unknown vs 两列已知；若需修改须在测试前提出 | 必须用原始值定义信息完整性组，不能用填补后的值；当前程序只接受手册分组 |
| 解释复核 | 对 D 的全部结果，特别是变差或无变化，提供真实姓名、日期和书面意见 | 形成最终讨论与视频讲稿；小样本规则为 n<100 或正/负例<20 |

建议 B 交付：`B_discovery.ipynb`、`B_history_crosstab.csv`、`B_unknown_strata.csv`、`B_findings.md`、`B_groups_review.md`。文件名是建议，内容尚未收到。请索取代码和原始表，不能只要截图或口头结论。

## 向 C 获取什么

| 材料 | 必需内容与格式 | D 如何使用 / 验收 |
| --- | --- | --- |
| 正式管线 | 可导入模块的 `build_pipeline(use_s: bool, use_u: bool)`，输出完整 sklearn Pipeline，支持 fit / predict_proba / get_feature_names_out | 替换 `config/d_protocol.json` 中 pipeline_module，然后重跑所有验证；如 C 直接采纳 D 参考管线，须记录真实审阅与测试依据 |
| 四配置与参数 | E0=(off,off)、ES=(on,off)、EU=(off,on)、ESU=(on,on)；同一白名单、模型、阈值、行顺序 | 不得用个人 split、另一个数据版本、class_weight 或搜索最优参数 |
| S 边界测试 | -1→状态0/占位0；0→状态1/间隔0；正天数→状态1/原值；原 pdays 移除；<-1 或真缺失停止核查 | 无联系状态不能解释成当天联系；共同数值 StandardScaler，状态标记原样 0/1 |
| U 与拟合边界测试 | U=off 只把 job/education 的 unknown 转 NaN 并在 fit 数据上众数填补；U=on 保留；poutcome 不改；未见类别允许推断 | 核对训练众数/中位数/缩放/OHE 只从当前 fit 分区获得；验证和测试只 transform |
| 测试证据与运行环境 | 测试代码、通过日志、版本锁、提交号/源文件、特征名与编码后维数 | D 重新验证收敛、概率合法、row_id 对齐；不能只交已经在全表 fit 的矩阵 |
| 最终 Notebook 装配 | 合并 A/B/C/D 核心函数，保留输出、GenAI 声明、引用、贡献和待办状态 | `D_evaluation.ipynb` 已内嵌 D 关键源代码，便于 C 按章节合并；C 装配后由 A 新内核 Run All |

建议 C 交付：`src/preprocessing.py`、`C_preprocessing_tests.py`、测试报告、环境锁和整合 Notebook。若 C 采用不同状态标记缩放约定，必须先全组确定、记录理由并重跑验证，不能混用为同一实验结果。

## D 文件地图与复现

- `notebooks/D_evaluation.ipynb`：LIANG SHUANG 的独立 D 章节，包含源代码、实际验证输出、指标定义、结果解释与后续测试入口；并非全组最终 Notebook。
- `src/d_reference_pipeline.py`：D 临时补齐的 C 接口，贡献与待审状态明确。
- `src/d_evaluation.py`：统一指标、原始分组、五项对比。
- `scripts/run_d.py`：validate / freeze / test；验证拟合 27,126 行，评价 9,042 行；最终程序拟合 train+val 36,168 行，评价 test 9,043 行。
- `outputs/D/validation/`：metrics、references、contrasts、subgroups、subgroup_contrasts、predictions、coefficients、run_manifest。
- `outputs/D/qa.json`、`outputs/D/notebook_validation.json`：自动化检查；不能替代真实人工复核。
- `requirements.D.lock.txt`：本次 D 实测环境锁；A 的原锁保留。Python 3.12；具体补丁版本见运行 manifest。

PowerShell，在仓库根目录使用已准备好的环境：

```powershell
.\.venv\Scripts\python.exe scripts/run_d.py validate
.\.venv\Scripts\python.exe scripts/validate_d.py
.\.venv\Scripts\python.exe scripts/execute_d_notebook.py
```

新机器先 `py -3.12 -m venv .venv`，再 `.\.venv\Scripts\python.exe -m pip install -r requirements.D.lock.txt`。所有路径相对项目；不需要复制当前机器的绝对路径。

## 收齐 B/C 后的最后步骤

1. D/C 核对正式管线及测试；B 核对分组与解释。必要时替换 pipeline_module 并重跑四组验证及 QA；保留变更依据。禁止先看测试分数再修改策略。
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

本包由 OpenAI Codex 辅助编写代码、运行实验、检查指标和组织解释。LIANG SHUANG 是用户声明的负责人 D；本包没有声称其已经完成代码人工审阅、录制或签字。B/C 材料缺失、模型依赖与人工核验状态如实保留；最终应由成员阅读代码、复算指标、理解结论并补上真实复核记录。

参考：手册第 6-10 页；A 的 DATA_CONTRACT.md / HANDOFF.md；[AP 官方文档](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html)、[Brier 官方文档](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.brier_score_loss.html)、[拟合泄漏官方说明](https://scikit-learn.org/1.8/common_pitfalls.html)。
