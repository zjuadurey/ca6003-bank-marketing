# CA6003 小组进度与交接

课程最新原文确认只提交 `.ipynb`：正式交付已恢复自包含版本，不需要附上 `BankMarketing_Data.zip`。该 ZIP 仅留作本地拆分版资料。用户已确认第 17 组，最终文件名为 `Assignment_Grp_17.ipynb`，位于上级“最终版”目录；其内容与仓库 `analysis.ipynb` 一致。

2026-10-18 更新：成员已确认：A **Yu Qiushi**、B **Chen Junyu**、C **Sun Shaojie**、D **Liang Shuang**。姓名确认不代表人工复核完成。

当前提交入口为全英文、单文件自包含的 `analysis.ipynb`：内嵌原始数据、固定划分和冻结评价证据，核心代码直接展示。使用指定 Python 环境，无须克隆仓库即可 Run All。独立空目录新内核运行结果见 `outputs/integration/english_submission_validation.json`。只重拟合四个验证模型；最终测试核算保存预测，不重训或修改冻结证据。生成与验证入口分别为 `scripts/build_english_submission.py`、`scripts/validate_english_submission.py`。

更新：2026-10-16。本文件是当前工作入口；历史 A/C/D 交付中的“B 未收到”“测试未执行”等描述是当时状态。**A+B+C+D 已整合，B 证据、四组验证与一次性最终测试已实际完成；B 和最终 Notebook 新内核执行通过。成员人工复核、视频录制和课程提交仍待完成。**

| 成员 | 已有成果 | 当前任务 | 下一步 / 所需信息 |
| --- | --- | --- | --- |
| A 数据治理 | 原始数据、来源许可、字段政策、固定 split、审计保留；本机 10 项检查通过 | 计算侧完成 | 本人复核字段政策；确认组号、提交人和截止 |
| B 数据洞察（Chen Junyu） | 训练集交叉表、5 条例外、unknown 总体/年龄/历史分层、年龄标准化、数值尾部、4 张主图、独立计数校验、已执行 B Notebook、讲稿 | 计算与材料完成 | 实际理解/复核待补；不代签 |
| C 准备实现（Sun Shaojie） | 正式源码与 C ZIP 字节一致；本机 14 项测试通过；总 Notebook 内嵌核心代码 | 原 C 实现已接入，本次 Codex 按用户要求完成装配 | 本人复核实现和贡献归属 |
| D 对照评估 | 本机 12 项检查通过；四组验证复现；冻结后四组最终测试完成；哈希与 sklearn 独立指标复算通过 | 计算侧完成；用户执行授权与人工复核状态分开记录 | D 本人核对当前最终数字、解释和视频 |

## 已执行与证据入口

1. A 仓库已克隆；C/D 导入已完成，来源清单在 `outputs/integration/CD_import.json`。
2. 项目独立 `.venv` 已安装。实际 Python 3.12.12；全部锁定依赖及 `pip check` 通过，见 `outputs/integration/environment.json` 和 `checks.json`。其他项目环境未修改。
3. B 证据见 `notebooks/B_discovery.ipynb`、`notes/B_findings.md`、`outputs/B/`；独立原始 CSV 计数和训练行序核对通过。A10+C14+D12+执行授权2项测试通过，日志在 `outputs/integration/`。
4. 冻结记录 `outputs/D/freeze.json`；实际测试完成记录 `outputs/D/test_completed.json`；全部四方案及5项增量在 `outputs/D/test/`。不删标记、不重跑测试选策略。原导入结果已保留于 `outputs/D/history/imported_C_20261016/`。
5. 最终 Notebook：`analysis.ipynb`；新内核记录 `outputs/integration/notebook_validation.json`。B执行4个代码单元，总Notebook执行20个代码单元；测试回放只核算保存预测，不重训测试模型。当前解释见 `notes/RESULTS.md`。
6. 全组四段英文稿：`PRESENTATION_SCRIPT_EN.md`，附单词计数和时长预算；B补充理解提纲见 `notes/B_VIDEO_SCRIPT.md`。时间是估算，尚未实际彩排。

## 关键结果（不要使用历史验证稿冒充测试）

最终测试 E0 / ES / EU / ESU 的 AP 分别约 0.347165 / 0.347626 / 0.346853 / 0.346903。ESU−E0 的 AP 为 −0.000262，Recall 为 −0.0945 个百分点，Brier 为 +0.00000422。S 有微小收益；U 在本次两种 S 条件下均使 AP、Recall 和 Brier 变差。完整准备未改善所有指标，语义清晰不等于性能保证。≥60 岁及原始 unknown 组各少识别一个正例；分组重叠，不能相加成两个人。无显著性或因果结论。

## 尚需真实人员完成的信息和动作

- 各自实际复核记录；组号 17 和四位姓名已经用户确认，不表示已经审阅本次修改。
- 四人实际出镜录制单个连续视频（不超过 12 分钟），公开链接和实际上传凭证。
- NTULearn 上确认截止与提交状态；课程文件写 10 月 18 日 12:00 前交 Notebook + 视频，10 月 21 日 12:00 前交互评，时区以平台为准。
- GenAI 使用需声明，成员须能解释代码、证据、处理理由和结论。自动化通过不能替代此责任。

## 外部状态

用户已授权完成后提交并推送。原仓库对当前账号 `zjuadurey` 返回 403；用户随后授权 fork，已创建 `https://github.com/zjuadurey/ca6003-bank-marketing`。本地 `origin` 指向用户 fork，`upstream` 保留 `raederhans/ca6003-bank-marketing`。本次交付推送目标为用户 fork 的 main；原仓库未修改。未联系队友、未发布视频、未提交课程作业；这些外部动作未获授权。
