# LIANG SHUANG · 负责人 D：从这里开始

**现在的状态：D 评估代码与四组验证实验完成；B/C 交接与最终测试待完成。**

1. 先读 `output/pdf/LIANG_SHUANG_D_Evaluation_Report.pdf`：6 页中文报告，含当前结果、分组代价、向 B/C 索取的材料和交接请求草稿。
2. 打开 `notebooks/D_evaluation.ipynb`：已在新内核运行，关键函数内嵌，给 C 合并 D 章节。
3. 看 `notes/D_HANDOFF.md`：完整材料清单、验收标准、文件地图、安装和复现命令。
4. 看 `notes/D_VIDEO_SCRIPT.md`：D 的约 2 分 45 秒英文讲稿初稿，附中文理解提纲；目前是验证版。
5. 数值原表在 `outputs/D/validation/`；全部五项差值、五个原始分组、逐行预测、系数和运行环境均保留。

当前观察：S 的排序改善很小，U 效果混合。完整方案并未改善所有指标；≥60 岁组少识别一个正例，unknown 组 Brier 略差。阈值0.5下 Recall 仅约17%，不能只展示约89.4%的 Accuracy。

不要把当前验证结果填入测试列。D 临时管线依据手册实现，必须请 C 正式审阅采纳或替换，不能声称 C 已完成。B 需交训练集发现、交叉/分层表和解释复核。收齐后按 D_HANDOFF.md 冻结，最终程序会用 train+val 重拟合全部四组并一次性测试。

自动化检查：A 的10项治理检查通过，D 的12项评估检查通过；D Notebook 新内核9个代码单元执行成功。D 最终测试流程的程序检查使用训练集切片模拟，未使用真实测试集算分。人工复核、全组 Notebook、四人视频与实际课程提交仍由成员完成。

代码与全部结果均在当前本地包内，未推送到 GitHub。原始课程手册不包含在可共享 ZIP 中。A 的 release_manifest.json 对应原 A 基线；D 完整包使用 outputs/D/delivery_manifest.json 校验。
