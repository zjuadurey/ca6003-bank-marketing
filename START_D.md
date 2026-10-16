# LIANG SHUANG · D / 2026-10-16更新

> **历史交付入口。** 以下为原 D 包验证阶段状态。当前 B 和最终测试已完成；请先读 [PROGRESS.md](PROGRESS.md)、[最终结果](notes/RESULTS.md) 和 [总 Notebook](analysis.ipynb)。原始版本已归档，不能使用下方旧状态描述当前完成度。

**C正式管线已接入，D验证实验与报告已更新。B发现、成员实际复核和最终测试待完成。**

1. output/pdf/LIANG_SHUANG_D_Evaluation_Report.pdf：6页中文报告，含结果、客群代价、C验收和剩余缺口。
2. analysis.ipynb：A+C+D整合草稿，B留空，尚不可作为完整课程作业提交。
3. notebooks/D_evaluation.ipynb：D独立章节，内嵌C正式管线与D评价函数。
4. notes/D_MISSING_ITEMS.md：现在向B索取什么、C需确认什么、全组还缺什么。
5. notes/D_HANDOFF.md：完整交接、复现及最后freeze/test命令。
6. notes/D_VIDEO_SCRIPT.md：验证版英文讲稿和中文提纲。

当前使用src/preprocessing.py。旧D结果在outputs/D/history/reference_20261015/，原C证据在outputs/C/received/，本机结果在outputs/D/validation/与outputs/C/。代码、结果和文档的贡献归属分开记录。

本次环境用requirements.D.lock.txt；C原锁保留作来源，不同时安装两份锁。A10+C14+D12项检查通过。特征顺序改变，系数已重新生成；规范名对齐后矩阵相同，概率差异<1e-12，分类不变。

最终测试尚未运行。收到B发现并完成实际复核后按D_HANDOFF冻结，重拟合train+val并测试四组。不要把验证结果填成测试结果。未推送GitHub。
