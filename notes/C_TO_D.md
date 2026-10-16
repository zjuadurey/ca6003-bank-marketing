# C → D：正式管线与验收交接

日期：2026-10-15。C 真实姓名及人工复核：pending。D：LIANG SHUANG（沿用 D 包声明）。

## 本次交付

C 按执行手册第 6 页独立实现 `src/preprocessing.py`，不是把 D 的临时参考实现改名计为 C 贡献。D 的统一评价代码保留作者归属；原始 `../D/` 文件夹未改动。

| 内容 | 文件 | 状态 |
|---|---|---|
| 正式完整管线 | src/preprocessing.py | E0/ES/EU/ESU 可 fit / predict_proba / get_feature_names_out |
| C 转换与防泄漏测试 | C_preprocessing_tests.py；outputs/C/tests.json、tests.log | 14 项通过，未读取真实 test 分区 |
| 四组运行证据 | outputs/C/pipeline_checks.csv、feature_names.csv、fit_statistics.json | 训练 27,126；验证 9,042；编码维数 31/32/33/34；均 25 次迭代，无警告 |
| 环境与来源 | requirements.C.lock.txt；outputs/C/environment.json、d_integration.json | Python 3.12.14；sklearn 1.8.0；没有 Git 元数据，用源文件 SHA-256 追踪 |
| D 代码对接 | config/d_protocol.json；scripts/run_d.py；src/d_evaluation.py | pipeline_module 已设为 src.preprocessing；已重跑四组验证 |
| D 验收 | scripts/validate_d.py；outputs/D/qa.json | 已适配 C 的类名和步骤名；12 项通过 |
| Notebook | notebooks/C_preprocessing.ipynb；analysis.ipynb | 独立 C 章节及 A+C+D 整合草稿；B/最终测试待补 |

## D 的调用方式

```python
from src.data_contract import load_partition
from src.preprocessing import build_pipeline

X_train, y_train = load_partition('train')
X_val, y_val = load_partition('val')
pipe = build_pipeline(use_s=True, use_u=True)
pipe.fit(X_train, y_train)
probability = pipe.predict_proba(X_val)[:, 1]
feature_names = pipe.get_feature_names_out()
# D 原有 pipe[:-1].get_feature_names_out() 同样支持。
# 写预测表时保留 X_val.index（row_id）顺序；正类 yes=1，p>=0.5。
```

本仓库已接好 D 的协议，只需从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe C_preprocessing_tests.py
.\.venv\Scripts\python.exe scripts/run_d.py validate
.\.venv\Scripts\python.exe scripts/validate_d.py
.\.venv\Scripts\python.exe scripts/build_c_delivery.py
.\.venv\Scripts\python.exe scripts/execute_c_notebooks.py
```

如 D 合并到自己的仓库，先复制 C 模块和测试、必要的说明与证据，再将其 d_protocol.json 的 pipeline_module 改为 `src.preprocessing`，在自己的环境重跑验证。不要覆盖 D 原有结果后仍引用旧报告/旧图中的数字。

## 必须保持的约定

- E0=(False,False)，ES=(True,False)，EU=(False,True)，ESU=(True,True)。11 列白名单及 A 的原始行顺序不变，不重新拆分。
- S：-1→(状态0,间隔0)，0→(状态1,间隔0)，正数→(状态1,原值)；原 pdays 移除；状态不缩放，其他数值缩放。
- U：仅 job/education 的 unknown 转 NaN 并用 fit 数据众数填补，或保留独立类别。poutcome 不改。推断阶段未见类别可编码为全零。
- 原始真缺失、pdays<-1/非整数/无穷等停止核查；不自动修复。数值中位数步骤保留，但在正确原始数据上无真缺失可填。若 U=off 的 fit 数据中某列全部 unknown，停止而不虚构众数。
- `LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, class_weight=None, random_state=42)`。random_state 与 D 协议保持一致，不做调参或阈值搜索。
- C 的类别块把 job/education 与其余类别拆开，特征顺序/名称前缀不同于 D 参考实现。应使用本管线提供的特征名对应系数，不按旧表位置拼接。完整 OHE，drop=None。
- D 原锁使用 numpy 2.5.4/pandas 3.0.6，本次沿用 A 锁 numpy 2.4.6/pandas 3.0.3。不要将两次运行的小数差异直接归因于 S/U；封版应统一环境再跑全部四组。

## 现有验证结果如何使用

新结果位于 outputs/D/validation，包含 metrics、references、contrasts、subgroups、subgroup_contrasts、predictions、coefficients、run_manifest。数值来源为 D 的评价程序 + C 的正式管线。所有文件是验证结果，不能写成测试结果。run_manifest 的 `provisional_pending_B_C` 仍表示团队交接/人工确认未完成，不表示 C 代码尚不存在。

E0/ES/EU/ESU 的 AP 约为 0.358497/0.359485/0.358526/0.359403；完整准备的 AP 差值约 +0.000905，Recall 差值约 -0.000945（-0.0945 个百分点），Brier 差值约 +0.00000465。差异很小且并非全部改善。最终数值与图表以当前 CSV 为准。

## 还不能完成的封版

待 B 提供真实发现、分层证据及分组确认；待 C/D 本人阅读代码并真实记录复核。`d_handoff_receipt.template.json` 保留 pending，不能代签。
证据收到并确认后，由 D 复制模板为 d_handoff_receipt.json，填写真实复核姓名、日期、确认项和证据路径，先 freeze 再 test。C 不提前查看 test 分数。
最终测试后，D 交完整测试结果与解释给 C 装配，A 用新内核验收，D 核对视频和 Notebook 数字。
