# 环境与复现入口

## 单个英文 Notebook 提交

`analysis.ipynb` 使用短代码下载数据与冻结证据，安装下述环境后可 Run All，无须克隆仓库或提交其他文件。第 17 组提交文件名为 `Assignment_Grp_17.ipynb`。首次运行需访问 `raw.githubusercontent.com`；固定 commit 的 ZIP 经 SHA-256 校验后缓存在当前目录，后续可离线读取缓存。没有 Base64 数据字符串或隐藏数据单元。下载失败时保留原始错误，不跳过校验或切换数据版本。核心分析、清洗、EDA 和四方案比较代码全部直接展示。

英文提交版本生成：`python scripts/build_english_submission.py`；独立空目录、新内核完整验证：`python scripts/validate_english_submission.py`。验证报告为 `outputs/integration/english_submission_validation.json`。生成会清空执行输出，必须随后运行验证再交付。旧 `build_final_delivery.py` 会生成历史混合语言版本；如使用它，必须随后重新生成并验证英文版本。

## 当前统一环境

使用 **CPython 3.12.x**，版本系列记录于 `.python-version`；所有第三方依赖通过 **`requirements.current.txt`** 安装，该文件引用 D 最新交付的完整锁 **`requirements.D.lock.txt`**。

| 主要依赖 | 锁定版本 | 用途 |
| --- | --- | --- |
| numpy | 2.5.4 | 数值运算 |
| pandas | 3.0.6 | CSV、交叉表和分层 |
| scipy | 1.18.1 | sklearn 数值依赖 |
| scikit-learn | 1.8.0 | 固定拆分、预处理和逻辑回归 |
| matplotlib | 3.11.2 | 图表 |
| nbformat | 5.11.1 | 生成与校验 Notebook |
| nbclient | 0.11.0 | 新内核运行 Notebook |
| ipykernel | 7.4.0 | 项目 Python 内核 |

`requirements.lock.txt` 是 A 历史环境；`requirements.C.lock.txt` 是 C 历史环境。保留它们作来源追踪，**不要混装**。原 D 包记录 Python 3.12.14；本机实际 Python 补丁版本与依赖校验结果以 `outputs/integration/environment.json` 为准（未生成时不得声称已验证）。迁移后四组必须在同一环境验证，不能把环境差异解释为 S/U 效果。

Linux/WSL 另在统一入口中条件锁定 IPython 所需的 `pexpect==4.9.0` 和 `ptyprocess==0.7.0`，Windows 不安装这两项。`scripts/check_environment.py` 逐项验证实际版本；不要只检查顶层库或文件名。

## 安装（需要本机操作者授权）

Windows PowerShell，从仓库根目录：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.current.txt
.\.venv\Scripts\python.exe -m pip check
```

Linux / WSL：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.current.txt
.venv/bin/python -m pip check
```

无需安装到系统或全局环境，不要求激活环境，不提交 `.venv`。上游已随仓库提供固定的 UCI 原始数据与 ZIP，不需重新下载。不要用 `pip install -U` 自动更新。若锁定包不可用，报告错误，选择统一替代环境后记录版本并重新验证四组；不可默默升级。

## 验证与生成顺序

下列 `python` 均指项目 `.venv` 的 Python。

```text
python scripts/validate_project.py
python C_preprocessing_tests.py
python scripts/build_b_delivery.py
python scripts/validate_b.py
```

**冻结之前**才运行 `python scripts/run_d.py validate`，然后 `python scripts/validate_d_with_c.py`。这会写验证结果；已有 `outputs/D/freeze.json` 时不要重写，也不要删除冻结记录。

最终测试入口是 `scripts/run_d.py freeze --receipt ...` 后 `scripts/run_d.py test`。本次用户授权通过显式 `--user-authorized` 记录，仍保持人工复核 pending；执行记录不可被后来克隆者当成新的授权。仓库已有测试结果时，应核查保存的逐行预测及哈希，**不再次运行最终测试来选策略**。

最终 Notebook 含核心代码：验证四组可以在内存中复算；最终测试从保存的逐行预测独立核算指标，而不自动重新拟合或覆盖冻结结果。详见 `PROGRESS.md` 的实际完成状态。

已有冻结结果的复核入口：`python scripts/validate_final_results.py`（哈希及 sklearn 独立指标复算），`python scripts/execute_final_notebooks.py`（新内核执行 B 和总 Notebook）。文档生成：`scripts/build_final_delivery.py`、`scripts/build_presentation.py`。不要运行历史 `build_acd_draft.py` 覆盖最终 A+B+C+D Notebook。

冻结协议与历史证据通过 `.gitattributes` 保留原始换行字节，PDF 按二进制处理，避免跨平台 Git 转换破坏哈希。`outputs/integration/git_index_validation.json` 记录推送前 Git 暂存内容与37项冻结对象的逐字节核对。A 的 release manifest 和 D 导入的 delivery manifest 只描述各自原交付；当前最终结果以 freeze、test_completed 和 integration 的验证记录为准。

## Agent 读取顺序

`AGENTS.md` → `PROGRESS.md` → 本文件 → `requirements.current.txt` → `requirements.D.lock.txt` → `outputs/integration/environment.json`。先确认运行记录存在且版本相符，再决定需要做什么；不把“有脚本”或“有锁文件”当作“已运行通过”。
