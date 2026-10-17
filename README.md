# CA6003 — Bank Marketing: A+B+C+D 整合

课程提交只上传一个 `.ipynb`。当前 `analysis.ipynb` 已恢复自包含版本，第 17 组提交时命名为 `Assignment_Grp_17.ipynb`。`BankMarketing_Data.zip` 是先前拆分版的本地资料，不是本次提交附件。

**当前进度与下一步：[PROGRESS.md](PROGRESS.md)。环境与版本：[ENVIRONMENT.md](ENVIRONMENT.md)。Agent 入口：[AGENTS.md](AGENTS.md)。**

统一安装入口为 `requirements.current.txt`（引用 D 版本锁），不混装历史 A/C 锁。B 核心代码为 `src/b_discovery.py`，成果生成入口为 `scripts/build_b_delivery.py`；下方原 A 文档和命令保留作来源说明，其“B/C/D 待完成”是初始交付状态，当前状态以 PROGRESS 为准。

当前成果：[总 Notebook](analysis.ipynb)、[B 独立 Notebook](notebooks/B_discovery.ipynb)、[B 发现](notes/B_findings.md)、[最终测试解释](notes/RESULTS.md)、[四人英文讲稿](PRESENTATION_SCRIPT_EN.md)。最终测试已运行，不要重跑测试来调整策略。

**单文件提交使用 `analysis.ipynb`：正文、注释、图表和输出均为英文。** 数据、划分和冻结评价证据已内嵌，无须克隆仓库；安装 Notebook 开头列出的环境后即可在任意目录 Run All。核心分析、转换和评价函数直接展示。独立空目录复核入口为 `scripts/validate_english_submission.py`，完整环境详见 ENVIRONMENT。成员为 A Yu Qiushi、B Chen Junyu、C Sun Shaojie、D Liang Shuang；人工复核仍待完成。

## 原 A 交付说明

> **2026-10-16 · C正式管线已接入D：** 当前使用src.preprocessing，全部验证已重跑，A10+C14+D12项检查通过。先看[缺失清单](notes/D_MISSING_ITEMS.md)、[D交接](notes/D_HANDOFF.md)及[A+C+D草稿](analysis.ipynb)。B发现、实际人工核验和最终测试待完成；统一复跑用requirements.D.lock.txt。下文保留A原始交付说明。

本仓库交付四人项目的 **A / D1 数据基础包**。研究问题：特殊值语义与未知信息处理，如何影响银行营销分类的可靠性？仅使用 UCI 的旧版 `bank-full.csv`。B 的核心发现、C 的四组预处理、D 的模型与评价、最终整合 Notebook 和全组视频仍由对应成员完成。

## 从这里开始

1. 阅读 [数据卡](notes/data_card.md)、[数据契约](notes/DATA_CONTRACT.md) 和 [交接清单](notes/HANDOFF.md)。
2. 使用下面命令复现下载、校验和固定划分。
3. 打开 [A 数据章节 Notebook](notebooks/A_data_foundation.ipynb)，查看已执行结果。
4. 人工复核字段语义与 AI 辅助内容，填写真实复核人；当前 reviewer 均为 pending。

## 一键数据准备与复现

已验证 CPython 3.12.10。依赖仅安装到本项目 `.venv`，无需激活环境或修改系统执行策略。

Windows PowerShell，从仓库根目录：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.lock.txt
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/validate_project.py
.\.venv\Scripts\python.exe scripts/validate_notebook.py
```

macOS/Linux（同样使用 Python 3.12；本机实际验证平台见 outputs/environment.json）：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.lock.txt
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/validate_project.py
.venv/bin/python scripts/validate_notebook.py
```

GitHub 仓库包含经许可分发的原始数据及来源 ZIP，因此克隆后可用 `scripts/prepare_data.py --offline`，无需再次联网。若删去原始数据但保留外 ZIP，也能从锁定包重建。完整重新下载可在一个新的空目录副本中运行，不要手工改已有 raw 或 split。脚本发现已有不可变文件与预期不同会报错，并保留该文件供核查。

核心代码与 Notebook 均使用相对项目路径。脚本根据自身位置定位根目录；Notebook 从根目录或 notebooks/ 执行。`requirements.lock.txt` 锁定实际环境全部依赖版本；Python 版本单独记录。

## 文件地图

| 路径 | 用途 |
| --- | --- |
| `config/dataset_lock.json` | 官方 URL、四份原始文件哈希、17 列、白名单和拆分约定 |
| `data/source/` | 原始总 ZIP、嵌套 bank.zip、下载时间与 provenance |
| `data/raw/` | 原始字节未改的 bank-full.csv 和 bank-names.txt |
| `data/splits/` | 固定 row_id→split 清单与原拆分返回顺序 |
| `src/data_contract.py` | 原始验证、精确划分与队友加载接口 |
| `src/audit.py` | 训练集 profiling / 质量审计、预定格式检查 |
| `outputs/` | 已运行的审计证据、环境、验证报告与发布清单 |
| `notes/` | 17 字段字典、字段政策、来源许可、交接、AI 声明 |
| `notebooks/A_data_foundation.ipynb` | A 的已执行数据章节，供 C 合并 |
| `scripts/` | 下载准备、只读验证、Notebook 验证和文档生成 |

## 已固定的研究边界

- 联系前的历史记录离线研究。允许的 11 列为 `age, job, marital, education, default, balance, housing, loan, pdays, previous, poutcome`。
- `duration, campaign, contact, day, month` 不进入任何正式实验。`y` 仅为目标（yes=1/no=0）；`row_id` 仅用于追踪。
- 原始行序不变；两次 `train_test_split` 均用 stratify、random_state=42，得到 60/20/20。任何成员不得重新洗牌或换 seed。
- A 不填补、不缩放、不编码、不删 unknown、不去重。特殊状态与极端值先保留并记录。
- 仅 train 用于开放 profiling/EDA。test 目前仅做预先约定的格式、范围、覆盖检查检查，禁止用其决定策略。所有学得的参数只在当前训练分区拟合。
- 缺少客户 ID、完整时间戳和特征快照；无法证明客户级独立性或未来泛化。`row_id` 不能弥补这一限制。

实测：45,211 行、17 列；train/val/test = 27,126/9,042/9,043。训练分区实际统计与分母见 outputs/，不是手册中的预计值。A 包不含模型分数。

## 许可、隐私与学术披露

数据按 [UCI 页面](https://archive.ics.uci.edu/dataset/222/bank+marketing) 标明的 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 分发；保留来源、作者、旧版引用和修改说明，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。私有仓库不改变数据原有许可。没有上传课程 PDF、内部手册、登录信息或个人密钥。

本包由 OpenAI Codex 辅助编写并实际运行检查；[AI 声明](notes/AI_USAGE.md) 明确列出用途、验证方式及尚待完成的学生人工复核。自动化通过不代表人工签字或完整课程作业已完成。
