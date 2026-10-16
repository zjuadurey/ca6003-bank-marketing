"""Update current handoff documents after C receipt; historical copies are retained."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

handoff=ROOT/'notes/D_HANDOFF.md'
text=handoff.read_text(encoding='utf-8')
start=text.index('## 当前交付状态')
end=text.index('## 向 B 获取什么')
text=text[:start]+'''## 当前交付状态（2026-10-16）

C正式管线`src/preprocessing.py`已收到并接入，原始管线源码字节未修改。D已用C正式管线重跑四组验证、全部增量、分组误差、逐行概率、特征名、系数、图表和Notebook。A10项+C14项+D12项技术检查均通过。B发现尚未收到，实际人工复核和最终测试待完成。

旧D结果和报告已归档至`outputs/D/history/reference_20261015/`。C原包及原机证据在`outputs/C/received/`，本机复跑在`outputs/C/`。本次统一沿用D环境（Python3.12.14、numpy2.5.4、pandas3.0.6、sklearn1.8.0），未混用C原机分数。

独立核对：规范特征名对齐后，C与旧D输入矩阵差异为0，概率最大差异<1e-12，阈值分类变化0条。C类别块顺序不同，系数已按C特征名重新生成，不能按旧表位置拼接。证据见`outputs/C/migration_comparison.csv`。

当前只用train/val计算模型结果；B仍需交核心发现并确认分组和解释，之后冻结协议并一次性测试。默认分组来自手册，在验证运行前已固定，未根据验证结果重分组。测试代码已备好，不填造测试分数、不代填人工确认。

'''+text[end:]
start=text.index('## 向 C 获取什么')
end=text.index('## D 文件地图与复现')
text=text[:start]+'''## C已经收到什么、还需什么

| 材料 | 本次状态 |
|---|---|
| 正式build_pipeline和四组配置 | 已收到、接入；配置指向src.preprocessing，四组验证重跑 |
| S/U转换、防泄漏、未见类别、白名单、序列化测试 | C14项和适配后的D12项本机通过 |
| 环境锁、特征名、拟合统计和原机运行证据 | 已收到；本机证据重新生成，原机材料单独保留 |
| C Notebook | 已收到，将在本机新内核运行；完整评价以D章节为准 |
| 整合Notebook | 原ZIP未含说明中的analysis.ipynb；本次补A+C+D草稿，B明确留空 |
| C姓名和实际理解/复核 | 原包为pending，需C本人阅读并确认；不需重复索取已有代码 |

原包提及的build_c_delivery.py、execute_c_notebooks.py也未提供，但当前D不依赖它们。本次已补整合与复现脚本。若C有更新全项目，可再提供对照；C最后仍负责核对单一Notebook的装配和贡献。完整清单见`D_MISSING_ITEMS.md`。

'''+text[end:]
text=text.replace('日期：2026-10-15。','更新：2026-10-16。')
text=text.replace('- `src/d_reference_pipeline.py`：D 临时补齐的 C 接口，贡献与待审状态明确。','- `src/preprocessing.py`：当前使用的C正式管线；`src/d_reference_pipeline.py`仅作历史迁移核对。\n- `analysis.ipynb`：A+C+D整合草稿，B留空，不是完整课程成品。\n- `outputs/C/local_integration.json`：原包SHA-256、来源文件和接入记录。')
text=text.replace('.\\.venv\\Scripts\\python.exe scripts/validate_d.py','.\\.venv\\Scripts\\python.exe C_preprocessing_tests.py\n.\\.venv\\Scripts\\python.exe scripts/validate_d.py\n.\\.venv\\Scripts\\python.exe scripts/validate_c_integration.py')
text=text.replace('1. D/C 核对正式管线及测试；B 核对分组与解释。必要时替换 pipeline_module 并重跑四组验证及 QA；保留变更依据。禁止先看测试分数再修改策略。','1. C正式代码的技术接入已完成。收B核心发现和分层表，核对分组与解释，成员实际理解并确认。如果改任何规则，重跑验证并记录依据。禁止先看测试分数再修改策略。')
text=text.replace('B/C 材料缺失、模型依赖与人工核验状态如实保留','B材料尚缺、C技术交付已收到、人工核验待完成的状态如实保留')
handoff.write_text(text,encoding='utf-8')

(ROOT/'notes/D_MISSING_ITEMS.md').write_text('''# 2026-10-16 · C接入后还缺什么

C正式源码、测试与C章节已收到并接入。D已重跑四组验证、增量、分组误差、概率、特征名和系数，更新报告与Notebook。A+C+D整合草稿已补齐，B章节仍留空。

## 现在优先向B索取

| 必需材料 | 最低可用形式 | 用途 |
|---|---|---|
| 一项非显而易见发现 | 仅固定train的代码/Notebook、发现文字、至少1张可重建交叉或分层表；写清n、正例数、分母 | 补齐核心研究内容；A质量审计和D模型结果不能替代B发现 |
| 历史联系三字段证据 | pdays=-1/≥0 × previous=0/>0 × poutcome；例外row_id和说明 | 解释S语义及已有历史变量可能带来的冗余 |
| unknown分层证据 | job/education unknown比例、认购率、年龄或联系史分层；比较总体与分层差异 | 解释U效果与分组代价，提供替代解释 |
| 分组与结论确认 | 确认手册年龄3组和原始unknown2组；复核D对改善/恶化/无变化的解释 | 测试前完成规则与解释核对 |

可复制给B（未代发）：

> 我是LIANG SHUANG，D。C管线已经接好并通过本机检查。请交你仅基于固定train的核心发现、历史三字段交叉表、unknown分层表和可复现代码，每张表附n、正例数与分母。请确认手册分组，并核对D目前“整体改善很小、部分客群有代价”的解释。收到后我们补齐整合Notebook，再冻结最终测试。

## C还需确认什么

- 不缺正式管线、14项测试代码、环境锁、特征名或原机证据；不需重复索取。
- C本人姓名和实际理解/复核仍为pending；请C阅读并确认这版管线及共同约定，补真实贡献信息。
- 原ZIP是增量包，没有说明提及的analysis.ipynb、build_c_delivery.py、execute_c_notebooks.py。本次已补整合草稿和复现脚本；这些缺失脚本不阻碍当前D运行。若C有更新整合版，可提供对照。
- C最后需要把B内容、最终测试和讨论并入单一Notebook，核对引用、GenAI声明和成员贡献。

## 全组还缺的最终动作

1. 收B证据，补第03章；成员实际核对代码、字段政策、分组和解释。
2. 填真实交接记录，运行freeze；用train+val重拟合全部四组，再一次性测试。当前无测试分数，不可把验证分数移入测试列。
3. 用最终测试更新结果、图、报告和讲稿；保留全部增量及变差情况。
4. C核对单一最终Notebook，A在新内核Run All。此次草稿自动执行成功不能代替成员本人最终验收。
5. 四人出镜并讲解，录制≤12分钟单个连续视频，检查公开链接、组号与实际课程提交；各自完成互评。

## 当前证据位置

- C正式源码：src/preprocessing.py，原字节未改；配置已指向该模块。
- 原包和原机证据：outputs/C/received/；接入记录：outputs/C/local_integration.json。
- 本机C测试：outputs/C/tests.json；D检查：outputs/D/qa.json；迁移核对：outputs/C/migration_comparison.csv。
- 旧D结果：outputs/D/history/reference_20261015/，保留但不是当前主结果。
- 整合草稿：analysis.ipynb；D独立章节：notebooks/D_evaluation.ipynb。

本次统一使用requirements.D.lock.txt：Python3.12.14、numpy2.5.4、pandas3.0.6、sklearn1.8.0。C原锁另作来源保留，不同时安装两份冲突锁。规范名对齐后的输入矩阵差异0，概率差异<1e-12，分类变化0条；这些不是新增S/U提升。未推送GitHub、联系成员、代作人工确认或执行真实测试。
''',encoding='utf-8')

(ROOT/'START_D.md').write_text('''# LIANG SHUANG · D / 2026-10-16更新

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
''',encoding='utf-8')

readme=ROOT/'README.md'
text=readme.read_text(encoding='utf-8')
lines=text.splitlines()
for i,line in enumerate(lines):
    if line.startswith('> **2026-10-15'):
        lines[i]='> **2026-10-16 · C正式管线已接入D：** 当前使用src.preprocessing，全部验证已重跑，A10+C14+D12项检查通过。先看[缺失清单](notes/D_MISSING_ITEMS.md)、[D交接](notes/D_HANDOFF.md)及[A+C+D草稿](analysis.ipynb)。B发现、实际人工核验和最终测试待完成；统一复跑用requirements.D.lock.txt。下文保留A原始交付说明。'
readme.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Updated current handoff, missing-items checklist, START_D and README.')
