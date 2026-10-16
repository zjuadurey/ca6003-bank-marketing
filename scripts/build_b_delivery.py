"""Build B evidence narrative and a standalone B chapter from computed tables."""
from pathlib import Path
import sys
import json
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.b_discovery import write_outputs


def markdown_table(frame):
    def value(x):
        return f'{x:.6g}' if isinstance(x, float) else str(x)
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    t = write_outputs()
    h, u = t['history_crosstab'], t['unknown_strata']
    contacted_unknown = int(h.loc[h.history_group.eq('previously_contacted') & h.poutcome.eq('unknown'),'n'].sum())
    unknown_total = int(h.loc[h.poutcome.eq('unknown'),'n'].sum())
    detail = u[u.field.eq('any_unknown') & u.dimension.isin(['overall','age_group'])][
        ['dimension','stratum','information','n','n_positive','subscription_rate','small_sample']]
    standardized = t['age_standardized'].query("field == 'any_unknown'").set_index('information')
    standardized_difference = 100*(standardized.loc['unknown','age_standardized_rate']-standardized.loc['known','age_standardized_rate'])
    crude = u[u.field.eq('any_unknown') & u.dimension.eq('overall')].set_index('information')
    crude_difference = 100*(crude.loc['unknown','subscription_rate']-crude.loc['known','subscription_rate'])
    findings = f'''# B / D2：数据洞察与处理依据

本报告仅探索固定训练集：27,126 条，3,173 个正例。代码：`src/b_discovery.py`；原始表：`outputs/B/`。比例均注明分母，不对验证或测试数据做开放式 EDA。分析由 OpenAI Codex 按用户指令辅助完成；B 本人的实际理解与复核待记录。

## 核心发现：未知值的语义必须由跨字段关系解释

训练集中 `pdays>=0` 与 `previous>0` 的状态完全对应；但 `poutcome=unknown` 有两种观察情境：22,235 条无联系历史，另有 {contacted_unknown} 条已联系历史。后者占所有 outcome unknown 的 {contacted_unknown}/{unknown_total}（{contacted_unknown/unknown_total:.4%}）。

{markdown_table(h[['history_group','previous_group','poutcome','n','n_positive','subscription_rate']])}

单看 unknown 计数，无法区分“没有前次活动可记录”与“有前次活动但结果未知”；后者的真实原因无法从公开字段识别，不能自动归为解析错误。例外 row_id 和原值见 `history_exceptions.csv`，全部保留。

这支持 S 明确区分联系状态与间隔、U 仅作用于 job/education，而不统一填补 poutcome。与此同时，previous 的零/正状态已能表达相同联系状态，poutcome 也高度重叠，因此 S 的额外预测收益可能有限。不过 previous 作为连续数值进入逻辑回归并不等价于二元状态，不能证明 S 数学上冗余。S 还改变维数和正则化几何，模型对比不是纯因果效应。

## 辅助发现一：unknown 的总体关联在年龄组中不一致

{markdown_table(detail)}

总体未知组认购率 13.52%，两列已知组 11.61%；年轻组未知状态对应更高认购率，≥60 岁组则方向不同。≥60 未知组仅 97 条，因此必须标注小样本。并非所有分层方向与总体相反，不称为辛普森悖论。原始 job、education 的独立分层，以及联系历史分层均见 `unknown_strata.csv`；该表还报告 unknown/所在分层样本数。

替代解释包括客户构成、记录流程差异、未测量因素和随机波动。未知信息不必然随机出现，但本结果也不能判定缺失机制为 MAR/MNAR。众数填补会合并可观察的信息状态，保留类别可能保留有用信息，也可能加剧小样本估计波动。因此 U 应用相同模型实测，而不能预先承诺提升或公平性改善。

### 年龄构成的描述性检查

以下按同一训练集年龄权重重加权各年龄组认购率；不拟合模型、不改变分组、不做因果调整。

{markdown_table(t['age_standardized'])}

权重见 `age_weights.csv`。标准化只控制这里的粗年龄构成，不能排除年龄组内差异或其他混杂；97 条样本的波动仍保留。

对任一 unknown 的信息组，粗认购率差为 {crude_difference:+.4f} 个百分点；使用共同年龄构成后为 {standardized_difference:+.4f} 个百分点。两者差异显示年龄构成会影响总体比较，但不把剩余差值解释成 unknown 的因果作用。该重加权仅作为训练集解释证据，不改变模型训练数据或 D 的评价分组。

## 辅助发现二：数值尾部需要治理，不能默认删除

{markdown_table(t['numeric_quality'])}

`numeric_tail_examples.csv` 保存余额与历史联系次数的极端训练记录及 row_id。负余额在字段语义上可允许，高联系次数值得核查但缺少逐次历史，不能证明错误，也不能确认每个极端值真实有效。本研究保留全部行；所有实验使用同样的 StandardScaler，未引入删行、截尾、log 变换或额外异常值因素。线性模型仍可能受长尾影响，这是本研究范围限制。

## 从观察到实验

| 观察 | 准备决策 | 验证方式 | 解释边界 |
| --- | --- | --- | --- |
| 哨兵状态与间隔不同，其他历史字段已高度重叠 | S 拆分状态与条件间隔 | ES−E0、ESU−EU | 小收益不代表语义无价值，也不证明完全冗余 |
| job/education unknown 的关联随分层变化 | U 保留类别 vs 训练众数 | EU−E0、ESU−ES | 不能因关联而推断因果或缺失机制 |
| 总体平均可能掩盖群体差异 | 固定年龄及原始信息完整性组 | Recall、FPR、Brier 与分母 | 分组差异不等于歧视判定 |

验证/最终测试结果与解释另见 `notes/RESULTS.md`，避免把测试结果混入训练集发现。所有改进、恶化和无变化均报告。没有置信区间或显著性检验，不使用“显著改善”。无客户 ID 和精确快照时间，不能保证客户级独立性、未来泛化或营销因果收益。

## 图与交接

三张训练主图：`01_overview.png`、`02_history.png`、`03_unknown_age.png`；第四张由 D 结果生成 `04_model_tradeoffs.png`，与训练图明确区分。每张图的原表随包提供。

C：将本章放入最终 Notebook 的 EDA 章节，沿用 S/U 实现。D：沿用 `notes/B_groups_review.md` 的固定分组，将模型变化与上述可能解释对应。全组：人工核查 5 条例外、主要表分母和结果解释；不要把 AI 自动化验证写成人工签字。
'''
    (ROOT/'notes/B_findings.md').write_text(findings,encoding='utf-8')
    contract = (ROOT/'src/data_contract.py').read_text().replace('from __future__ import annotations\n','').replace('ROOT = Path(__file__).resolve().parents[1]','ROOT = Path.cwd() if (Path.cwd()/"config/dataset_lock.json").exists() else Path.cwd().parent')
    source = (ROOT/'src/b_discovery.py').read_text().replace('from src.data_contract import ROOT, load_partition','')
    source = source[:source.index("if __name__ == '__main__':")]
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md('# B / Training-only discovery\n\nRun from the repository root or notebooks/. All core analysis and loading code is included below. No model fitting or holdout EDA. Human review remains pending.'),
             code(contract), code(source), code('tables = investigate()\nfrom IPython.display import display\nfor name, table in tables.items():\n    if name != "training_row_ids":\n        print(name)\n        display(table)'), md(findings),
             code('import tempfile\nfrom IPython.display import Image, display\nwith tempfile.TemporaryDirectory() as folder:\n    draw_figures(tables, folder)\n    for path in sorted(Path(folder).glob("*.png")):\n        display(Image(filename=str(path)))'),
             md('## GenAI disclosure\nOpenAI Codex assisted code, analysis, charts, narrative, and automated validation. Independent CSV reconstruction and frozen row alignment checks are in scripts/validate_b.py. Members must understand and personally verify the work; no human signature is implied.')]
    nb = nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.write(nb,ROOT/'notebooks/B_discovery.ipynb')
    print('B tables, three figures, findings and notebook built.')


if __name__ == '__main__':
    main()
