"""Assemble current B narrative, final results and A/B/C/D notebook without rerunning test fits."""
from pathlib import Path
from copy import deepcopy
import json
import sys

import pandas as pd
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_b_delivery import markdown_table


def main():
    out = ROOT/'outputs/D'
    assert (out/'test_completed.json').exists(), 'Final test must have actually completed'
    validation = pd.read_csv(out/'validation/metrics.csv')
    test = pd.read_csv(out/'test/metrics.csv')
    delta = pd.read_csv(out/'test/contrasts.csv')
    groups = pd.read_csv(out/'test/subgroup_contrasts.csv')
    group_counts = pd.read_csv(out/'test/subgroups.csv').query("experiment == 'E0'").set_index(['dimension','group'])
    full = delta[delta.effect.eq('full')].iloc[0]
    references = pd.read_csv(out/'test/references.csv')
    results = f'''# 四方案结果与解释（最终测试已执行）

训练集用于 B 探索和验证拟合；验证集只用于预定模型对比。冻结后四方案分别以 train+val=36,168 条重拟合，在固定 test=9,043 条报告。无调参、无阈值搜索，原白名单和分组不变。实际环境、拟合迭代、输入与代码哈希见 `outputs/D/test/run_manifest.json`；冻结授权见 `outputs/D/freeze.json`。用户授权完成计算，不代表成员已人工签字。

## 验证结果（fit=train，evaluation=val）

{markdown_table(validation[['experiment','n','n_positive','AP','Recall','F1','Brier','Accuracy']])}

## 最终测试（fit=train+val，evaluation=test）

{markdown_table(test[['experiment','n','n_positive','TP','FP','TN','FN','AP','Recall','F1','Brier','Accuracy']])}

## 五项增量（原始分数差）

{markdown_table(delta)}

完整准备 ESU−E0：AP {full.delta_AP:+.8f}，Recall {100*full.delta_Recall:+.4f} 个百分点，F1 {full.delta_F1:+.8f}，Brier {full.delta_Brier:+.8f}。Brier/FPR 越低越好，AP/Recall/F1 越高越好。变差和无变化同样保留；没有显著性检验，不称“显著提升”。

S 应同时看 ES−E0 与 ESU−EU；U 同时看 EU−E0 与 ESU−ES。联系历史在 pdays、previous、poutcome 中重叠，是小变化的一种可能解释，不能据此证明机制。U 保留的信息状态具有年龄异质性，也不能保证对所有分组有利。S/U 改变表征、维数与正则化几何，不能将差值解释为纯因果效果。测试与验证拟合样本规模不同，两阶段分数之差不是准备步骤的效应。

本次测试中，S 在两种 U 条件下均略提高 AP、降低 Brier；仅在 U 关闭时多识别一个正例。U 在两种 S 条件下均降低 AP 和 Recall、提高 Brier。因此“保留 unknown 可能保留信息”是值得检验的理由，不是已经证实的性能收益；本次模型下未观察到该收益。不能据此推广为众数填补永远更好。

## 分组完整方案差值

{markdown_table(groups[groups.effect.eq('full')])}

分母、正负例数、TP/FP/TN/FN 和完整四方案组内指标见 `outputs/D/test/subgroups.csv`，全部五项分组对照见 `subgroup_contrasts.csv`。信息组按原始值定义；n<100 或正/负例少于20显示 small_sample。年龄与信息维度相互重叠，不可相加。分组差异是描述结果，不等于歧视或公平性证明。

本次完整方案相对基线，≥60 岁组 Recall 从 30/112 降到 29/112（−0.8929 个百分点），原始 unknown 组从 10/59 降到 9/59（−1.6949 个百分点）；对应组的 Brier 也上升。年龄与信息分组有重叠，不能把两项变化相加成漏识别两个人。总体变化很小仍可能掩盖某组更明显的代价。

## 无特征参考与限制

{markdown_table(references[['reference','n','AP','Recall','F1','Brier','Accuracy','probability']])}

Accuracy 需与全预测 no 比较，同时看低 Recall 对筛选的限制。AP 衡量排序，不是梯形 PR-AUC；Brier 同时反映校准与区分度，不能单独视为校准检验。无客户 ID、完整快照时间或随机营销分配，不能证明客户独立、未来泛化、电话因果效果或实际收益。尾部值未无依据删除，可能影响线性模型。GenAI 辅助与人工理解复核的边界必须在提交中保留。
'''
    (ROOT/'notes/RESULTS.md').write_text(results,encoding='utf-8')

    fig, axes = plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    for ax, metric in zip(axes[:2], ['AP','Brier']):
        ax.plot(test.experiment,test[metric],marker='o',color='#28758b')
        ax.set(title=f'Test {metric}'+(' (lower better)' if metric=='Brier' else ' (higher better)'))
        ax.ticklabel_format(axis='y',useOffset=False)
    g = groups[groups.effect.eq('full')]
    labels = [f'{r.dimension}: {r.group} (n={int(group_counts.loc[(r.dimension,r.group),"n"])}, '
              f'yes={int(group_counts.loc[(r.dimension,r.group),"n_positive"])})' for r in g.itertuples()]
    axes[2].barh(labels,g.delta_Recall*100,color='#bd8831')
    axes[2].axvline(0,color='black',lw=.7)
    axes[2].set(title='ESU - E0: test group Recall',xlabel='Percentage points')
    fig.suptitle(f'Final test: overall changes and group tradeoffs (n=9,043; yes={int(test.iloc[0].n_positive)}; threshold=0.5)')
    fig.supxlabel('Metric axes are zoomed; no confidence intervals. Age and information groups overlap.')
    fig.savefig(ROOT/'outputs/B/figures/04_model_tradeoffs.png',dpi=160);plt.close(fig)

    a = nbformat.read(ROOT/'notebooks/A_data_foundation.ipynb',as_version=4)
    d = nbformat.read(ROOT/'notebooks/D_evaluation.ipynb',as_version=4)
    b = nbformat.read(ROOT/'notebooks/B_discovery.ipynb',as_version=4)
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md('# CA6003 Bank Marketing — A+B+C+D\n\n研究问题：特殊值语义与未知信息处理如何影响预测可靠性？\n\n本 Notebook 汇总 A 数据治理、B 训练集调查、C 正式准备管线和 D 四方案评价。最终测试已实际执行；成员人工复核、录制和提交仍待完成。核心代码内嵌；从仓库根目录或 notebooks/ 启动，环境见 ENVIRONMENT.md。Run All 复算训练分析和验证模型，并从保存的测试预测核算结果，不重新打开测试训练流程。'),
           deepcopy(d.cells[2]),code('from IPython.display import display'),
           md('## 01–02 / A：数据来源、边界与质量治理'),deepcopy(d.cells[4])]
    audit=(ROOT/'src/audit.py').read_text().replace('from __future__ import annotations\n','').replace('from src.data_contract import CATEGORIES, EXPECTED_COLUMNS, FEATURES, NUMERIC, ROOT','')
    for index in range(1,13):
        cell=deepcopy(a.cells[index])
        if cell.cell_type=='code' and 'from src.audit import profile_train, audit_train' in cell.source:
            cells.append(code(audit));cell.source=cell.source.replace('from src.audit import profile_train, audit_train','')
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('B 仍须独立完成其核心发现、分层证据与限制。','B 的独立调查和分层证据见下一章。')
        cells.append(cell)
    cells.append(md('## 03 / B：训练集核心发现与分层证据'))
    cells.extend(deepcopy(b.cells[2:]))
    cells.append(md('## 04–05 / C 正式准备与 D 验证评价'))
    cells.extend(deepcopy(d.cells[5:17]))
    # B uses a noninteractive backend for reproducible exported images.
    # Embed D's validation figure explicitly so it remains visible in that backend.
    cells[-1].source = cells[-1].source.replace('plt.show()',
        'from io import BytesIO\nfrom IPython.display import Image\n'
        'buffer = BytesIO()\nfig.savefig(buffer, format="png", dpi=140)\n'
        'display(Image(data=buffer.getvalue()))\nplt.close(fig)')
    cells.append(md('## 05b / 最终测试：保存的预测与完整指标\n\n下面从冻结后实际生成的逐行预测重新计算指标，检查哈希、样本顺序和标签一致性。不会重新训练测试模型，不使用测试结果开发新策略。'))
    cells.append(code('''import hashlib
frozen = json.loads((ROOT/'outputs/D/freeze.json').read_text())
completed = json.loads((ROOT/'outputs/D/test_completed.json').read_text())
assert completed['freeze_sha256'] == hashlib.sha256((ROOT/'outputs/D/freeze.json').read_bytes()).hexdigest()
for name, expected in completed['test_hashes'].items():
    assert hashlib.sha256((ROOT/'outputs/D/test'/name).read_bytes()).hexdigest() == expected, name
for rel, expected in frozen['fingerprint']['files'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
for rel, expected in frozen['evidence_hashes'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
for name, expected in frozen['validation_hashes'].items():
    assert hashlib.sha256((ROOT/'outputs/D/validation'/name).read_bytes()).hexdigest() == expected, name
prediction_table = pd.read_csv(ROOT/'outputs/D/test/predictions.csv',float_precision='round_trip')
saved_test = pd.read_csv(ROOT/'outputs/D/test/metrics.csv')
X_holdout, y_holdout = load_partition('test', allow_test=True)
test_rows, test_group_rows = [], []
for experiment, pred in prediction_table.groupby('experiment',sort=False):
    assert pred.row_id.tolist() == y_holdout.index.tolist()
    assert pred.y_true.tolist() == y_holdout.tolist()
    assert pred.prediction.tolist() == (pred.probability >= .5).astype(int).tolist()
    metrics = evaluate(pred.y_true,pred.probability)
    saved = saved_test.set_index('experiment').loc[experiment]
    for key,value in metrics.items():
        np.testing.assert_allclose(value,saved[key],atol=1e-12,rtol=0,equal_nan=True)
    test_rows.append(dict(experiment=experiment,**metrics))
    test_group_rows.extend(subgroup_metrics(X_holdout,y_holdout,pred.probability.to_numpy(),experiment))
final_metrics = pd.DataFrame(test_rows)
final_groups = pd.DataFrame(test_group_rows)
pd.testing.assert_frame_equal(final_groups,pd.read_csv(ROOT/'outputs/D/test/subgroups.csv'),check_exact=False,atol=1e-12)
pd.testing.assert_frame_equal(contrasts(final_metrics),pd.read_csv(ROOT/'outputs/D/test/contrasts.csv'),check_exact=False,atol=1e-12)
pd.testing.assert_frame_equal(contrasts(final_groups,['dimension','group']),pd.read_csv(ROOT/'outputs/D/test/subgroup_contrasts.csv'),check_exact=False,atol=1e-12)
display(final_metrics)
display(contrasts(final_metrics))
display(final_groups)
display(contrasts(final_groups,['dimension','group']))
display(pd.read_csv(ROOT/'outputs/D/test/references.csv'))
from IPython.display import Image
display(Image(filename=str(ROOT/'outputs/B/figures/04_model_tradeoffs.png')))
'''))
    cells.append(md('## 06 / 讨论与限制\n\n'+results))
    cells.append(md('## 贡献、GenAI 与待办\n\nA：固定数据来源、治理、原始审计和划分。B（本次用户）：在 Codex 辅助下形成训练集发现、独立计数验证、分层证据、图表与叙事。C：原 ZIP 中正式 S/U 管线和转换测试。D（LIANG SHUANG）：统一评价及原交付结果。本次 Codex 按用户要求完成 C/D 本地整合、B 实现、统一环境验证、获授权的冻结测试和最终 Notebook 装配；不冒称 C/D 本人完成本次新增工作。\n\nGenAI 工具：OpenAI Codex；用途：代码、排错、分析建议、图表、解释草稿、自动验证。核验：原始 CSV 独立计数、数据哈希和固定划分检查、转换边界与防泄漏测试、逐行预测复算指标、新内核运行。实际人工阅读、理解和签字仍 pending。\n\n还需成员姓名、组号、人工复核、四人出镜视频、公开视频链接、NTULearn 提交与互评。参考 UCI Bank Marketing（DOI 10.24432/C5K306）和前文数据来源；按 CC BY 4.0 保留作者与许可。'))
    for c in cells:
        if c.cell_type=='code':c.outputs=[];c.execution_count=None
    nbformat.write(nbformat.v4.new_notebook(cells=cells,metadata=d.metadata),ROOT/'analysis.ipynb')
    print('Built current results, fourth B figure and integrated notebook.')


if __name__=='__main__':
    main()
