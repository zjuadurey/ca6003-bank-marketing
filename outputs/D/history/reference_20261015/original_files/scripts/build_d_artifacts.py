"""Build D's notebook, evidence charts, Chinese PDF and reviewable narration."""
from pathlib import Path
import html
import json
import textwrap
import nbformat as nbf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/D'
VAL = OUT/'validation'
FIG = OUT/'figures'
FIG.mkdir(parents=True, exist_ok=True)
table = pd.read_csv(VAL/'metrics.csv').set_index('experiment')
delta = pd.read_csv(VAL/'contrasts.csv')
groups = pd.read_csv(VAL/'subgroups.csv')
gd = pd.read_csv(VAL/'subgroup_contrasts.csv')
refs = pd.read_csv(VAL/'references.csv').set_index('reference')


def figures():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), layout='constrained')
    for ax, metric, title in zip(axes, ['AP','Recall','Brier'], ['AP (higher is better)', 'Recall at 0.5 (higher is better)', 'Brier (lower is better)']):
        values = table[metric].to_numpy()
        ax.scatter(np.arange(4), values, c=['#51758c','#007e87','#b57834','#16384d'], s=65)
        ax.set_xticks(np.arange(4), table.index)
        ax.set_xlim(-.5, 3.5)
        ax.set_title(title, fontsize=11)
        pad = max(np.ptp(values)*.7, .0001)
        ax.set_ylim(values.min()-pad, values.max()+pad)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        for i, value in enumerate(values):
            ax.annotate(f'{value:.6f}', (i,value), xytext=(0,10), textcoords='offset points', ha='center', fontsize=8)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Validation only | train n=27,126; validation n=9,042 (1,058 positives)\nZoomed axes show small differences; no uncertainty interval or significance claim.', fontsize=11)
    fig.savefig(FIG/'validation_overall.png', dpi=180)
    plt.close(fig)
    full = gd[gd.effect.eq('full')]
    fig, axes = plt.subplots(1,2,figsize=(10.2,3.6), layout='constrained')
    labels = ['Age <30','Age 30-59','Age >=60','Any unknown','Both known']
    for ax, metric, title, factor in zip(axes,['delta_Recall','delta_Brier'],['Recall change (percentage points)','Brier change (score units)'],[100,1]):
        values=full[metric].to_numpy()*factor
        ax.barh(labels,values,color=['#b57834' if (v<0 if metric=='delta_Recall' else v>0) else '#007e87' for v in values])
        ax.axvline(0,color='#243e50',lw=.7)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis='x',alpha=.15)
        from matplotlib.ticker import MaxNLocator, FuncFormatter
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if metric == 'delta_Brier':
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v*1e4:.1f}'))
            ax.set_xlabel('Tick values in 1e-4 score units')
    fig.suptitle('ESU minus E0 | validation only | lower Brier is better\nAge and information are separate partitions; do not add them together.',fontsize=11)
    fig.savefig(FIG/'validation_subgroup_changes.png',dpi=180)
    plt.close(fig)


def notebook():
    md=nbf.v4.new_markdown_cell
    code=nbf.v4.new_code_cell
    cells=[md('# D4 · 对照评估 / LIANG SHUANG\n\n**验证结果已运行；B/C 人工交接尚未完成；测试集尚未进行预测评价。**\n\n依据执行手册第6–10页及 A 的固定数据契约。D 临时实现了手册规定的参考管线，便于先完成评估工作；它不代表 C 已交付或已审阅。最终全组 Notebook 由 C 合并，不能将此文件当作全组作业已经完成。'),
    md('## 1. Protocol and provenance\n\nSame 11 raw features, frozen A splits, same logistic regression (C=1, lbfgs, max_iter=2000, class_weight=None), threshold p>=0.5. E0=(S off,U off), ES=(on,off), EU=(off,on), ESU=(on,on). Validation fits train only; final test refits train+val after B/C review. No hyperparameter/threshold search. Age and original unknown groups were specified by the manual before evaluation. All core loading, transformation and metric functions are included below for C to merge.'),
    code("from pathlib import Path\nimport sys, json, platform\nROOT = Path.cwd()\nif not (ROOT / 'data/raw/bank-full.csv').exists():\n    ROOT = ROOT.parent\nassert (ROOT / 'data/raw/bank-full.csv').exists(), 'Run from repository root or notebooks/'\nimport sklearn, pandas as pd, numpy as np\nprint({'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__})"),
    md('## 2. Frozen A loading contract (included source)\n\nrow_id is a tracking index, never a predictor. Test loader is guarded; the validation cells below never request test. Hash/format checks on the raw source are part of A’s existing integrity contract, not exploratory test analysis.')]
    source=(ROOT/'src/data_contract.py').read_text(encoding='utf-8')
    source=source.replace('from __future__ import annotations\n','').replace('ROOT = Path(__file__).resolve().parents[1]\n','')
    cells += [code(source), md('## 3. D reference pipeline (awaiting C acceptance)\n\nS separates the prior-contact flag and conditional interval; the binary flag is passed through without scaling. Numeric fields are standardized. U only affects job/education. Unexpected genuine missing values or pdays<-1 stop for A review. This dataset contains no genuine missing values; the numeric median imputer is retained to match the common pipeline specification.'), code((ROOT/'src/d_reference_pipeline.py').read_text(encoding='utf-8')),
    md('## 4. Unified metrics and original groups\n\nAP is average precision, not trapezoidal PR-AUC. Brier=mean((p-y)^2). Recall=TP/(TP+FN), FPR=FP/(FP+TN), F1=2TP/(2TP+FP+FN). NaN means undefined denominator; AP is undefined without positives. Small sample: n<100 or positives<20 or negatives<20. Brier and FPR are lower-is-better. Recall changes multiplied by 100 are percentage points.'),code((ROOT/'src/d_evaluation.py').read_text(encoding='utf-8')),
    md('## 5. Reproduce all four validation experiments\n\nOnly the 60% train partition is fitted. Each probability retains the exact frozen validation row_id order. Training contains 27,126 rows / 3,173 positives; validation contains 9,042 rows / 1,058 positives.'),
    code(textwrap.dedent('''\
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        X_train, y_train = load_partition('train')
        X_val, y_val = load_partition('val')
        rows, group_rows, probability_frames, fitted = [], [], [], {}
        for experiment, (s,u) in EXPERIMENTS.items():
            pipe = build_pipeline(s,u)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                pipe.fit(X_train, y_train)
            probability = pipe.predict_proba(X_val)[:,1]
            rows.append(dict(experiment=experiment, **evaluate(y_val, probability)))
            group_rows.extend(subgroup_metrics(X_val,y_val,probability,experiment))
            probability_frames.append(pd.DataFrame({'row_id':X_val.index,'experiment':experiment,'y_true':y_val.to_numpy(),'probability':probability}))
            fitted[experiment] = pipe
        validation = pd.DataFrame(rows)
        subgroups = pd.DataFrame(group_rows)
        probabilities = pd.concat(probability_frames, ignore_index=True)
        expected = pd.read_csv(ROOT/'outputs/D/validation/metrics.csv')
        np.testing.assert_allclose(validation[METRICS], expected[METRICS], rtol=1e-10, atol=1e-12)
        display(validation[['experiment','n','n_positive','AP','Recall','F1','Brier','Accuracy','TP','FP','TN','FN']])
        ''')),
    md('## 6. References and all five contrasts\n\nAll-no accuracy is 88.299%, so roughly 89.4% model accuracy should not hide low positive recall. The constant training-prior probability is an extra no-feature probability reference.'),
    code("reference_rows=[]\nfor label,p in [('all_no',0.),('training_prior',float(y_train.mean()))]:\n    reference_rows.append(dict(reference=label, **evaluate(y_val,np.full(len(y_val),p))))\ndisplay(pd.DataFrame(reference_rows))\ndisplay(contrasts(validation))"),
    md('## 7. Subgroup audit\n\nInformation groups use original job/education values. Each dimension partitions validation independently. Rows from age and information dimensions overlap and must not be added together.'),
    code("display(subgroups[['experiment','dimension','group','n','n_positive','n_negative','TP','FP','TN','FN','Recall','FPR','Brier','small_sample']])\ngroup_deltas=contrasts(subgroups,['dimension','group'])\ndisplay(group_deltas[group_deltas.effect.eq('full')])"),
    md('## 8. Evidence charts (generated from reproduced values)'),
    code(textwrap.dedent('''\
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,3,figsize=(12,3.2),layout='constrained')
        for ax,metric in zip(axes,['AP','Recall','Brier']):
            ax.plot(validation.experiment,validation[metric],marker='o')
            ax.set_title(metric+(' (lower better)' if metric=='Brier' else ' (higher better)'))
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        fig.suptitle('Validation only; zoomed axes; no confidence intervals')
        plt.show()
        ''')),
    md('## 9. Interpretation and limits\n\nS increases overall AP by about 0.00099 (U off) or 0.00088 (U on), with very small Brier changes. U has mixed AP effects and slightly worse Brier under both S settings. ESU-E0: AP +0.000905, Recall -0.0945 percentage points, F1 -0.001306, Brier +0.00000465. These are small descriptive changes, with no significance claim.\n\nAt age>=60, ESU identifies 28/128 positives vs E0’s 29/128: recall drops 0.78125 percentage points. Unknown-information Brier worsens by 0.00012293 while recall stays 10/46. Overall AP can improve while some groups incur a cost. None of the displayed validation groups triggers the manual’s small-sample warning, which does not establish statistical precision.\n\nPossible redundancy with previous/poutcome awaits B’s train-only evidence. S changes representation, feature count and regularization geometry. Brier is not a stand-alone calibration test. No causal, discrimination, customer-independent, time-generalization or financial-return claim. Retaining unknown does not automatically improve performance. A semantics improvement can remain defensible even if scores barely change.'),
    md('## 10. Final-test status and handoff\n\n**Not executed.** Obtain B’s training-only discovery tables and real group/interpretation review; obtain C’s accepted build_pipeline and boundary/leakage tests. See notes/D_HANDOFF.md. Revalidate the accepted source, fill the real handoff receipt, freeze, then run scripts/run_d.py test once. Final fits must use 36,168 train+val rows; test contains 9,043 rows. Do not copy validation scores into test columns. The final CLI computes all four tests, all references, confusion counts and subgroup contrasts, without choosing a test winner.'),
    code("final_test_status = {'test_evaluated': False, 'B_review': 'pending', 'C_review': 'pending', 'D_human_review': 'pending'}\nprint(final_test_status)"),
    md('## 11. GenAI and contribution declaration\n\nUser identifies LIANG SHUANG as D. OpenAI Codex assisted with code, experiment execution, metric checks, figures and explanation drafts. Human understanding, real review signatures, B/C contributions, continuous four-person video and course submission remain to be completed by the group. No human review is claimed on anyone’s behalf.\n\nSources: local execution manual pp6–10; [A repository](https://github.com/raederhans/ca6003-bank-marketing) commit b4c4ff1be75a6e48baacd920f405982b20399352; [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing); [AP](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html); [Brier](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.brier_score_loss.html); [fit boundaries](https://scikit-learn.org/1.8/common_pitfalls.html).')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb, ROOT/'notebooks/D_evaluation.ipynb')


def report():
    pdfmetrics.registerFont(TTFont('CJK', 'C:/Windows/Fonts/simhei.ttf'))
    navy,teal=colors.HexColor('#19394c'),colors.HexColor('#087d84')
    styles={
        'title':ParagraphStyle('title',fontName='CJK',fontSize=24,leading=32,textColor=navy,spaceAfter=12,wordWrap='CJK'),
        'h':ParagraphStyle('h',fontName='CJK',fontSize=15,leading=21,textColor=teal,spaceBefore=12,spaceAfter=8,wordWrap='CJK'),
        'p':ParagraphStyle('p',fontName='CJK',fontSize=10,leading=16,spaceAfter=8,wordWrap='CJK'),
        'small':ParagraphStyle('small',fontName='CJK',fontSize=8.5,leading=12,spaceAfter=6,wordWrap='CJK'),
        'cell':ParagraphStyle('cell',fontName='CJK',fontSize=8,leading=12,wordWrap='CJK'),
    }
    story=[]
    def p(t,kind='p'): story.append(Paragraph(t,styles[kind]))
    def heading(k,title):
        p(f'D4 / LIANG SHUANG / 2026.10.15 / {k}','small');p(title,'title')
    def tab(headers,rows,widths):
        white=ParagraphStyle('header',parent=styles['cell'],textColor=colors.white)
        data=[[Paragraph(html.escape(str(v)),white if i==0 else styles['cell']) for v in row] for i,row in enumerate([headers]+rows)]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef4f6'),colors.white]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),1,teal)]))
        story.append(t);story.append(Spacer(1,9))
    heading('01','四组对照评估：当前可交付成果')
    p('负责人 D：LIANG SHUANG','h')
    p('<b>状态：验证实验已完成；最终测试等待 B/C 交接。</b> 本报告使用 A 的固定数据与划分，并由 D 按手册第6页临时补齐管线。结果可复现，但 C 尚未确认实现，B 尚未复核解释，不能称为全组最终结论。')
    p('研究问题：特殊值语义与未知信息处理如何影响同一逻辑回归的可靠性？不调参，不改变阈值，不增加复杂模型。')
    tab(['实验','S：联系历史表示','U：job/education unknown'],[
        ['E0','原始 pdays，保留 -1','训练集众数填补'],['ES','状态标记 + 条件间隔','训练集众数填补'],
        ['EU','原始 pdays，保留 -1','独立类别'],['ESU','状态标记 + 条件间隔','独立类别']],[45,210,256])
    p('实际验证结果','h')
    tab(['实验','AP ↑','Recall ↑','F1 ↑','Brier ↓','Accuracy'],[[e]+[f'{table.loc[e,m]:.6f}' for m in ['AP','Recall','F1','Brier','Accuracy']] for e in table.index],[45,90,90,90,96,100])
    p('train：27,126 行、3,173 正例；validation：9,042 行、1,058 正例、7,984 负例。阈值 p≥0.5。四组输入样本完全相同，编码后维数依次为 31 / 32 / 33 / 34；全部收敛，无收敛警告。','small')
    p('核心结论：语义更清楚，不等于所有指标都更好。','h')
    p('S 的 AP 改善很小；U 的方向不稳定。ESU 相对 E0 的 AP 增加 0.000905，但 Recall 减少 0.0945 个百分点、Brier 增加约 0.00000465。没有不确定性区间，不能称为显著改善。')
    p('本次环境：Python 3.12.14，scikit-learn 1.8.0，numpy 2.5.4，pandas 3.0.6。A 的原环境锁保留；D 的实测锁见 requirements.D.lock.txt。','small')
    story.append(PageBreak())
    heading('02','每一步的影响，全部保留')
    p('逐项对照','h')
    tab(['对照','ΔAP','ΔRecall（百分点）','ΔF1','ΔBrier'],[[r.contrast,f'{r.delta_AP:+.8f}',f'{r.delta_Recall*100:+.5f}',f'{r.delta_F1:+.6f}',f'{r.delta_Brier:+.8f}'] for r in delta.itertuples()],[90,100,116,100,105])
    p('Δ 均为前者减后者。AP / Recall / F1 越大越好，Brier 越小越好；不把不同指标单位混为“提升百分比”。S 有两种 U 条件下的比较，U 也有两种 S 条件下的比较，避免把总变化全归于一个步骤。','small')
    story.append(Image(str(FIG/'validation_overall.png'),width=511,height=146))
    p('无特征参考与阈值背景','h')
    p('全部预测 no：Accuracy=88.2990%，Recall=0，AP=0.117010，Brier=0.117010。训练正例率恒定概率参考：p=0.116973，Brier=0.103318；0.5 阈值下同样全预测 no。四组 AP 与 Brier 优于这些参考，但 Recall 仅约17%，仍会漏掉大部分正例。')
    p('混淆矩阵计数','h')
    tab(['实验','TP','FP','TN','FN'],[[e]+[int(table.loc[e,k]) for k in ['TP','FP','TN','FN']] for e in table.index],[75,109,109,109,109])
    p('AP 是 average_precision_score，不是梯形 PR-AUC。Brier 是 mean((p-y)²)，不单独证明校准改善。分母为0时指标记为 NaN；无正例时 AP 记为无定义。保存了逐行概率，可独立复算。','small')
    story.append(PageBreak())
    heading('03','整体平均之外的客群代价')
    p('手册预定分组；以下均为验证集','h')
    e0=groups[groups.experiment.eq('E0')].set_index(['dimension','group'])
    esu=groups[groups.experiment.eq('ESU')].set_index(['dimension','group'])
    labels={'<30':'年龄 <30','30-59':'年龄 30-59','>=60':'年龄 ≥60','any_unknown':'任一 unknown','both_known':'两列均已知'}
    rows=[]
    for idx,r in e0.iterrows():
        z=esu.loc[idx]
        rows.append([labels[idx[1]],f'{int(r.n)} / {int(r.n_positive)} / {int(r.n_negative)}',f'{r.Recall:.6f} → {z.Recall:.6f}',f'{r.FPR:.6f} → {z.FPR:.6f}',f'{r.Brier:.6f} → {z.Brier:.6f}'])
    tab(['分组','n / 正例 / 负例','Recall E0→ESU','FPR E0→ESU','Brier E0→ESU'],rows,[78,115,106,106,106])
    p('年龄与信息完整性是两套分组，互相重叠；不可相加。完整性由原始 job 或 education 是否包含文字 unknown 定义，不使用填补后的值。所有四组的完整组表和差值见 CSV。','small')
    story.append(Image(str(FIG/'validation_subgroup_changes.png'),width=511,height=180))
    p('需要在结论中保留的变化','h')
    p('≥60 岁：E0 识别 29/128 个正例，ESU 为 28/128，少1个；Recall 下降0.78125个百分点，Brier 增加0.00014935。任一 unknown 组：两方案均识别10/46个正例，Recall 不变，但 Brier 增加0.00012293。')
    p('本次各组均未触发 n<100 或正/负例<20 的展示警告；这不等于结果精确或差异显著。没有客户级独立性、时间外验证或歧视判定依据。解释须与 B 的训练集证据核对。','small')
    story.append(PageBreak())
    heading('04','向负责人 B 获取的材料')
    p('目的：把模型变化接回真实的训练集发现，避免凭分数猜测数据机制。','p')
    tab(['必须获取','验收内容','D 的用途'],[
      ['核心发现 + 可重建证据','train-only Notebook/代码；至少1张交叉表或分层表；分区、n、正例数、分母；≤4主图','解释 S/U 决策的观察依据；A 审计不能替代 B 独立发现'],
      ['历史联系交叉表','pdays=-1 / ≥0 × previous=0 / >0 × poutcome；例外原始 row_id；解释和未知原因','核对 S 的语义；讨论 previous/poutcome 的潜在冗余'],
      ['unknown 分层证据','按年龄或联系状态的 job/education unknown 比例及认购率；总体/分层比较','解释 U 和组内代价；没有反转证据时不称辛普森悖论'],
      ['提前固定分组','确认年龄 <30、30-59、≥60；原始两列任一 unknown / 均已知','保证组别不是看过测试后挑选；若修改须先重新确认协议'],
      ['解释人工复核','对全部改善/恶化/无变化提出书面意见；真实姓名与日期','形成最终结论和视频；不得用自动化代签']
    ],[100,226,185])
    p('建议索取的文件','h')
    p('B_discovery.ipynb；B_history_crosstab.csv；B_unknown_strata.csv；B_findings.md；B_groups_review.md。名称可以不同，必须拿到代码和原始表，不能只有截图或口头结论。')
    p('可以直接复制给 B 的交接请求','h')
    p('“我是 LIANG SHUANG，负责 D。请给我仅基于固定 train 的核心发现、历史联系三字段交叉表、unknown 的分层表，以及每张图的 n/正例数/分母和可复现代码。请确认手册预定分组，并复核我四组验证结果中‘改善很小、部分客群变差’的解释。请附真实复核姓名和日期，之后再冻结并执行最终测试。”')
    p('以上是草稿，未代你联系或发送给任何成员。','small')
    story.append(PageBreak())
    heading('05','向负责人 C 获取的材料')
    tab(['必须获取','验收内容','D 的用途'],[
      ['正式 build_pipeline','build_pipeline(use_s,use_u)；fit / predict_proba / 特征名接口；11列原始输入；同一固定 LR','替换临时模块或正式采纳并审阅 D 的参考实现'],
      ['S 转换测试','-1、0、正数区分；原 pdays 移除；<-1 / 真缺失停止核查；状态标记0/1','排除把未联系当作当天联系；共同缩放约定一致'],
      ['U 与拟合边界测试','只改 job/education unknown；poutcome 不改；众数/中位数/缩放/OHE 只在当前 fit 数据学得','避免泄漏；四组行数、顺序、固定 split 不变'],
      ['版本与证据','源文件/提交号、四组配置、测试代码和日志、依赖锁、编码后特征名与维数','在 D 环境复现；不能仅交全表 fit 后的矩阵'],
      ['最终 Notebook 装配','合并 A/B/C/D 关键函数、结果、引用和 GenAI 声明；保留待办和真实贡献','C 总装；A 新内核 Run All；D 校对所有图文数字']
    ],[100,226,185])
    p('可以直接复制给 C 的交接请求','h')
    p('“我是 LIANG SHUANG，负责 D。请交正式 build_pipeline(use_s,use_u) 模块、四组配置和 S/U 边界及无泄漏测试。D 已按手册做参考实现并完成验证，你可以审阅采纳或替换；请明确状态标记的缩放约定、全部学习参数的 fit 分区和真实复核记录。我会使用接受后的管线重新验证，再冻结最终测试。请将 D_evaluation.ipynb 的核心单元并入全组 Notebook。”')
    p('收齐之后运行','h')
    p('1. 接受/替换管线并重跑 validate、QA；2. 填真实 d_handoff_receipt.json，附 B/C 证据路径；3. 运行 freeze；4. test 一次性重拟合 train+val 并报告四组测试；5. 更新结论并由 B 复核。')
    p('命令：.venv/Scripts/python.exe scripts/run_d.py validate<br/>.venv/Scripts/python.exe scripts/run_d.py freeze --receipt config/d_handoff_receipt.json<br/>.venv/Scripts/python.exe scripts/run_d.py test','small')
    p('冻结记录保存源码、数据、split、环境和验证证据哈希；测试独占标记阻止重复执行。若测试中途故障，保留原始记录供复核，不自动删除标记或重试。','small')
    story.append(PageBreak())
    heading('06','D 的讲解主线与最终核对')
    p('视频 07:50–10:35：约 2 分 45 秒','h')
    p('当前是验证版讲稿。最终视频应改用正式测试结果并明确 train+val 重拟合；不能把当前验证数值改称测试数值。英文完整稿与阶段提示见 notes/D_VIDEO_SCRIPT.md。')
    for title,body in [
        ('先交代公平对照','四组同一模型、同一数据划分、同一0.5阈值。只比较 S/U 两项准备规则；先验证，后冻结与测试。'),
        ('整体结果','AP 约0.3585至0.3595。S 有很小排序改善；U 的效果混合，不能承诺提升。完整准备并未同时改善所有指标。'),
        ('背景参考','全部预测 no 的 Accuracy 已达88.30%。模型约89.4%的 Accuracy 不应掩盖只有17%左右的 Recall。'),
        ('分组代价','≥60岁组 ESU 比 E0 少识别1个正例。原始信息含unknown组的Recall不变，Brier稍差。保留变差结果，说明总体平均可能掩盖局部代价。'),
        ('解释与限制','语义正确性与可解释性仍有价值；已有历史变量可能冗余，待B证据支持。无显著性、因果、未来泛化或收益承诺。')]:
        p(title,'h');p(body)
    p('交付与诚信核对','h')
    p('已提供：D Notebook、源码、四组验证 CSV、逐行概率、5项增量、分组表、系数、检查与讲稿。未完成的依赖：B/C 材料与真实签字、最终测试、C 全组整合、A 新内核复核、四人视频和课程实际提交。')
    p('GenAI：OpenAI Codex 辅助代码、运行、核算与文字组织。用户声明 LIANG SHUANG 为 D；未声称任何学生已经人工核验。需要学生实际理解、审阅和录制。','small')
    p('自动化：A 的10项数据治理检查、D 的边界/指标/拟合边界/封版检查均通过；D Notebook 新内核9个代码单元运行成功，无错误。检查报告见 outputs/D/。','small')
    p('依据：本地执行手册第6-10页；A仓库 commit b4c4ff1be75a6e48baacd920f405982b20399352；scikit-learn 1.8 AP / Brier / Common pitfalls 官方文档。详见 notes/D_HANDOFF.md 和 Notebook 的可点击引用。','small')
    dest=ROOT/'output/pdf/LIANG_SHUANG_D_Evaluation_Report.pdf'
    dest.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setStrokeColor(teal);canvas.line(42,43,553,43)
        canvas.setFont('CJK',8);canvas.setFillColor(navy)
        canvas.drawString(42,29,'CA6003 / 验证版 / B、C 交接待完成 / test 未评价')
        canvas.drawRightString(553,29,str(doc.page))
    SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=56).build(story,onFirstPage=footer,onLaterPages=footer)


def speech():
    text='''# LIANG SHUANG / D · video narration draft

Status: VALIDATION ONLY. Target 07:50–10:35, around 2m45s. Rehearse at your natural speed. Replace the numbers with frozen final-test results before recording the group’s final video. Do not rename validation results as test results.

## 0:00–0:30 / Experiment design

I am Liang Shuang, responsible for the controlled evaluation. We compare four preparation settings using the same logistic regression, fixed data split and point-five threshold. S separates the never-contacted state from elapsed days. U keeps unknown job and education values as categories instead of replacing them with training modes. Everything else stays fixed.

## 0:30–1:10 / Overall results and contrasts

These are our current validation results, fitted on the training partition. Average precision ranges from about point-three-five-eight-five to point-three-five-nine-five. S gives a small ranking improvement under both U settings. U has mixed effects and slightly worsens the Brier score under both S settings. Compared with the baseline, the combined setting improves average precision by about point-zero-zero-zero-nine, but recall decreases by about point-zero-nine-five percentage points. We report all five planned contrasts, including deterioration, rather than selecting only a favourable comparison.

## 1:10–1:35 / Reference and threshold

Accuracy needs context: predicting no for everyone already achieves eighty-eight-point-three percent. Our model accuracy is around eighty-nine-point-four percent, while recall is only about seventeen percent. Better ranking and probability scores therefore do not mean that the fixed threshold identifies most subscribers. We did not tune that threshold after seeing the results.

## 1:35–2:15 / Group costs

We also use the manual’s predefined age and information groups. Among people aged sixty or above, the combined setting detects twenty-eight of one hundred and twenty-eight positives, compared with twenty-nine for the baseline. In the original unknown-information group, recall stays at ten of forty-six, but Brier becomes slightly worse. These small changes show why an overall average cannot replace subgroup reporting.

## 2:15–2:45 / Interpretation and status

Clearer semantics remain useful even when predictive gains are small. Existing history variables may already encode similar information, but that explanation still needs B’s training evidence. We make no significance or causal claim. The dataset cannot establish customer independence or future performance. After B and C review the evidence and pipeline, we will freeze the rules, refit on training plus validation, and evaluate all four settings on the sealed test partition once.

## 中文理解提纲

四组公平对照 → S改善很小、U方向混合 → Accuracy要与全no基线比较 → ≥60岁与unknown组存在局部代价 → 语义正确性不保证涨分 → 收齐B/C后才做最终测试。

最终版本操作：将“current validation”改为真实最终测试阶段的描述，并从 outputs/D/test 的表重新取全部数值；改写最后一段为已实际完成的流程。先交 B 核解释，再由 D 与表格逐项校对。每位成员必须实际出镜并讲解；稿件不代表视频已完成。
'''
    (ROOT/'notes/D_VIDEO_SCRIPT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    figures();report()
    if not args.report_only:
        notebook();speech()
    print('Built D artifacts (executed notebook preserved with --report-only).')
