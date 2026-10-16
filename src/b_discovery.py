"""B: descriptive investigation restricted to A's frozen training partition."""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_contract import ROOT, load_partition


def summarize(frame, keys):
    result = frame.groupby(keys, observed=True, dropna=False)['target'].agg(
        n='size', n_positive='sum').reset_index()
    result['n_negative'] = result.n - result.n_positive
    result['subscription_rate'] = result.n_positive / result.n
    result['small_sample'] = (result.n < 100) | (result.n_positive < 20) | (result.n_negative < 20)
    result['partition'] = 'train'
    return result


def investigate():
    X, y = load_partition('train')
    f = X.copy()
    f['target'] = y
    f['age_group'] = np.where(f.age < 30, '<30', np.where(f.age < 60, '30-59', '>=60'))
    f['history_group'] = np.where(f.pdays < 0, 'never_contacted', 'previously_contacted')
    f['previous_group'] = np.where(f.previous == 0, 'zero', 'positive')
    history = summarize(f, ['history_group', 'previous_group', 'poutcome'])
    history['share_of_train'] = history.n / len(f)
    exceptional = f.loc[(f.pdays.ge(0) & f.poutcome.eq('unknown')) |
                        (f.pdays.ge(0) != f.previous.gt(0))].copy()
    exceptional['reason'] = np.where(exceptional.pdays.ge(0) != exceptional.previous.gt(0),
        'pdays_previous_disagreement', 'contact_recorded_outcome_unknown')
    exceptional['interpretation'] = 'Observed combination; original cause is not identifiable; retain unchanged'

    overview = []
    for label, mask in [('target_yes', y.eq(1)), ('job_unknown', f.job.eq('unknown')),
                        ('education_unknown', f.education.eq('unknown')),
                        ('any_unknown', f.job.eq('unknown') | f.education.eq('unknown')),
                        ('pdays_minus_one', f.pdays.eq(-1)), ('pdays_nonnegative', f.pdays.ge(0))]:
        overview.append(dict(measure=label, n=int(mask.sum()), denominator=len(f),
                             proportion=float(mask.mean()), n_positive=int(y[mask].sum()), partition='train'))
    unknown_tables = []
    for field in ['job', 'education', 'any_unknown']:
        mask = (f.job.eq('unknown') | f.education.eq('unknown')) if field == 'any_unknown' else f[field].eq('unknown')
        for dimension in ['overall', 'age_group', 'history_group']:
            g = f[['target', 'age_group', 'history_group']].copy()
            g['stratum'] = 'all' if dimension == 'overall' else g[dimension]
            g['information'] = np.where(mask, 'unknown', 'known')
            table = summarize(g, ['stratum', 'information'])
            table['field'] = field
            table['dimension'] = dimension
            table['stratum_n'] = table.groupby('stratum').n.transform('sum')
            table['share_in_stratum'] = table.n / table.stratum_n
            unknown_tables.append(table)
    unknown = pd.concat(unknown_tables, ignore_index=True)
    differences = []
    for (field, dimension, stratum), group in unknown.groupby(['field', 'dimension', 'stratum'], sort=False):
        indexed = group.set_index('information')
        if {'unknown', 'known'} <= set(indexed.index):
            differences.append(dict(field=field, dimension=dimension, stratum=stratum,
                unknown_minus_known_pp=100*(indexed.loc['unknown', 'subscription_rate']-indexed.loc['known', 'subscription_rate']),
                unknown_n=int(indexed.loc['unknown','n']), known_n=int(indexed.loc['known','n']),
                small_sample=bool(indexed.small_sample.any()), partition='train'))

    # Direct standardization describes age composition; it is not causal adjustment.
    weights = f.age_group.value_counts(normalize=True)
    standardized = []
    for field in ['job', 'education', 'any_unknown']:
        part = unknown[(unknown.field == field) & (unknown.dimension == 'age_group')]
        for info in ['known', 'unknown']:
            rates = part[part.information == info].set_index('stratum').subscription_rate.reindex(weights.index)
            standardized.append(dict(field=field, information=info,
                age_standardized_rate=float((rates*weights).sum()) if rates.notna().all() else np.nan,
                weights='pooled training age distribution', partition='train'))

    quality, tails = [], []
    for field in ['balance', 'previous']:
        s = f[field]
        quality.append(dict(field=field, n=len(s), negative_n=int(s.lt(0).sum()), zero_n=int(s.eq(0).sum()),
            minimum=int(s.min()), q01=float(s.quantile(.01)), q25=float(s.quantile(.25)),
            median=float(s.median()), q75=float(s.quantile(.75)), q95=float(s.quantile(.95)),
            q99=float(s.quantile(.99)), maximum=int(s.max()), partition='train'))
        selected = pd.concat([f.nsmallest(5, field), f.nlargest(5, field)]).drop_duplicates()
        selected = selected.reset_index()[['row_id', 'age', 'balance', 'previous', 'pdays', 'poutcome', 'target']]
        selected['selected_by'] = field + '_five_lowest_and_highest'
        selected['partition'] = 'train'
        tails.append(selected)
    assert history.n.sum() == len(f)
    assert history.n_positive.sum() == int(y.sum())
    assert all(t.n.sum() == len(f) for _, t in unknown.groupby(['field','dimension']))
    return {'overview': pd.DataFrame(overview), 'history_crosstab': history,
            'history_exceptions': exceptional.reset_index(), 'unknown_strata': unknown,
            'unknown_differences': pd.DataFrame(differences), 'age_standardized': pd.DataFrame(standardized),
            'numeric_quality': pd.DataFrame(quality), 'numeric_tail_examples': pd.concat(tails, ignore_index=True),
            'training_row_ids': pd.DataFrame({'row_id': X.index}),
            'age_weights': weights.rename_axis('age_group').rename('weight').reset_index()}


def draw_figures(tables, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140})
    overview = tables['overview']
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    ax.barh(overview.measure, overview.proportion*100, color='#28758b')
    for i, row in overview.iterrows():
        ax.text(row.proportion*100+.5, i, f'{row.n:,}/{row.denominator:,} ({row.proportion:.1%})', va='center', fontsize=9)
    ax.set(xlim=(0,115), xlabel='Percent of fixed training partition', title='Special states are different kinds of information (train only)')
    fig.savefig(folder/'01_overview.png'); plt.close(fig)

    h = tables['history_crosstab']
    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    labels = h.history_group + ' / ' + h.previous_group + ' / ' + h.poutcome
    ax.barh(labels, h.n, color='#28758b')
    ax.set_xscale('log')
    for i, row in h.iterrows():
        ax.text(row.n*1.12, i, f'n={row.n:,}; yes={row.n_positive:,}', va='center', fontsize=9)
    ax.set(xlim=(1,1000000), xlabel='Record count (log scale); training n=27,126',
           title='History status overlaps across fields; five contacted records still have unknown outcome')
    fig.savefig(folder/'02_history.png'); plt.close(fig)

    u = tables['unknown_strata']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    ages = ['<30','30-59','>=60']
    age = u[u.dimension.eq('age_group')]
    for field, offset, color in [('job',-.24,'#28758b'),('education',0,'#bd8831'),('any_unknown',.24,'#785e9c')]:
        part = age[age.field.eq(field) & age.information.eq('unknown')].set_index('stratum').reindex(ages)
        axes[0].bar(np.arange(3)+offset, part.share_in_stratum*100, width=.23, label=field, color=color)
    denominators = age[age.field.eq('any_unknown') & age.information.eq('unknown')].set_index('stratum').stratum_n
    age_labels = [f'{label}\nn={int(denominators[label]):,}' for label in ages]
    axes[0].set(xticks=np.arange(3), xticklabels=age_labels, ylabel='Unknown / age-stratum n (%)', title='Information completeness varies with age')
    axes[0].legend(fontsize=8)
    for info, offset, color in [('known',-.19,'#28758b'),('unknown',.19,'#bd8831')]:
        part = age[age.field.eq('any_unknown') & age.information.eq(info)].set_index('stratum').reindex(ages)
        axes[1].bar(np.arange(3)+offset, part.subscription_rate*100, width=.36, label=info, color=color)
        for i, (_, row) in enumerate(part.iterrows()):
            axes[1].text(i+offset,row.subscription_rate*100+.7,f'{int(row.n_positive)}/{int(row.n)}'+('*' if row.small_sample else ''),
                         ha='center', fontsize=8, rotation=20)
    axes[1].set(xticks=np.arange(3), xticklabels=ages, ylim=(0,44), ylabel='Yes / information-group n (%)', title='Overall unknown association is not uniform by age')
    axes[1].legend(fontsize=8)
    fig.supxlabel('Train only; * small sample (n<100 or either outcome count<20). Descriptive, not causal.')
    fig.savefig(folder/'03_unknown_age.png'); plt.close(fig)


def write_outputs(root=ROOT):
    tables = investigate()
    out = Path(root)/'outputs/B'
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out/f'{name}.csv', index=False)
    draw_figures(tables, out/'figures')
    report = {'partition': 'train', 'n': len(tables['training_row_ids']),
        'n_positive': int(tables['history_crosstab'].n_positive.sum()),
        'exploratory_holdout_access': False, 'human_review': 'pending',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
        'inputs_sha256': {p:hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in
             ['data/raw/bank-full.csv','data/splits/split_indices.json','data/splits/split_manifest.csv']}}
    (out/'run_manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return tables


if __name__ == '__main__':
    write_outputs()
