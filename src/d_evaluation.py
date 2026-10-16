"""Shared evaluation, no model selection or holdout-dependent choices."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

EXPERIMENTS = {'E0': (False, False), 'ES': (True, False),
               'EU': (False, True), 'ESU': (True, True)}
THRESHOLD = 0.5
METRICS = ['AP', 'Recall', 'F1', 'Brier', 'Accuracy', 'FPR']
CONTRASTS = [('S_Uoff', 'ES', 'E0'), ('S_Uon', 'ESU', 'EU'),
             ('U_Soff', 'EU', 'E0'), ('U_Son', 'ESU', 'ES'),
             ('full', 'ESU', 'E0')]


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float('nan')


def evaluate(y, probability):
    y = np.asarray(y)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Labels/probabilities must be aligned binary 1-D arrays')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Non-finite or out-of-range probability')
    pred = p >= THRESHOLD
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    n = len(y)
    return dict(n=n, n_positive=tp+fn, n_negative=tn+fp, TP=tp, FP=fp, TN=tn, FN=fn,
                AP=float(average_precision_score(y, p)) if tp+fn else float('nan'),
                Recall=ratio(tp, tp+fn), FPR=ratio(fp, fp+tn),
                Precision=ratio(tp, tp+fp), F1=ratio(2*tp, 2*tp+fp+fn),
                Brier=float(np.mean((p-y)**2)) if n else float('nan'),
                Accuracy=ratio(tp+tn, n), threshold=THRESHOLD)


def group_labels(X):
    # Raw pre-imputation information status, never recomputed from encoded features.
    age = pd.Series(np.where(X.age < 30, '<30', np.where(X.age < 60, '30-59', '>=60')), index=X.index)
    missing = X[['job', 'education']].eq('unknown').any(axis=1)
    return {'age': age, 'information': pd.Series(np.where(missing, 'any_unknown', 'both_known'), index=X.index)}


def subgroup_metrics(X, y, p, experiment):
    rows = []
    for dimension, labels in group_labels(X).items():
        levels = ['<30', '30-59', '>=60'] if dimension == 'age' else ['any_unknown', 'both_known']
        for label in levels:
            mask = labels.eq(label).to_numpy()
            row = evaluate(np.asarray(y)[mask], np.asarray(p)[mask])
            row.update(experiment=experiment, dimension=dimension, group=label,
                       small_sample=bool(row['n'] < 100 or row['n_positive'] < 20 or row['n_negative'] < 20))
            rows.append(row)
    return rows


def contrasts(table, group_columns=()):
    rows = []
    iterator = table.groupby(list(group_columns), sort=False) if group_columns else [((), table)]
    for keys, frame in iterator:
        keys = keys if isinstance(keys, tuple) else (keys,)
        indexed = frame.set_index('experiment')
        for effect, after, before in CONTRASTS:
            row = dict(zip(group_columns, keys))
            row.update(effect=effect, contrast=after+'-'+before)
            for metric in METRICS:
                row['delta_'+metric] = indexed.loc[after, metric] - indexed.loc[before, metric]
            if group_columns:
                row['n'] = int(indexed.loc[after, 'n'])
                row['small_sample'] = bool(indexed.loc[after, 'small_sample'])
            rows.append(row)
    return pd.DataFrame(rows)
