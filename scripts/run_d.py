"""D experiment runner. Validation is open; freeze/test require real handoff receipts."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_contract import load_partition
from src.d_evaluation import EXPERIMENTS, evaluate, subgroup_metrics, contrasts

OUT = ROOT / 'outputs/D'
CONFIG = ROOT / 'config/d_protocol.json'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn']}}


def fingerprint():
    protocol = json.loads(CONFIG.read_text(encoding='utf-8'))
    module = importlib.import_module(protocol['pipeline_module'])
    paths = ['config/d_protocol.json', 'config/dataset_lock.json', 'data/splits/split_indices.json',
             'data/splits/split_manifest.csv', 'data/raw/bank-full.csv', 'src/data_contract.py',
             'src/d_evaluation.py', 'scripts/run_d.py']
    paths += [str(Path(module.__file__).relative_to(ROOT)).replace('\\', '/')]
    return {'files': {p: digest(ROOT/p) for p in paths}, 'environment': environment()}


def run_stage(stage):
    protocol = json.loads(CONFIG.read_text(encoding='utf-8'))
    factory = importlib.import_module(protocol['pipeline_module']).build_pipeline
    Xtr, ytr = load_partition('train')
    Xval, yval = load_partition('val')
    if stage == 'test':
        # Called only after freeze verification and exclusive test-start marker.
        Xfit, yfit = pd.concat([Xtr, Xval]), pd.concat([ytr, yval])
        Xev, yev = load_partition('test', allow_test=True)
    else:
        Xfit, yfit, Xev, yev = Xtr, ytr, Xval, yval
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=True)
    all_rows, groups, predictions, coefficients, fit_records = [], [], [], [], []
    for experiment, (use_s, use_u) in EXPERIMENTS.items():
        pipe = factory(use_s=use_s, use_u=use_u)
        # Enforce fixed estimator even when C supplies a replacement factory.
        estimator = pipe.steps[-1][1]
        for key, value in protocol['model'].items():
            if estimator.get_params().get(key) != value:
                raise ValueError(f'Fixed model mismatch: {key}')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            pipe.fit(Xfit, yfit)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError('ConvergenceWarning: inspect scaling; do not report silently')
        classes = np.asarray(pipe.classes_)
        if not np.array_equal(classes, [0, 1]):
            raise ValueError('Expected classes [0,1]')
        p = pipe.predict_proba(Xev)[:, 1]
        row = evaluate(yev, p)
        row.update(experiment=experiment, stage=stage, fit_n=len(Xfit), fit_n_positive=int(yfit.sum()))
        all_rows.append(row)
        groups.extend(subgroup_metrics(Xev, yev, p, experiment))
        predictions.append(pd.DataFrame({'row_id': Xev.index, 'y_true': yev.to_numpy(),
                                         'experiment': experiment, 'probability': p, 'prediction': (p >= .5).astype(int)}))
        names = pipe[:-1].get_feature_names_out()
        for feature, coef in zip(names, estimator.coef_[0]):
            coefficients.append({'experiment': experiment, 'feature': feature, 'coefficient': float(coef)})
        fit_records.append({'experiment': experiment, 'use_s': use_s, 'use_u': use_u,
                            'fit_n': len(Xfit), 'fit_n_positive': int(yfit.sum()),
                            'evaluation_n': len(Xev), 'feature_count': len(names),
                            'n_iter': estimator.n_iter_.tolist(),
                            'warnings': [str(w.message) for w in caught]})
    table, group_table = pd.DataFrame(all_rows), pd.DataFrame(groups)
    baseline_rows = []
    for name, value in [('all_no', 0.), ('training_prior', float(yfit.mean()))]:
        row = evaluate(yev, np.full(len(yev), value))
        row.update(reference=name, fit_n=len(Xfit), probability=value)
        baseline_rows.append(row)
    table.to_csv(folder/'metrics.csv', index=False)
    pd.DataFrame(baseline_rows).to_csv(folder/'references.csv', index=False)
    group_table.to_csv(folder/'subgroups.csv', index=False)
    contrasts(table).to_csv(folder/'contrasts.csv', index=False)
    contrasts(group_table, ['dimension', 'group']).to_csv(folder/'subgroup_contrasts.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(folder/'predictions.csv', index=False, float_format='%.17g')
    pd.DataFrame(coefficients).to_csv(folder/'coefficients.csv', index=False)
    write_json(folder/'run_manifest.json', {'stage': stage, 'time_utc': datetime.now(timezone.utc).isoformat(),
               'status': protocol['status'] if stage == 'validation' else 'final_frozen_protocol',
               'fingerprint': fingerprint(), 'fits': fit_records,
               'undefined_values': 'NaN in tables = denominator zero; AP undefined if no positives',
               'test_loaded': stage == 'test'})
    print(table[['experiment', 'n', 'n_positive', 'AP', 'Recall', 'F1', 'Brier', 'Accuracy']].to_string(index=False))
    return table


def freeze(receipt_path, *, user_authorized=False):
    if (OUT/'freeze.json').exists():
        raise ValueError('Protocol already frozen; retain original record')
    receipt = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    required = ['B_reviewed_by', 'C_reviewed_by', 'D_reviewed_by', 'reviewed_at']
    if user_authorized:
        # Explicit task-owner direction may authorize execution, but never impersonates
        # the separate human reviewers or changes the scientific protocol.
        if (receipt.get('execution_authorization') != 'explicit_user_instruction'
                or not receipt.get('authorization_quote')
                or receipt.get('human_review') != 'pending'):
            raise ValueError('Explicit user authorization and truthful pending review status required')
    elif any(not receipt.get(k) or str(receipt[k]).lower() == 'pending' for k in required):
        raise ValueError('Real B/C/D review records required; do not invent signatures')
    if receipt.get('groups_match_manual') is not True or receipt.get('pipeline_accepted') is not True:
        raise ValueError('Manual group definitions and pipeline must be accepted')
    evidence = receipt.get('evidence_files', [])
    if len(evidence) < 2:
        raise ValueError('Provide B evidence and C transformation-test evidence files')
    evidence_hashes = {p: digest(ROOT/p) for p in evidence}
    manifest = json.loads((OUT/'validation/run_manifest.json').read_text(encoding='utf-8'))
    if manifest['fingerprint'] != fingerprint():
        raise ValueError('Code/config/environment changed since validation; rerun validation first')
    write_json(OUT/'freeze.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
               'fingerprint': fingerprint(), 'receipt': receipt, 'evidence_hashes': evidence_hashes,
               'validation_hashes': {p.name: digest(p) for p in sorted((OUT/'validation').glob('*')) if p.is_file()}})
    print('Frozen. All four experiments will be reported; no test-dependent model selection.')


def final_test():
    frozen = json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))
    if frozen['fingerprint'] != fingerprint():
        raise ValueError('Frozen source/config/environment changed')
    for p, h in frozen['evidence_hashes'].items():
        if digest(ROOT/p) != h:
            raise ValueError('Reviewed evidence changed')
    for name, h in frozen['validation_hashes'].items():
        if digest(OUT/'validation'/name) != h:
            raise ValueError('Frozen validation evidence changed')
    # Exclusive creation refuses repeated testing, including after partial failure.
    marker = OUT/'test_started.json'
    with marker.open('x', encoding='utf-8') as stream:
        json.dump({'started_utc': datetime.now(timezone.utc).isoformat(), 'freeze_sha256': digest(OUT/'freeze.json')}, stream)
    run_stage('test')
    write_json(OUT/'test_completed.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
               'freeze_sha256': digest(OUT/'freeze.json'),
               'test_hashes': {p.name: digest(p) for p in sorted((OUT/'test').glob('*')) if p.is_file()}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['validate', 'freeze', 'test'])
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--user-authorized', action='store_true',
                        help='Execute under explicit task-owner authorization; does not certify human review')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'validate':
        if (OUT/'freeze.json').exists():
            raise ValueError('Validation is frozen; do not overwrite')
        run_stage('validation')
    elif args.stage == 'freeze':
        if not args.receipt:
            parser.error('--receipt is required')
        freeze(args.receipt, user_authorized=args.user_authorized)
    else:
        final_test()


if __name__ == '__main__':
    main()
