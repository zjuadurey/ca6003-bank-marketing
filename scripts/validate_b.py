"""Independent count reconstruction from CSV, not a repetition of B's groupby implementation."""
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src import b_discovery
from src.data_contract import load_partition


def main():
    calls = []
    def train_only(name, **kwargs):
        calls.append(name)
        assert name == 'train', 'B attempted to access holdout'
        return load_partition(name, **kwargs)
    with patch.object(b_discovery, 'load_partition', side_effect=train_only):
        tables = b_discovery.investigate()
    assert calls == ['train']
    ids = json.loads((ROOT/'data/splits/split_indices.json').read_text())
    with (ROOT/'data/raw/bank-full.csv').open() as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    history, unknown = Counter(), Counter()
    for i in ids['train']:
        r = rows[i]
        h = 'never_contacted' if int(r['pdays']) < 0 else 'previously_contacted'
        prev = 'zero' if int(r['previous']) == 0 else 'positive'
        history[h, prev, r['poutcome']] += 1
        age = '<30' if int(r['age'])<30 else ('30-59' if int(r['age'])<60 else '>=60')
        info = 'unknown' if r['job']=='unknown' or r['education']=='unknown' else 'known'
        unknown[age, info, r['y']] += 1
    for r in tables['history_crosstab'].itertuples():
        assert r.n == history[r.history_group, r.previous_group, r.poutcome]
    for r in tables['unknown_strata'].query("field == 'any_unknown' and dimension == 'age_group'").itertuples():
        assert r.n_positive == unknown[r.stratum,r.information,'yes']
        assert r.n_negative == unknown[r.stratum,r.information,'no']
    assert tables['training_row_ids'].row_id.tolist() == ids['train']
    assert not set(tables['history_exceptions'].row_id) & (set(ids['val']) | set(ids['test']))
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, pd.read_csv(ROOT/f'outputs/B/{name}.csv'), check_dtype=False, atol=1e-12)
    report = {'passed': True, 'checks': ['train-only loader', 'original CSV history counts',
        'original CSV age/information positive and negative counts', 'exact frozen training order',
        'exception rows exclude holdouts', 'all saved tables reproduce'], 'human_review':'pending'}
    (ROOT/'outputs/B/validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
