"""C 转换验收；合成边界和仅训练分区样本，不打开最终测试分区。"""
import io
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.data_contract import FEATURES, load_partition
from src.preprocessing import EXPERIMENTS, SemanticTransformer, build_pipeline

ROOT = Path(__file__).resolve().parent


class CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = load_partition('train')
        cls.X, cls.y = X.iloc[:200].copy(), y.iloc[:200].copy()

    def test_s_boundaries_and_index(self):
        X = self.X.iloc[:3].copy()
        X['pdays'] = [-1, 0, 15]
        Z = SemanticTransformer(True, True).fit_transform(X)
        self.assertEqual(Z.has_previous_contact.tolist(), [0, 1, 1])
        self.assertEqual(Z.days_since_previous.tolist(), [0, 0, 15])
        self.assertNotIn('pdays', Z)
        self.assertTrue(Z.index.equals(X.index))
        pd.testing.assert_frame_equal(Z.drop(columns=['has_previous_contact', 'days_since_previous']), X.drop(columns='pdays'))

    def test_s_off_keeps_sentinel(self):
        X = self.X.iloc[:3].copy()
        X['pdays'] = [-1, 0, 15]
        pd.testing.assert_frame_equal(SemanticTransformer(False, True).fit_transform(X), X)

    def test_u_only_changes_two_columns(self):
        X = self.X.iloc[:3].copy()
        X[['job', 'education', 'poutcome']] = 'unknown'
        Z = SemanticTransformer(False, False).fit(self.X).transform(X)
        self.assertTrue(Z[['job', 'education']].isna().all().all())
        pd.testing.assert_frame_equal(Z.drop(columns=['job', 'education']), X.drop(columns=['job', 'education']))
        pd.testing.assert_frame_equal(SemanticTransformer(False, True).fit_transform(X), X)

    def test_invalid_pdays_and_actual_missing_stop(self):
        for value in [-2, np.nan, np.inf, .5]:
            for s, u in EXPERIMENTS.values():
                with self.subTest(value=value, s=s, u=u):
                    X = self.X.copy()
                    X['pdays'] = X.pdays.astype(float)
                    X.loc[X.index[0], 'pdays'] = value
                    with self.assertRaises(ValueError):
                        build_pipeline(s, u).fit(X, self.y)
        X = self.X.copy()
        X.loc[X.index[0], 'job'] = None
        with self.assertRaisesRegex(ValueError, 'actual missing'):
            build_pipeline(False, False).fit(X, self.y)

    def test_whitelist_and_input_immutable(self):
        original = self.X.copy(deep=True)
        extended = self.X.assign(y=1, row_id=123, duration=999, campaign=999, contact='x', day=1, month='jan')
        for s, u in EXPERIMENTS.values():
            p = build_pipeline(s, u).fit(extended, self.y)
            self.assertFalse(any(any(n.endswith('__'+f) or n.startswith('categorical__'+f+'_') for f in ['y','row_id','duration','campaign','contact','day','month']) for n in p.get_feature_names_out()))
            np.testing.assert_allclose(p.predict_proba(extended), p.predict_proba(self.X))
        pd.testing.assert_frame_equal(original, self.X)

    def test_fit_statistics_do_not_learn_from_future(self):
        X = self.X.copy()
        X['job'] = ['admin.']*150 + ['unknown']*50
        X['education'] = ['primary']*150 + ['unknown']*50
        p = build_pipeline(False, False).fit(X, self.y)
        prep = p.named_steps['preprocess']
        cat = prep.named_transformers_['job_education']
        self.assertEqual(cat.named_steps['imputer'].statistics_.tolist(), ['admin.', 'primary'])
        numeric = prep.named_transformers_['numeric']
        cols = ['age','balance','previous','pdays']
        np.testing.assert_allclose(numeric.named_steps['imputer'].statistics_, X[cols].median())
        np.testing.assert_allclose(numeric.named_steps['scaler'].mean_, X[cols].mean())
        before = joblib.hash(p)
        future = X.iloc[:10].copy()
        future['balance'] = 1e9
        future['job'] = 'unseen_job'
        future['education'] = 'unseen_education'
        future['poutcome'] = 'unseen_outcome'
        result = p.predict_proba(future)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(before, joblib.hash(p))
        self.assertFalse(any('unseen' in n for n in p.get_feature_names_out()))

    def test_status_flag_is_unscaled_and_pdays_removed(self):
        X = self.X.copy()
        X.loc[X.index[:3], 'pdays'] = [-1, 0, 15]
        p = build_pipeline(True, True).fit(X, self.y)
        Z = p[:-1].transform(X)
        Z = Z.toarray() if hasattr(Z, 'toarray') else Z
        names = p.get_feature_names_out().tolist()
        self.assertFalse(any('pdays' in n for n in names))
        np.testing.assert_array_equal(Z[:, names.index('status__has_previous_contact')], (X.pdays >= 0).astype(int))

    def test_unknown_levels_and_poutcome_preserved(self):
        X = self.X.copy()
        X.loc[X.index[:3], ['job','education','poutcome']] = 'unknown'
        for use_u in [False, True]:
            names = build_pipeline(False, use_u).fit(X, self.y).get_feature_names_out().tolist()
            self.assertEqual('job_education__job_unknown' in names, use_u)
            self.assertEqual('job_education__education_unknown' in names, use_u)
            self.assertIn('categorical__poutcome_unknown', names)

    def test_four_configs_fixed_model_and_clone(self):
        for s, u in EXPERIMENTS.values():
            p = clone(build_pipeline(s, u))
            params = p.named_steps['model'].get_params()
            for k, v in dict(C=1.0, solver='lbfgs', max_iter=2000, class_weight=None, random_state=42).items():
                self.assertEqual(params[k], v)
            p.fit(self.X, self.y)
            result = p.predict_proba(self.X)
            self.assertEqual(result.shape, (len(self.X), 2))
            self.assertTrue(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all())
            np.testing.assert_allclose(result.sum(axis=1), 1)
            self.assertEqual(len(p.get_feature_names_out()), p.named_steps['model'].coef_.shape[1])

    def test_pickle_round_trip(self):
        p = build_pipeline(True, True).fit(self.X, self.y)
        stream = io.BytesIO()
        joblib.dump(p, stream)
        stream.seek(0)
        np.testing.assert_allclose(p.predict_proba(self.X), joblib.load(stream).predict_proba(self.X))

    def test_all_unknown_fit_fails_without_inventing_mode(self):
        X = self.X.assign(job='unknown')
        with self.assertRaisesRegex(ValueError, 'no observed'):
            build_pipeline(False, False).fit(X, self.y)
        build_pipeline(False, True).fit(X, self.y)

    def test_schema_and_unfitted_errors(self):
        with self.assertRaises(NotFittedError):
            SemanticTransformer().transform(self.X)
        with self.assertRaises(ValueError):
            build_pipeline(False, False).fit(self.X.drop(columns='age'), self.y)
        with self.assertRaises(TypeError):
            build_pipeline('False', False)
        with self.assertRaises(ValueError):
            SemanticTransformer().fit(pd.concat([self.X, self.X[['age']]], axis=1))

    def test_negative_balance_and_tail_are_retained(self):
        X = self.X.copy()
        X.loc[X.index[0], 'balance'] = -100000
        X.loc[X.index[1], 'previous'] = 1000
        Z = SemanticTransformer(True, True).fit_transform(X)
        pd.testing.assert_series_equal(Z.balance, X.balance)
        pd.testing.assert_series_equal(Z.previous, X.previous)

    def test_real_test_loader_remains_sealed(self):
        with self.assertRaisesRegex(ValueError, 'sealed'):
            load_partition('test')


def run_tests():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CTests))
    folder = ROOT / 'outputs/C'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'tests.log').write_text(stream.getvalue(), encoding='utf-8')
    report = dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), passed=result.wasSuccessful(), real_test_partition_loaded=False)
    (folder/'tests.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(stream.getvalue())
    return result.wasSuccessful()


if __name__ == '__main__':
    raise SystemExit(0 if run_tests() else 1)
