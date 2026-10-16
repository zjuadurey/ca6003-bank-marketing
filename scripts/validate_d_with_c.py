"""Meaningful edge, leakage, alignment and holdout-gate checks; never fit on test."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / '.local' / 'test-temp'
TEMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TEMP)
sys.path.insert(0, str(ROOT))
from src.data_contract import load_partition
from src.preprocessing import SemanticTransformer as SemanticFeatures, build_pipeline
from src.d_evaluation import evaluate, group_labels, contrasts
from scripts import run_d


class DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_partition('train')

    def test_sentinel_zero_and_positive_are_distinct(self):
        X = self.X.iloc[:3].copy()
        X['pdays'] = [-1, 0, 15]
        Z = SemanticFeatures(True, True).fit_transform(X)
        self.assertEqual(Z.has_previous_contact.tolist(), [0, 1, 1])
        self.assertEqual(Z.days_since_previous.tolist(), [0, 0, 15])
        self.assertNotIn('pdays', Z)
        pd.testing.assert_series_equal(Z.poutcome, X.poutcome)

    def test_bad_semantics_stop_and_forbidden_columns_do_not_enter(self):
        X = self.X.iloc[:3].assign(y=1, row_id=123, duration=999)
        Z = SemanticFeatures().fit_transform(X)
        self.assertFalse({'y', 'row_id', 'duration'} & set(Z.columns))
        X.loc[X.index[0], 'pdays'] = -2
        with self.assertRaises(ValueError):
            SemanticFeatures().fit_transform(X)
        X.loc[X.index[0], 'pdays'] = np.nan
        with self.assertRaises(ValueError):
            SemanticFeatures().fit_transform(X)

    def test_fit_only_learning_and_unknown_category(self):
        X, y = self.X.iloc[:100].copy(), self.y.iloc[:100]
        X['job'] = ['admin.']*75 + ['unknown']*25
        X['education'] = ['primary']*75 + ['unknown']*25
        pipe = build_pipeline(False, False).fit(X, y)
        prep = pipe.named_steps['preprocess']
        cat = prep.named_transformers_['job_education']
        self.assertEqual(cat.named_steps['imputer'].statistics_[0], 'admin.')
        self.assertEqual(cat.named_steps['imputer'].statistics_[1], 'primary')
        scale = prep.named_transformers_['numeric'].named_steps['scaler']
        np.testing.assert_allclose(scale.mean_, X[['age','balance','previous','pdays']].mean())
        old_mean = scale.mean_.copy()
        future = X.iloc[:3].copy()
        future['balance'] = 1000000000
        future['job'] = 'unseen_job'
        p = pipe.predict_proba(future)[:, 1]
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_array_equal(scale.mean_, old_mean)

    def test_u_only_changes_two_fields_and_raw_groups_survive(self):
        X = self.X.iloc[:3].copy()
        X['job'], X['education'], X['poutcome'] = 'unknown', 'unknown', 'unknown'
        Z = SemanticFeatures(False, False).fit(self.X).transform(X)
        self.assertTrue(Z[['job','education']].isna().all().all())
        self.assertTrue(Z.poutcome.eq('unknown').all())
        self.assertTrue(group_labels(X)['information'].eq('any_unknown').all())

    def test_metrics_and_threshold_match_independent_reference(self):
        y, p = [0, 0, 1, 1], np.asarray([.1, .5, .8, .4])
        m = evaluate(y, p)
        self.assertEqual([m[k] for k in ['TN','FP','FN','TP']], confusion_matrix(y, p>=.5).ravel().tolist())
        self.assertEqual(m['AP'], average_precision_score(y, p))
        self.assertAlmostEqual(m['Brier'], brier_score_loss(y, p))
        self.assertAlmostEqual(m['Recall'], .5)
        self.assertAlmostEqual(m['F1'], .5)
        for p_bad in [[float('nan')]*4, [1.1]*4]:
            with self.assertRaises(ValueError):
                evaluate(y, p_bad)

    def test_zero_denominator_is_undefined_not_perfect(self):
        m = evaluate([0,0], [0,0])
        self.assertTrue(np.isnan(m['AP']) and np.isnan(m['Recall']) and np.isnan(m['F1']))
        self.assertTrue(np.isnan(evaluate([1,1], [.8,.9])['FPR']))
        self.assertEqual(evaluate([1,0], [0,0])['F1'], 0)

    def test_age_boundaries(self):
        X = self.X.iloc[:4].copy()
        X['age'] = [29,30,59,60]
        self.assertEqual(group_labels(X)['age'].tolist(), ['<30','30-59','30-59','>=60'])

    def test_pending_receipt_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run_d, 'OUT', Path(temp)):
            with self.assertRaisesRegex(ValueError, 'Real B/C/D'):
                run_d.freeze(ROOT/'config/d_handoff_receipt.template.json')
            self.assertFalse((Path(temp)/'freeze.json').exists())

    def test_test_cannot_run_without_freeze_or_load_test(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run_d, 'OUT', Path(temp)), patch.object(run_d, 'load_partition') as loader:
            with self.assertRaises(FileNotFoundError):
                run_d.final_test()
            loader.assert_not_called()

    def test_test_stage_refits_train_plus_val_with_synthetic_holdout(self):
        # All rows here are slices of TRAIN; no real test labels are read.
        partitions = {'train': (self.X.iloc[:200], self.y.iloc[:200]),
                      'val': (self.X.iloc[200:300], self.y.iloc[200:300]),
                      'test': (self.X.iloc[300:400], self.y.iloc[300:400])}
        def synthetic_loader(name, **kwargs):
            if name == 'test':
                self.assertIs(kwargs.get('allow_test'), True)
            return partitions[name]
        with tempfile.TemporaryDirectory() as temp, patch.object(run_d, 'OUT', Path(temp)), patch.object(run_d, 'load_partition', side_effect=synthetic_loader), patch.object(run_d, 'fingerprint', return_value={}):
            table = run_d.run_stage('test')
            self.assertTrue(table.fit_n.eq(300).all())
            self.assertTrue(table.n.eq(100).all())
            self.assertEqual(table.experiment.tolist(), ['E0','ES','EU','ESU'])

    def test_final_stage_verifies_freeze_and_is_single_use(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run_d, 'OUT', Path(temp)), patch.object(run_d, 'fingerprint', return_value={'synthetic':'unit-test'}), patch.object(run_d, 'run_stage') as stage:
            folder = Path(temp)
            run_d.write_json(folder/'freeze.json', {'fingerprint': {'synthetic':'unit-test'}, 'evidence_hashes':{}, 'validation_hashes':{}})
            run_d.final_test()
            stage.assert_called_once_with('test')
            self.assertTrue((folder/'test_completed.json').exists())
            with self.assertRaises(FileExistsError):
                run_d.final_test()
            self.assertEqual(stage.call_count,1)
            with patch.object(run_d,'fingerprint',return_value={'synthetic':'changed'}):
                with self.assertRaisesRegex(ValueError,'changed'):
                    run_d.final_test()

    def test_saved_results_recalculate_from_predictions(self):
        folder = ROOT/'outputs/D/validation'
        if not (folder/'predictions.csv').exists():
            self.skipTest('Run validation experiments first')
        predictions = pd.read_csv(folder/'predictions.csv', float_precision='round_trip')
        table = pd.read_csv(folder/'metrics.csv').set_index('experiment')
        groups = pd.read_csv(folder/'subgroups.csv')
        _, yval = load_partition('val')
        for exp, pred in predictions.groupby('experiment', sort=False):
            self.assertEqual(pred.row_id.tolist(), yval.index.tolist())
            self.assertEqual(pred.y_true.tolist(), yval.tolist())
            actual = evaluate(pred.y_true, pred.probability)
            for key in ['AP','Recall','F1','Brier','Accuracy','TP','FP','TN','FN']:
                self.assertAlmostEqual(actual[key], table.loc[exp,key], places=12)
            for _, frame in groups[groups.experiment.eq(exp)].groupby('dimension'):
                self.assertEqual(int(frame.n.sum()), len(yval))
                self.assertEqual(int(frame.n_positive.sum()), int(yval.sum()))
        expected = contrasts(table.reset_index())
        pd.testing.assert_frame_equal(expected, pd.read_csv(folder/'contrasts.csv'), check_exact=False, atol=1e-12)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DTests))
    report = {'run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'status': 'passed' if result.wasSuccessful() else 'failed',
              'test_partition_loaded': False}
    run_d.write_json(ROOT/'outputs/D/qa.json', report)
    sys.exit(0 if result.wasSuccessful() else 1)
