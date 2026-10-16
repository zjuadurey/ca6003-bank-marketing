"""Exercise the explicit-user execution path with synthetic receipts, never real holdout data."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_d


class AuthorizationTests(unittest.TestCase):
    def test_explicit_authorization_does_not_impersonate_reviews(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'outputs';(out/'validation').mkdir(parents=True)
            (root/'b.csv').write_text('synthetic B evidence')
            (root/'c.json').write_text('synthetic C evidence')
            (out/'validation/run_manifest.json').write_text(json.dumps({'fingerprint':{'synthetic':True}}))
            receipt={'B_reviewed_by':'pending','C_reviewed_by':'pending','D_reviewed_by':'pending',
                'reviewed_at':'pending','execution_authorization':'explicit_user_instruction',
                'authorization_quote':'synthetic test authorization','human_review':'pending',
                'groups_match_manual':True,'pipeline_accepted':True,'evidence_files':['b.csv','c.json']}
            path=root/'receipt.json';path.write_text(json.dumps(receipt))
            with patch.object(run_d,'ROOT',root),patch.object(run_d,'OUT',out),patch.object(run_d,'fingerprint',return_value={'synthetic':True}):
                with self.assertRaisesRegex(ValueError,'Real B/C/D'):
                    run_d.freeze(path)
                run_d.freeze(path,user_authorized=True)
                frozen=json.loads((out/'freeze.json').read_text())
                self.assertEqual(frozen['receipt']['B_reviewed_by'],'pending')
                self.assertEqual(frozen['receipt']['human_review'],'pending')
                self.assertEqual(set(frozen['evidence_hashes']),{'b.csv','c.json'})
                with self.assertRaisesRegex(ValueError,'already frozen'):
                    run_d.freeze(path,user_authorized=True)

    def test_missing_explicit_record_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'receipt.json';p.write_text('{}')
            with patch.object(run_d,'OUT',root):
                with self.assertRaisesRegex(ValueError,'Explicit user authorization'):
                    run_d.freeze(p,user_authorized=True)
            self.assertFalse((root/'freeze.json').exists())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuthorizationTests))
    report={'run':result.testsRun,'passed':result.wasSuccessful(),'real_test_accessed':False}
    (ROOT/'outputs/integration/authorization_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
