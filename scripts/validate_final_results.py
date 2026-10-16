"""Verify frozen final artifacts and independently recompute metrics using sklearn."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, recall_score, accuracy_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_d import fingerprint


def main():
    folder=ROOT/'outputs/D'
    frozen=json.loads((folder/'freeze.json').read_text())
    completed=json.loads((folder/'test_completed.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert completed['freeze_sha256']==digest(folder/'freeze.json')
    assert frozen['fingerprint']==fingerprint()
    for p,h in frozen['evidence_hashes'].items():assert digest(ROOT/p)==h,p
    for p,h in frozen['validation_hashes'].items():assert digest(folder/'validation'/p)==h,p
    for p,h in completed['test_hashes'].items():assert digest(folder/'test'/p)==h,p
    pred=pd.read_csv(folder/'test/predictions.csv',float_precision='round_trip')
    metrics=pd.read_csv(folder/'test/metrics.csv').set_index('experiment')
    ids=json.loads((ROOT/'data/splits/split_indices.json').read_text())['test']
    assert metrics.index.tolist()==['E0','ES','EU','ESU']
    assert metrics.fit_n.eq(36168).all() and metrics.n.eq(9043).all()
    for name,p in pred.groupby('experiment',sort=False):
        assert p.row_id.tolist()==ids
        y,prob=p.y_true,p.probability
        guess=prob>=.5
        tn,fp,fn,tp=confusion_matrix(y,guess,labels=[0,1]).ravel()
        expected={'AP':average_precision_score(y,prob),'Brier':brier_score_loss(y,prob),
                  'Recall':recall_score(y,guess),'F1':f1_score(y,guess),'Accuracy':accuracy_score(y,guess),
                  'TP':tp,'FP':fp,'TN':tn,'FN':fn,'FPR':fp/(fp+tn)}
        for key,value in expected.items():np.testing.assert_allclose(value,metrics.loc[name,key],atol=1e-12,rtol=0)
    report={'passed':True,'test_model_refit':False,'checks':['freeze fingerprint','evidence hashes',
        'validation hashes','test artifact hashes','fixed row order and counts','independent sklearn metrics'],
        'human_review':'pending'}
    (ROOT/'outputs/integration/final_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
