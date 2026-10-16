"""Locally rebuild C evidence and compare canonical features against old D source."""
from pathlib import Path
import json
import sys
import platform
import importlib.metadata
import hashlib
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_contract import load_partition
from src.preprocessing import build_pipeline, EXPERIMENTS
from src.d_reference_pipeline import build_pipeline as reference_pipeline


def dense(matrix):
    return matrix.toarray() if hasattr(matrix,'toarray') else np.asarray(matrix)


def names(pipe):
    result=[n.split('__',1)[-1] for n in pipe.get_feature_names_out()]
    if len(set(result))!=len(result):
        raise ValueError('Canonical feature names must be unique')
    return result


def main():
    X,y=load_partition('train')
    V,yv=load_partition('val')
    out=ROOT/'outputs/C'
    feature_rows,check_rows,fit_stats,migration=[],[],{},[]
    history=ROOT/'outputs/D/history/reference_20261015/validation/predictions.csv'
    old=pd.read_csv(history,float_precision='round_trip')
    current=pd.read_csv(ROOT/'outputs/D/validation/predictions.csv',float_precision='round_trip')
    for exp,(s,u) in EXPERIMENTS.items():
        c=build_pipeline(s,u)
        ref=reference_pipeline(s,u)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            c.fit(X,y); ref.fit(X,y)
        assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        assert c.named_steps['semantic'].transform(X).index.equals(X.index)
        ref_names=[n.split('__',1)[-1] for n in ref[:-1].get_feature_names_out()]
        c_names=names(c)
        order=[ref_names.index(n) for n in c_names]
        cz=dense(c[:-1].transform(V))
        rz=dense(ref[:-1].transform(V))[:,order]
        np.testing.assert_allclose(cz,rz,rtol=0,atol=1e-12)
        assert not any(any(name==f or name.startswith(f+'_') for f in ['duration','campaign','contact','day','month','row_id','y']) for name in c_names)
        p=c.predict_proba(V)[:,1]
        saved=current[current.experiment.eq(exp)]
        past=old[old.experiment.eq(exp)]
        assert saved.row_id.tolist()==V.index.tolist()==past.row_id.tolist()
        np.testing.assert_allclose(p,saved.probability,rtol=1e-12,atol=1e-14)
        for i,name in enumerate(c.get_feature_names_out()):
            feature_rows.append(dict(experiment=exp,position=i,feature=name))
        numeric=c.named_steps['preprocess'].named_transformers_['numeric']
        je=c.named_steps['preprocess'].named_transformers_['job_education']
        fit_stats[exp]={'fit_n':len(X),'fit_n_positive':int(y.sum()),
            'numeric_medians':numeric.named_steps['imputer'].statistics_.tolist(),
            'numeric_means':numeric.named_steps['scaler'].mean_.tolist(),
            'numeric_scales':numeric.named_steps['scaler'].scale_.tolist(),
            'job_education_modes':None if u else je.named_steps['imputer'].statistics_.tolist(),
            'n_iter':c.named_steps['model'].n_iter_.tolist()}
        check_rows.append(dict(experiment=exp,use_s=s,use_u=u,fit_n=len(X),evaluation_n=len(V),feature_count=len(c_names),n_iter=int(c.named_steps['model'].n_iter_[0]),warnings=len(caught),canonical_design_matches_reference=True))
        migration.append(dict(experiment=exp,design_matrix_max_abs_difference=float(np.abs(cz-rz).max()),
            probability_max_abs_difference_vs_historical_D=float(np.abs(p-past.probability.to_numpy()).max()),
            changed_predictions_vs_historical_D=int(((p>=.5)!=(past.probability.to_numpy()>=.5)).sum()),
            c_source_sha256=hashlib.sha256((ROOT/'src/preprocessing.py').read_bytes()).hexdigest()))
    pd.DataFrame(feature_rows).to_csv(out/'feature_names.csv',index=False)
    pd.DataFrame(check_rows).to_csv(out/'pipeline_checks.csv',index=False)
    pd.DataFrame(migration).to_csv(out/'migration_comparison.csv',index=False)
    (out/'fit_statistics.json').write_text(json.dumps(fit_stats,indent=2)+'\n',encoding='utf-8')
    env={'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','nbclient','nbformat','ipykernel']},
         'role':'D local rerun of user-supplied C source','real_test_evaluated':False,'human_review':'pending'}
    (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(check_rows).to_string(index=False))
    print(pd.DataFrame(migration).drop(columns='c_source_sha256').to_string(index=False))


if __name__=='__main__':
    main()
