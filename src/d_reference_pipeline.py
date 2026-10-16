"""D-authored implementation of manual p6; provisional until C reviews/replaces it."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['age', 'job', 'marital', 'education', 'default', 'balance',
            'housing', 'loan', 'pdays', 'previous', 'poutcome']
CATEGORICAL = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']


class SemanticFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, use_s=False, use_u=False):
        self.use_s = use_s
        self.use_u = use_u

    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or not set(FEATURES).issubset(X.columns):
            raise ValueError('Expected raw DataFrame with all 11 allowed fields')
        Z = X.loc[:, FEATURES].copy()  # Never use labels, tracking, or campaign fields.
        if Z.isna().any().any() or (Z['pdays'] < -1).any():
            raise ValueError('Unexpected missing value / pdays < -1: return to A')
        if self.use_s:
            Z['has_previous_contact'] = Z['pdays'].ge(0).astype(float)
            Z['days_since_previous'] = Z['pdays'].where(Z['pdays'].ge(0), 0)
            Z = Z.drop(columns='pdays')
        if not self.use_u:
            for field in ['job', 'education']:
                Z[field] = Z[field].astype(object).replace('unknown', np.nan)
        return Z

    def get_feature_names_out(self, input_features=None):
        fields = [c for c in FEATURES if not (self.use_s and c == 'pdays')]
        return np.asarray(fields + (['has_previous_contact', 'days_since_previous'] if self.use_s else []), dtype=object)


def build_pipeline(use_s: bool, use_u: bool):
    numeric = ['age', 'balance', 'previous'] + (['days_since_previous'] if use_s else ['pdays'])
    numeric_pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                             ('scale', StandardScaler())])
    # Only job and education contain synthetic NaN under U=off; poutcome is untouched.
    categorical_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                 ('encode', OneHotEncoder(handle_unknown='ignore', drop=None))])
    transforms = [('numeric', numeric_pipe, numeric), ('categorical', categorical_pipe, CATEGORICAL)]
    if use_s:
        transforms.append(('status', 'passthrough', ['has_previous_contact']))
    return Pipeline([
        ('semantics', SemanticFeatures(use_s, use_u)),
        ('preprocess', ColumnTransformer(transforms, remainder='drop')),
        ('model', LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000,
                                     class_weight=None, random_state=42)),
    ])
