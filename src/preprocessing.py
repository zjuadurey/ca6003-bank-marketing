"""C / D3：手册第 6 页的正式 S/U 管线；不读取任何分区。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data_contract import FEATURES

EXPERIMENTS = {"E0": (False, False), "ES": (True, False),
               "EU": (False, True), "ESU": (True, True)}
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "poutcome"]


class SemanticTransformer(TransformerMixin, BaseEstimator):
    """白名单、原始异常检查和不学习参数的语义转换；保留 index。"""

    def __init__(self, use_s=False, use_u=False):
        self.use_s = use_s
        self.use_u = use_u

    def _prepare(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame with named raw columns")
        if X.columns.duplicated().any() or not set(FEATURES).issubset(X.columns):
            raise ValueError("Missing or duplicate input columns")
        frame = X.loc[:, FEATURES].copy(deep=True)
        # 真空值属于上游 STOP；U 产生的 NaN 在此检查之后出现。
        if frame.isna().any().any():
            raise ValueError("STOP: actual missing values; ask A to verify raw data")
        for name in ["age", "balance", "pdays", "previous"]:
            if not pd.api.types.is_numeric_dtype(frame[name]) or not np.isfinite(frame[name]).all():
                raise ValueError(f"STOP: invalid numeric column {name}")
        if (frame.pdays < -1).any() or (frame.pdays % 1 != 0).any():
            raise ValueError("STOP: invalid pdays; expected -1 or nonnegative integer days")
        for name in CATEGORICAL:
            if not frame[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"STOP: invalid category in {name}")
        if self.use_s:
            frame["has_previous_contact"] = (frame.pdays >= 0).astype("int64")
            frame["days_since_previous"] = frame.pdays.where(frame.pdays >= 0, 0)
            frame = frame.drop(columns="pdays")
        if not self.use_u:
            for name in ["job", "education"]:
                frame[name] = frame[name].astype(object).replace("unknown", np.nan)
        return frame

    def fit(self, X, y=None):
        frame = self._prepare(X)
        if not self.use_u and frame[["job", "education"]].isna().all().any():
            raise ValueError("STOP: no observed job/education mode in fit data")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        return self._prepare(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


class MarketingPipeline(Pipeline):
    """完整分类器也提供编码特征名；标准 Pipeline 的末端模型无此接口。"""

    def get_feature_names_out(self, input_features=None):
        steps = self.steps[:-1] if self.steps and self.steps[-1][0] == 'model' else self.steps
        return Pipeline(steps).get_feature_names_out(input_features)


def build_pipeline(use_s: bool, use_u: bool):
    """每次返回未拟合的新对象，供 D 在当前 fit 分区独立训练。"""
    if type(use_s) is not bool or type(use_u) is not bool:
        raise TypeError("use_s and use_u must be bool")
    numeric = ["age", "balance", "previous", "days_since_previous" if use_s else "pdays"]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    unknown_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None)),
    ])
    other_categories = [x for x in CATEGORICAL if x not in ["job", "education"]]
    transformers = [
        ("numeric", numeric_pipe, numeric),
        ("job_education", OneHotEncoder(handle_unknown="ignore", drop=None) if use_u else unknown_pipe,
         ["job", "education"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", drop=None), other_categories),
    ]
    if use_s:
        transformers.append(("status", "passthrough", ["has_previous_contact"]))
    return MarketingPipeline([
        ("semantic", SemanticTransformer(use_s=use_s, use_u=use_u)),
        ("preprocess", ColumnTransformer(transformers, remainder="drop")),
        ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                     class_weight=None, random_state=42)),
    ])
