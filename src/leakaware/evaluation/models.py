"""Learners used in the experiments. All preprocessing is fitted on the training set of each fold or origin."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from xgboost import XGBClassifier


def _slog(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


class XGBModel:
    """XGBoost with native categorical inputs (fixed category vocabulary, built without labels)."""
    def __init__(self, num, cat, vocab, seed=42, n_jobs=2, **params):
        self.num, self.cat, self.vocab = num, cat, vocab
        self.clf = XGBClassifier(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
                                 eval_metric="logloss", random_state=seed, n_jobs=n_jobs, **params)

    def _X(self, df):
        X = df[self.num].astype(float).copy()
        for c in self.cat:
            X[c] = pd.Categorical(df[c], categories=self.vocab[c])
        return X

    def fit(self, df, y):
        self.clf.fit(self._X(df), y)
        return self

    def predict_proba(self, df):
        return self.clf.predict_proba(self._X(df))[:, 1]


class SKModel:
    def __init__(self, num, cat, kind="lr", seed=42, **params):
        self.num, self.cat = num, cat
        if kind == "lr":
            numt = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                             ("log", FunctionTransformer(_slog)), ("sc", StandardScaler())])
            catt = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)
            est = LogisticRegression(C=params.get("C", 0.5), max_iter=3000)
        else:  # rf
            numt = SimpleImputer(strategy="median", add_indicator=True)
            catt = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            est = RandomForestClassifier(n_estimators=params.get("n_estimators", 200), max_depth=params.get("max_depth", 12),
                                         min_samples_leaf=params.get("min_samples_leaf", 20),
                                         max_features=params.get("max_features", "sqrt"),
                                         max_samples=params.get("max_samples", 0.5), n_jobs=2, random_state=seed)
        self.pipe = Pipeline([("pre", ColumnTransformer([("n", numt, num), ("c", catt, cat)])), ("m", est)])

    def fit(self, df, y):
        self.pipe.fit(df[self.num + self.cat], y)
        return self

    def predict_proba(self, df):
        return self.pipe.predict_proba(df[self.num + self.cat])[:, 1]


def make_model(name, num, cat, vocab, cfg, **override):
    if name == "xgb":
        p = dict(cfg["models"]["xgb"]); p.update(override)
        return XGBModel(num, cat, vocab, seed=cfg["seed"], **p)
    if name == "lr":
        p = dict(cfg["models"]["lr"]); p.update(override)
        return SKModel(num, cat, "lr", seed=cfg["seed"], **p)
    if name == "rf":
        p = dict(cfg["models"].get("rf", {})); p.update(override)
        return SKModel(num, cat, "rf", seed=cfg["seed"], **p)
    raise ValueError(name)
