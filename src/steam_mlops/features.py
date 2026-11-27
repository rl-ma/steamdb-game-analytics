"""Feature engineering and the scikit-learn pipeline.

The model takes one row per game in a simple *input schema* (INPUT_COLUMNS) that a user can
fill in on a web form. ``make_features`` turns that input into model features, and it is the
first step of the pipeline, so training and serving always use exactly the same code.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from steam_mlops.config import LEAKY_CATEGORIES, SOCIAL_CATEGORIES

# ---------------------------------------------------------------- input schema
# What the API (and the frontend form) sends for one game.
INPUT_COLUMNS = [
    "name",  # str: game title (used only for the sequel flag)
    "release_date",  # str: "YYYY-MM-DD"
    "regular_price",  # float: price in USD before any discount (0 = free)
    "required_age",  # int: 0-21
    "achievements",  # int
    "n_languages",  # int: number of supported languages
    "n_audio_languages",  # int: number of fully voiced languages
    "n_screenshots",  # int: screenshots on the store page
    "mac",  # bool
    "linux",  # bool
    "has_content_notes",  # bool: store page has mature-content notes
    "genres",  # str: comma-separated, e.g. "Strategy,Indie"
    "categories",  # str: comma-separated Steam features, e.g. "Single-player,Online Co-op"
]

SEQUEL_PATTERN = re.compile(
    r"(?<![\w.])(?:[2-9]|1[0-9]|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)(?![\w.+])"
    r"|\b(?:Remastered|Remake|Sequel)\b"
)

# ---------------------------------------------------------------- model features
NUM_LOG = [
    "regular_price",
    "achievements",
    "n_languages",
    "n_audio_languages",
    "n_screenshots",
    "n_genres",
    "n_categories",
    "n_social_features",
]
NUM_PLAIN = ["release_year", "required_age"]
FLAGS = ["is_free", "is_sequel", "mac", "linux", "has_content_notes"]
CATEGORICAL = ["release_month", "release_weekday"]
MULTI = ["genres", "categories"]
FEATURE_COLUMNS = NUM_LOG + NUM_PLAIN + FLAGS + CATEGORICAL + MULTI


def split_list(value) -> list[str]:
    """'Strategy, Indie' -> ['Strategy', 'Indie']. Accepts strings, lists or missing values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def make_features(X: pd.DataFrame) -> pd.DataFrame:
    """Input schema -> model features. Works row by row, so it is safe on any data split."""
    missing = [c for c in INPUT_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")

    out = pd.DataFrame(index=X.index)
    release = pd.to_datetime(X["release_date"], errors="coerce")
    price = pd.to_numeric(X["regular_price"], errors="coerce")

    out["regular_price"] = price
    out["is_free"] = (price == 0).astype(int)
    for col in [
        "required_age",
        "achievements",
        "n_languages",
        "n_audio_languages",
        "n_screenshots",
    ]:
        out[col] = pd.to_numeric(X[col], errors="coerce")
    for col in ["mac", "linux", "has_content_notes"]:
        out[col] = X[col].fillna(False).astype(bool).astype(int)
    out["is_sequel"] = (
        X["name"].fillna("").astype(str).map(lambda s: int(bool(SEQUEL_PATTERN.search(s))))
    )

    out["release_year"] = release.dt.year
    out["release_month"] = release.dt.month.fillna(0).astype(int).astype(str)
    out["release_weekday"] = release.dt.day_name().fillna("Unknown")

    out["genres"] = X["genres"].map(split_list)
    out["categories"] = X["categories"].map(
        lambda v: [c for c in split_list(v) if c not in LEAKY_CATEGORIES]
    )
    out["n_genres"] = out["genres"].map(len)
    out["n_categories"] = out["categories"].map(len)
    out["n_social_features"] = out["categories"].map(lambda c: len(set(c) & SOCIAL_CATEGORIES))
    return out[FEATURE_COLUMNS]


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """One 0/1 column per list value that appears in at least `min_freq` of the training rows."""

    def __init__(self, min_freq: float = 0.01):
        self.min_freq = min_freq

    def fit(self, X, y=None):
        column = X.iloc[:, 0]
        self.name_ = X.columns[0]
        counts = Counter(item for items in column for item in items)
        self.classes_ = sorted(k for k, v in counts.items() if v / len(column) >= self.min_freq)
        return self

    def transform(self, X):
        position = {c: i for i, c in enumerate(self.classes_)}
        out = np.zeros((len(X), len(self.classes_)))
        for row, items in enumerate(X.iloc[:, 0]):
            for item in items:
                if item in position:
                    out[row, position[item]] = 1
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.name_}={c}" for c in self.classes_], dtype=object)


def build_preprocessor(log_transform: bool = True, min_freq: float = 0.01) -> ColumnTransformer:
    """Imputation, optional log(1+x), scaling and encoding. Fitted on training data only."""
    log_step = (
        FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        if log_transform
        else "passthrough"
    )
    num_log = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", log_step),
            ("scale", StandardScaler()),
        ]
    )
    num_plain = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("num_log", num_log, NUM_LOG),
            ("num_plain", num_plain, NUM_PLAIN),
            ("flags", "passthrough", FLAGS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("genres", MultiHotEncoder(min_freq=min_freq), ["genres"]),
            ("categories", MultiHotEncoder(min_freq=min_freq), ["categories"]),
        ]
    )


def build_pipeline(model, log_transform: bool = True, min_freq: float = 0.01) -> Pipeline:
    """Full model: raw input schema -> features -> preprocessing -> estimator."""
    return Pipeline(
        [
            ("features", FunctionTransformer(make_features)),
            ("prep", build_preprocessor(log_transform=log_transform, min_freq=min_freq)),
            ("model", model),
        ]
    )
