"""Starter smoke tests validating the core ML environment and repository artifacts."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_version_is_313_or_newer() -> None:
    assert sys.version_info >= (3, 13)


def test_core_notebook_dependencies_importable() -> None:
    for pkg in ("numpy", "pandas", "pyarrow", "sklearn", "scipy", "matplotlib"):
        assert importlib.import_module(pkg) is not None


def test_source_notebook_is_valid_json() -> None:
    nb_path = REPO_ROOT / "steam_staying_power.ipynb"
    assert nb_path.exists()
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert "cells" in data and len(data["cells"]) > 0
    assert any(cell.get("cell_type") == "code" for cell in data["cells"])


def test_soft_voting_pipeline_smoke() -> None:
    """Verify the notebook's ensemble pattern fits and returns valid probabilities."""
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame(
        {
            "regular_price": rng.uniform(0, 60, size=n),
            "language_count": rng.integers(1, 15, size=n),
            "primary_genre": rng.choice(["Strategy", "Action", "Simulation"], size=n),
        }
    )
    y = rng.integers(0, 2, size=n)

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), ["regular_price", "language_count"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["primary_genre"]),
        ]
    )
    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("lr", LogisticRegression(max_iter=200)),
            ("hgb", HistGradientBoostingClassifier(max_iter=10, random_state=42)),
        ],
        voting="soft",
    )
    pipe = Pipeline([("prep", preprocessor), ("model", ensemble)])
    pipe.fit(X, y)

    probs = pipe.predict_proba(X)[:, 1]
    assert probs.shape == (n,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
