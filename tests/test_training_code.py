"""Unit tests for the training code.

They run on the small committed sample, so CI needs no download.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from steam_mlops.features import FEATURE_COLUMNS, INPUT_COLUMNS, build_pipeline, make_features

SAMPLE = Path(__file__).parent / "data" / "sample_games.csv"


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE)


def one_game(**overrides) -> pd.DataFrame:
    game = {
        "name": "Galaxy Tactics II",
        "release_date": "2022-05-12",
        "regular_price": 24.99,
        "required_age": 0.0,
        "achievements": 40.0,
        "n_languages": 8.0,
        "n_audio_languages": 1.0,
        "n_screenshots": 12.0,
        "mac": True,
        "linux": False,
        "has_content_notes": False,
        "genres": "Strategy,Indie",
        "categories": "Single-player,Online PvP,Steam Workshop,Steam Trading Cards",
    }
    game.update(overrides)
    return pd.DataFrame([game])


def test_sample_has_input_columns_and_valid_target(sample):
    """Data validation: every input column is present and the target is never missing."""
    assert set(INPUT_COLUMNS) <= set(sample.columns)
    assert sample["lasting"].notna().all()
    assert set(sample["lasting"].unique()) == {0, 1}


def test_make_features_on_a_known_game():
    """Preprocessing: a known input produces the expected features."""
    feats = make_features(one_game()).iloc[0]
    assert list(make_features(one_game()).columns) == FEATURE_COLUMNS
    assert feats["is_sequel"] == 1  # "II" in the title
    assert feats["release_year"] == 2022 and feats["release_weekday"] == "Thursday"
    assert feats["genres"] == ["Strategy", "Indie"]
    assert "Steam Trading Cards" not in feats["categories"]  # leaky label is removed
    assert feats["n_social_features"] == 2  # Online PvP + Steam Workshop
    assert feats["is_free"] == 0


def test_make_features_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing input columns"):
        make_features(one_game().drop(columns=["genres"]))


def test_model_beats_dummy_baseline(sample):
    """Model vs. baseline: a small random forest ranks lasting games better than chance."""
    X, y = sample[INPUT_COLUMNS], sample["lasting"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    model = build_pipeline(RandomForestClassifier(n_estimators=100, random_state=0)).fit(X_tr, y_tr)
    baseline = build_pipeline(DummyClassifier(strategy="prior")).fit(X_tr, y_tr)
    model_ap = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])
    baseline_ap = average_precision_score(y_te, baseline.predict_proba(X_te)[:, 1])
    assert model_ap > baseline_ap


def test_predictions_are_valid_probabilities(sample):
    """Sensible predictions: outputs are probabilities between 0 and 1."""
    X, y = sample[INPUT_COLUMNS], sample["lasting"].to_numpy()
    model = build_pipeline(RandomForestClassifier(n_estimators=50, random_state=0)).fit(X, y)
    proba = model.predict_proba(one_game())[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
