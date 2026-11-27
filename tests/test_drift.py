"""Drift detection tests on the committed sample data (no download)."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("evidently")  # skip if evidently isn't installed

from steam_mlops.drift import check_drift, inject_drift, monitoring_features  # noqa: E402

SAMPLE = Path(__file__).parent / "data" / "sample_games.csv"


@pytest.fixture(scope="module")
def halves() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(SAMPLE).sample(frac=1, random_state=0)
    return games.iloc[:150], games.iloc[150:]


def test_monitoring_features(halves):
    features = monitoring_features(halves[0])
    assert features.shape[1] == 19
    assert "name" not in features and "release_date" not in features
    assert set(features["genre_Action"]) <= {"0", "1"}


def test_inject_drift_changes_inputs(halves):
    games = halves[0]
    drifted = inject_drift(games)
    assert (drifted["regular_price"] == (games["regular_price"] * 3).round(2)).all()
    before = monitoring_features(games)["genre_Action"].eq("1").mean()
    after = monitoring_features(drifted)["genre_Action"].eq("1").mean()
    assert after > before


def test_no_drift_on_same_data(halves):
    reference, current = halves
    assert not check_drift(reference, current)["dataset_drift"]


def test_injected_drift_is_flagged(halves):
    reference, current = halves
    result = check_drift(reference, inject_drift(current))
    assert result["dataset_drift"]
    assert "regular_price" in result["drifted_columns"]
