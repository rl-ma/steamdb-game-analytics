"""Drift detection tests on synthetic contract rows (no data download needed)."""

import numpy as np
import pandas as pd

from monitoring.drift import CONTRACT_FIELDS, check_drift, inject_drift, monitoring_features

GENRE_POOL = ["Action", "Adventure", "Casual", "Indie", "RPG", "Simulation", "Strategy"]


def fake_games(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genres = [", ".join(rng.choice(GENRE_POOL, size=rng.integers(1, 4), replace=False))
              for _ in range(n)]
    return pd.DataFrame({
        "name": [f"Game {i}" for i in range(n)],
        "release_date": "2021-06-01",
        "regular_price": rng.choice([0, 4.99, 9.99, 14.99, 19.99, 29.99], size=n),
        "required_age": rng.choice([0, 0, 0, 13, 17], size=n),
        "achievements": rng.poisson(20, size=n),
        "n_languages": rng.integers(1, 15, size=n),
        "n_audio_languages": rng.integers(0, 4, size=n),
        "n_screenshots": rng.integers(3, 20, size=n),
        "mac": rng.random(n) < 0.3,
        "linux": rng.random(n) < 0.2,
        "has_content_notes": rng.random(n) < 0.1,
        "genres": genres,
        "categories": "Single-player, Steam Achievements",
    })


def test_features_from_contract_rows():
    games = fake_games(5, seed=0)
    assert list(games.columns) == CONTRACT_FIELDS
    features = monitoring_features(games)
    assert "name" not in features and "release_date" not in features
    assert (features["n_categories"] == 2).all()
    assert set(features["genre_Action"]) <= {"0", "1"}


def test_inject_drift_raises_prices_and_action_share():
    games = fake_games(400, seed=1)
    drifted = inject_drift(games)
    assert np.allclose(drifted["regular_price"], (games["regular_price"] * 3).round(2))
    before = monitoring_features(games)["genre_Action"].eq("1").mean()
    after = monitoring_features(drifted)["genre_Action"].eq("1").mean()
    assert after > before + 0.2


def test_same_distribution_is_not_flagged():
    result = check_drift(fake_games(800, seed=2), fake_games(300, seed=3))
    assert not result.dataset_drift


def test_injected_drift_is_flagged():
    reference = fake_games(800, seed=2)
    result = check_drift(reference, inject_drift(fake_games(300, seed=3)))
    assert result.dataset_drift
    assert {"regular_price", "genre_Action"} <= set(result.drifted_columns)
