"""Data drift check on the API inputs, using Evidently.

Compares a batch of model inputs (13 fields, same as the API gets) to the training set.
"""

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from steam_mlops.features import split_list

# Genres common enough to track one by one
TRACKED_GENRES = [
    "Action",
    "Adventure",
    "Casual",
    "Indie",
    "RPG",
    "Simulation",
    "Strategy",
    "Free To Play",
]

NUMERICAL = [
    "regular_price",
    "required_age",
    "achievements",
    "n_languages",
    "n_audio_languages",
    "n_screenshots",
    "n_genres",
    "n_categories",
]
CATEGORICAL = ["mac", "linux", "has_content_notes"] + [f"genre_{g}" for g in TRACKED_GENRES]

# KS test for numbers, chi-square for categories. A column drifts if p < 0.05 / 19
# (Bonferroni), and a batch drifts if any column does. Evidently's defaults gave too many
# false alarms on small batches, see monitoring/README.md.
COLUMN_ALPHA = 0.05 / len(NUMERICAL + CATEGORICAL)
# Share of drifted columns for a batch to count as drifted: 1 column out of 19.
# Set in Evidently too, so the HTML report gives the same verdict as our code.
DRIFT_SHARE = 1 / len(NUMERICAL + CATEGORICAL)


def monitoring_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Model inputs -> the 19 columns we monitor.

    name and release_date are skipped: names are unique and new games always have
    newer dates, so both would always show up as drift.
    """
    out = pd.DataFrame(index=inputs.index)
    for col in NUMERICAL[:6]:
        out[col] = pd.to_numeric(inputs[col], errors="coerce")
    genres = inputs["genres"].map(split_list)
    out["n_genres"] = genres.map(len)
    out["n_categories"] = inputs["categories"].map(split_list).map(len)
    for col in ["mac", "linux", "has_content_notes"]:
        out[col] = inputs[col].astype(int).astype(str)
    for genre in TRACKED_GENRES:
        out[f"genre_{genre}"] = genres.map(lambda g, genre=genre: str(int(genre in g)))
    return out


def inject_drift(
    inputs: pd.DataFrame, price_factor: float = 3.0, action_share: float = 0.5, seed: int = 42
) -> pd.DataFrame:
    """Fake drift for testing: prices x3 and half the non-Action games become Action."""
    rng = np.random.default_rng(seed)
    out = inputs.copy()
    out["regular_price"] = (out["regular_price"] * price_factor).round(2)
    no_action = ~out["genres"].map(split_list).map(lambda g: "Action" in g)
    picked = no_action & (rng.random(len(out)) < action_share)
    out.loc[picked, "genres"] = "Action," + out.loc[picked, "genres"].astype(str)
    return out


def build_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Run Evidently's drift preset. Both inputs are model-input rows."""
    definition = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    ref = Dataset.from_pandas(monitoring_features(reference), data_definition=definition)
    cur = Dataset.from_pandas(monitoring_features(current), data_definition=definition)
    preset = DataDriftPreset(
        num_method="ks",
        cat_method="chisquare",
        threshold=COLUMN_ALPHA,
        drift_share=DRIFT_SHARE,
    )
    return Report([preset]).run(cur, ref)


def summarize(snapshot, n_rows: int) -> dict:
    """Pull out the numbers that go in the drift_metrics table."""
    metrics = snapshot.dict()["metrics"]
    p_values = {
        m["config"]["column"]: float(m["value"])
        for m in metrics
        if m["config"]["type"].endswith("ValueDrift")
    }
    drifted = sorted(col for col, p in p_values.items() if p < COLUMN_ALPHA)
    return {
        "n_rows": n_rows,
        "n_columns": len(p_values),
        "n_drifted": len(drifted),
        "share_drifted": len(drifted) / len(p_values),
        "dataset_drift": len(drifted) >= 1,
        "drifted_columns": drifted,
        "p_values": p_values,
    }


def check_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    return summarize(build_report(reference, current), n_rows=len(current))
