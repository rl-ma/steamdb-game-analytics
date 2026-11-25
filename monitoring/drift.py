"""Drift checks on the model's input fields, using Evidently.

The API receives the 13 fields of the model contract (see the team plan). Most of
them can be compared directly; the list fields (genres, categories) are turned into
counts and flags for the most common genres so Evidently can test them.

This module is shared by the prototype (prototype_drift.py) and, later, the
scheduled monitoring service.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# The 13 input fields the API receives (model contract)
CONTRACT_FIELDS = [
    "name", "release_date", "regular_price", "required_age", "achievements",
    "n_languages", "n_audio_languages", "n_screenshots", "mac", "linux",
    "has_content_notes", "genres", "categories",
]

# Genres common enough among hits to track one by one
TRACKED_GENRES = ["Action", "Adventure", "Casual", "Indie", "RPG", "Simulation", "Strategy",
                  "Free To Play"]

NUMERICAL = ["regular_price", "required_age", "achievements", "n_languages",
             "n_audio_languages", "n_screenshots", "n_genres", "n_categories"]
CATEGORICAL = ["mac", "linux", "has_content_notes"] + [f"genre_{g}" for g in TRACKED_GENRES]

# Drift rule (chosen with prototype_drift.py, see monitoring/README.md):
# - every column gets a p-value test: Kolmogorov-Smirnov (numerical), chi-square (categorical)
# - a column drifts when p < 0.05 / number of columns (Bonferroni correction)
# - a batch drifts when at least one column drifts
# Evidently's defaults (distance measures, batch drift at 50% of columns) gave many false
# alarms on windows of 50-200 requests and missed drift in only a few features.
FAMILY_ALPHA = 0.05
COLUMN_ALPHA = FAMILY_ALPHA / len(NUMERICAL + CATEGORICAL)


def split_list(value) -> list[str]:
    """Comma-separated text (as the API receives it) -> list of stripped items."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def monitoring_features(contract: pd.DataFrame) -> pd.DataFrame:
    """Turn contract rows into the columns Evidently compares.

    `name` and `release_date` are not monitored: names are unique and release dates
    keep moving forward, so both would always look like drift.
    """
    out = pd.DataFrame(index=contract.index)
    for col in ["regular_price", "required_age", "achievements", "n_languages",
                "n_audio_languages", "n_screenshots"]:
        out[col] = pd.to_numeric(contract[col], errors="coerce")
    genres = contract["genres"].map(split_list)
    out["n_genres"] = genres.map(len)
    out["n_categories"] = contract["categories"].map(split_list).map(len)
    for col in ["mac", "linux", "has_content_notes"]:
        out[col] = contract[col].astype(int).astype(str)
    for genre in TRACKED_GENRES:
        out[f"genre_{genre}"] = genres.map(lambda g, genre=genre: str(int(genre in g)))
    return out


def inject_drift(contract: pd.DataFrame, price_factor: float = 3.0, action_share: float = 0.5,
                 seed: int = 42) -> pd.DataFrame:
    """Simulated drift for testing: prices go up and many more games are labelled Action.

    The traffic simulator owns the real drift switch; this mirrors it so the monitoring
    side can be tested without the full stack.
    """
    rng = np.random.default_rng(seed)
    drifted = contract.copy()
    drifted["regular_price"] = (drifted["regular_price"] * price_factor).round(2)
    no_action = ~drifted["genres"].map(split_list).map(lambda g: "Action" in g)
    picked = no_action & (rng.random(len(drifted)) < action_share)
    drifted.loc[picked, "genres"] = "Action, " + drifted.loc[picked, "genres"].astype(str)
    return drifted


@dataclass
class DriftResult:
    n_rows: int
    n_columns: int
    n_drifted: int
    share_drifted: float
    dataset_drift: bool
    column_p_values: dict[str, float] = field(default_factory=dict)

    @property
    def drifted_columns(self) -> list[str]:
        return sorted(c for c, p in self.column_p_values.items() if p < COLUMN_ALPHA)


def _definition() -> DataDefinition:
    return DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)


def build_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Run Evidently's data drift preset. Both inputs are contract rows."""
    ref = Dataset.from_pandas(monitoring_features(reference), data_definition=_definition())
    cur = Dataset.from_pandas(monitoring_features(current), data_definition=_definition())
    preset = DataDriftPreset(num_method="ks", cat_method="chisquare", threshold=COLUMN_ALPHA)
    return Report([preset]).run(cur, ref)


def summarize(snapshot, n_rows: int) -> DriftResult:
    """Pull the numbers we store in Postgres out of an Evidently snapshot."""
    metrics = snapshot.dict()["metrics"]
    p_values = {m["config"]["column"]: float(m["value"]) for m in metrics
                if m["config"]["type"].endswith("ValueDrift")}
    counts = next(m["value"] for m in metrics if m["config"]["type"].endswith("DriftedColumnsCount"))
    n_drifted = int(counts["count"])
    return DriftResult(n_rows=n_rows, n_columns=len(p_values), n_drifted=n_drifted,
                       share_drifted=float(counts["share"]), dataset_drift=n_drifted >= 1,
                       column_p_values=p_values)


def check_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult:
    return summarize(build_report(reference, current), n_rows=len(current))
