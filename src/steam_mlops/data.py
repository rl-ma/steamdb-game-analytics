"""Data ingestion: download the pinned dataset, load it, choose the games, split it.

Same logic as Sections 2.1-2.2 of the INSY 674 notebook.
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from steam_mlops import config
from steam_mlops.features import INPUT_COLUMNS

log = logging.getLogger(__name__)


def download_raw_data(raw_dir: Path = config.RAW_DIR, force: bool = False) -> dict[str, Path]:
    """Download the two raw files once (about 600 MB in total) and return their paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename, url in config.RAW_FILES.items():
        path = raw_dir / filename
        if force or not path.exists():
            log.info("Downloading %s ...", filename)
            urllib.request.urlretrieve(url, path)
        paths[filename] = path
    return paths


def _n_items(value) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return len(value)


def load_games(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Load the parquet file and add Discount and Tags from the CSV (which has a header bug)."""
    games = pd.read_parquet(raw_dir / "steam_games.parquet")

    csv_path = raw_dir / "steam_games.csv"
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    fixed_header = (
        header[:7] + ["Discount", "DLC count"] + header[8:]
    )  # "DiscountDLC count" is merged
    extras = pd.read_csv(
        csv_path,
        header=0,
        names=fixed_header,
        index_col=False,
        usecols=["AppID", "Discount", "Tags"],
        low_memory=False,
    )
    extras["AppID"] = extras["AppID"].astype(str)
    extras = extras.rename(columns={"AppID": "appID", "Discount": "discount", "Tags": "tags_csv"})
    games = games.merge(extras, on="appID", how="left")

    games["release_dt"] = pd.to_datetime(games["release_date"], format="mixed", errors="coerce")
    games["release_year"] = games["release_dt"].dt.year
    games["total_reviews"] = games["positive"] + games["negative"]
    return games


def select_hits(games: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Keep 2019-2023 hits that are real games. Returns the data and a log of each filter step."""
    steps = [("all games", len(games))]
    df = games[games["release_year"].between(config.WINDOW_START, config.WINDOW_END)]
    steps.append((f"released {config.WINDOW_START}-{config.WINDOW_END}", len(df)))

    complete = (
        (df["short_description"].fillna("").str.strip().str.len() > 0)
        & (df["screenshots"].map(_n_items) > 0)
        & (df["developers"].map(_n_items) > 0)
        & (df["genres"].map(_n_items) > 0)
        & ~df["name"].fillna("").str.contains("playtest", case=False)
    )
    df = df[complete]
    steps.append(("complete store page, not a playtest", len(df)))

    df = df[df["total_reviews"] >= config.HIT_THRESHOLD]
    steps.append((f"a hit: {config.HIT_THRESHOLD}+ reviews", len(df)))

    df = df[df["genres"].map(lambda g: len(set(g) & config.GAME_GENRES) > 0)]
    steps.append(("is a game (not only software genres)", len(df)))

    is_idle = (
        df["tags_csv"].fillna("").str.split(",").map(lambda t: len(set(t) & config.IDLE_TAGS) > 0)
    )
    df = df[~is_idle]
    steps.append(("not an idle / clicker game", len(df)))
    return df.copy(), steps


def to_model_input(games: pd.DataFrame) -> pd.DataFrame:
    """Convert dataset rows to the model's input schema (what the API receives)."""
    discount = games["discount"].fillna(0)
    regular_price = np.where(
        discount.between(1, 99), games["price"] / (1 - discount / 100), games["price"]
    )
    out = pd.DataFrame(
        {
            "name": games["name"].fillna(""),
            "release_date": games["release_dt"].dt.strftime("%Y-%m-%d"),
            "regular_price": np.round(regular_price, 2),
            "required_age": games["required_age"],
            "achievements": games["achievements"],
            "n_languages": games["supported_languages"].map(_n_items),
            "n_audio_languages": games["full_audio_languages"].map(_n_items),
            "n_screenshots": games["screenshots"].map(_n_items),
            "mac": games["mac"].astype(bool),
            "linux": games["linux"].astype(bool),
            "has_content_notes": games["notes"].fillna("").str.strip() != "",
            "genres": games["genres"].map(lambda g: ",".join(map(str, g))),
            "categories": games["categories"].map(lambda c: ",".join(map(str, c))),
        },
        index=games.index,
    )
    # Counts as floats, so a missing value in an API request doesn't break MLflow's schema check
    for col in [
        "required_age",
        "achievements",
        "n_languages",
        "n_audio_languages",
        "n_screenshots",
    ]:
        out[col] = out[col].astype(float)
    return out[INPUT_COLUMNS]


def split(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split, stratified by release year (adapted from Geron)."""
    train, test = train_test_split(
        games,
        test_size=config.TEST_SIZE,
        stratify=games["release_year"],
        random_state=config.RANDOM_STATE,
    )
    return train.copy(), test.copy()
