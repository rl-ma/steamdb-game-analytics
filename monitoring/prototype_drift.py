"""Evidently drift prototype: training set (reference) vs. test set (current).

Run from the repo root:
    pip install -r monitoring/requirements.txt
    python -m monitoring.prototype_drift

It downloads the pinned Steam data (about 570 MB, once), rebuilds the hits and the
80/20 split from the INSY 674 notebook, and checks:
  1. train vs. test                  -> should show no drift
  2. train vs. test with drift added -> should show drift
  3. many small random windows of test traffic, with and without drift, to check the
     alert rule's false-alarm rate and detection rate per window size

TEMPORARY: the loading code below copies the notebook so this can run before the
training package (src/steam_mlops) is merged. Once it is, import the data and split
from there instead, so monitoring and training always use the same reference data.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from monitoring.drift import build_report, check_drift, inject_drift, summarize

# Same settings as the notebook
RANDOM_STATE = 42
WINDOW_START, WINDOW_END = 2019, 2023
HIT_THRESHOLD = 500
TEST_SIZE = 0.2

BASE_URL = ("https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/"
            "ba4e26785af33bee500e96597068c6e77f4edea9/")
FILES = {"steam_games.parquet": BASE_URL + "data/train-00000-of-00001.parquet",
         "steam_games.csv": BASE_URL + "games.csv"}

GAME_GENRES = {"Action", "Adventure", "Casual", "Indie", "RPG", "Simulation", "Strategy", "Sports",
               "Racing", "Massively Multiplayer", "Free To Play", "Early Access", "Violent", "Gore"}
IDLE_TAGS = {"Idler", "Clicker", "Incremental"}
LEAKY_CATEGORIES = {
    "Steam Trading Cards", "SteamVR Collectibles", "Steam Timeline",
    "Adjustable Text Size", "Subtitle Options", "Color Alternatives", "Camera Comfort",
    "Custom Volume Controls", "Stereo Sound", "Surround Sound", "Narrated Game Menus",
    "Chat Speech-to-text", "Chat Text-to-speech", "Playable without Timed Input",
    "Keyboard Only Option", "Mouse Only Option", "Touch Only Option", "Adjustable Difficulty",
    "Save Anytime", "HDR available",
}


def n_items(value) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return len(value)


def to_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(v) for v in value]


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = data_dir / name
        if not path.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)


def load_hits(data_dir: Path) -> pd.DataFrame:
    """The notebook's filters: 2019-2023 hits (500+ reviews), real games, no idle games."""
    games = pd.read_parquet(data_dir / "steam_games.parquet")
    header = pd.read_csv(data_dir / "steam_games.csv", nrows=0).columns.tolist()
    fixed_header = header[:7] + ["Discount", "DLC count"] + header[8:]
    extras = pd.read_csv(data_dir / "steam_games.csv", header=0, names=fixed_header, index_col=False,
                         usecols=["AppID", "Discount", "Tags"], low_memory=False)
    extras["AppID"] = extras["AppID"].astype(str)
    games = games.merge(extras.rename(columns={"AppID": "appID", "Discount": "discount",
                                               "Tags": "tags_csv"}), on="appID", how="left")

    games["release_dt"] = pd.to_datetime(games["release_date"], format="mixed", errors="coerce")
    games["release_year"] = games["release_dt"].dt.year
    games["total_reviews"] = games["positive"] + games["negative"]

    games = games[games["release_year"].between(WINDOW_START, WINDOW_END)]
    games = games[(games["short_description"].fillna("").str.strip().str.len() > 0)
                  & (games["screenshots"].map(n_items) > 0)
                  & (games["developers"].map(n_items) > 0)
                  & (games["genres"].map(n_items) > 0)
                  & ~games["name"].fillna("").str.contains("playtest", case=False)]
    games = games[games["total_reviews"] >= HIT_THRESHOLD]
    games = games[games["genres"].map(lambda g: len(set(g) & GAME_GENRES) > 0)]
    is_idle = games["tags_csv"].fillna("").str.split(",").map(lambda t: len(set(t) & IDLE_TAGS) > 0)
    return games[~is_idle].copy()


def to_contract(games: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> the 13 fields the API receives."""
    discount = games["discount"].fillna(0)
    regular_price = np.where(discount.between(1, 99), games["price"] / (1 - discount / 100),
                             games["price"])
    return pd.DataFrame({
        "name": games["name"],
        "release_date": games["release_dt"].dt.strftime("%Y-%m-%d"),
        "regular_price": np.round(regular_price, 2),
        "required_age": games["required_age"],
        "achievements": games["achievements"],
        "n_languages": games["supported_languages"].map(n_items),
        "n_audio_languages": games["full_audio_languages"].map(n_items),
        "n_screenshots": games["screenshots"].map(n_items),
        "mac": games["mac"].astype(bool),
        "linux": games["linux"].astype(bool),
        "has_content_notes": games["notes"].fillna("").str.strip() != "",
        "genres": games["genres"].map(lambda g: ", ".join(to_list(g))),
        "categories": games["categories"].map(
            lambda c: ", ".join(x for x in to_list(c) if x not in LEAKY_CATEGORIES)),
    }, index=games.index)


def window_study(reference: pd.DataFrame, test: pd.DataFrame, sizes: list[int], repeats: int,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Drift result for many random windows of test traffic."""
    rows = []
    for size in sizes:
        for drifted in (False, True):
            for i in range(repeats):
                window = test.sample(size, random_state=int(rng.integers(1_000_000)))
                if drifted:
                    window = inject_drift(window, seed=i)
                result = check_drift(reference, window)
                rows.append({"window_size": size, "drift_injected": drifted,
                             "alert": result.dataset_drift, "share_drifted": result.share_drifted})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).parent
    parser.add_argument("--data-dir", type=Path, default=here / "data")
    parser.add_argument("--reports-dir", type=Path, default=here / "reports")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    download(args.data_dir)
    hits = load_hits(args.data_dir)
    train, test = train_test_split(hits, test_size=TEST_SIZE, stratify=hits["release_year"],
                                   random_state=RANDOM_STATE)
    reference, current = to_contract(train), to_contract(test)
    print(f"Hits: {len(hits)} | reference (train): {len(reference)} | current (test): {len(current)}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    for label, batch in [("train_vs_test", current), ("train_vs_test_drifted", inject_drift(current))]:
        snapshot = build_report(reference, batch)
        snapshot.save_html(str(args.reports_dir / f"{label}.html"))
        result = summarize(snapshot, n_rows=len(batch))
        print(f"\n{label}: {result.n_drifted}/{result.n_columns} columns drifted "
              f"(share {result.share_drifted:.2f}) -> dataset drift: {result.dataset_drift}")
        print("  drifted columns:", ", ".join(result.drifted_columns) or "none")

    study = window_study(reference, current, sizes=[50, 100, 200], repeats=args.repeats,
                         rng=np.random.default_rng(RANDOM_STATE))
    summary = (study.groupby(["window_size", "drift_injected"])
               .agg(alert_rate=("alert", "mean"), mean_share_drifted=("share_drifted", "mean"))
               .round(3))
    print(f"\nAlert rate over {args.repeats} random windows of test traffic:")
    print(summary.to_string())
    print(f"\nHTML reports saved in {args.reports_dir}")


if __name__ == "__main__":
    main()
