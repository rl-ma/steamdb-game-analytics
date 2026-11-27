"""Create a small sample of model inputs + labels for unit tests and CI.

CI can't download the 600 MB dataset on every push, so tests run on this ~300-row sample,
committed at tests/data/sample_games.csv. Re-run this script only if the input schema changes:

    python scripts/make_sample_data.py --raw-dir data/raw
"""

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from steam_mlops import config
from steam_mlops.train import prepare_data

OUT_PATH = Path("tests/data/sample_games.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()

    prepared = prepare_data(args.raw_dir, download=True)
    sample = prepared["X_train"].copy()
    sample["lasting"] = prepared["y_train"]
    # Stratified, so the sample keeps the same share of lasting games as the full training set
    sample, _ = train_test_split(
        sample, train_size=args.rows, stratify=sample["lasting"], random_state=config.RANDOM_STATE
    )
    sample = sample.reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(sample)} rows ({sample['lasting'].mean():.0%} lasting) to {OUT_PATH}")


if __name__ == "__main__":
    main()
