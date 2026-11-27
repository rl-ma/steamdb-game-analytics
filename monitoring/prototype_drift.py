"""Drift prototype: training set (reference) vs. test set, with and without fake drift.

Run from the repo root (downloads ~600 MB the first time):
    python monitoring/prototype_drift.py

Also checks many small windows of test games, to see how often the drift rule
false-alarms and how often it catches drift for each window size.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from steam_mlops import config, data
from steam_mlops.drift import build_report, check_drift, inject_drift, summarize


def window_study(reference, test, sizes, repeats, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        for drifted in (False, True):
            for i in range(repeats):
                window = test.sample(size, random_state=int(rng.integers(1_000_000)))
                if drifted:
                    window = inject_drift(window, seed=i)
                result = check_drift(reference, window)
                rows.append(
                    {
                        "window_size": size,
                        "drift_injected": drifted,
                        "alert": result["dataset_drift"],
                        "share_drifted": result["share_drifted"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--reports-dir", type=Path, default=Path("monitoring/reports"))
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()

    data.download_raw_data(args.raw_dir)
    hits, _ = data.select_hits(data.load_games(args.raw_dir))
    train, test = data.split(hits)
    reference, current = data.to_model_input(train), data.to_model_input(test)
    print(f"Reference (train): {len(reference)} games | current (test): {len(current)} games")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    for name, batch in [
        ("train_vs_test", current),
        ("train_vs_test_drifted", inject_drift(current)),
    ]:
        snapshot = build_report(reference, batch)
        snapshot.save_html(str(args.reports_dir / f"{name}.html"))
        r = summarize(snapshot, n_rows=len(batch))
        print(f"\n{name}: {r['n_drifted']}/{r['n_columns']} columns drifted")
        print("  alert:", r["dataset_drift"])
        print("  drifted:", ", ".join(r["drifted_columns"]) or "none")

    study = window_study(reference, current, sizes=[50, 100, 200], repeats=args.repeats)
    summary = study.groupby(["window_size", "drift_injected"]).agg(
        alert_rate=("alert", "mean"), mean_share_drifted=("share_drifted", "mean")
    )
    print(f"\nAlert rate over {args.repeats} random windows:")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
