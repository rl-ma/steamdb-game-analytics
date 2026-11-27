"""Train candidate models, track every run in MLflow, and register the best one.

Run it with:  python -m steam_mlops.train

Each candidate becomes one MLflow run with its parameters, cross-validated and test metrics,
and the fitted model. The candidate with the best cross-validated PR-AUC is registered as
``steam-staying-power`` and given the alias ``champion``, which the model API loads.

The three steps (prepare_data, train_candidates, register_best) are plain functions so they can
be wrapped as Prefect tasks in the training flow.
"""

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from skops.io import dumps, get_untrusted_types

from steam_mlops import config, data
from steam_mlops.features import build_pipeline
from steam_mlops.target import LastingLabeler

log = logging.getLogger(__name__)
PACKAGE_DIR = Path(__file__).resolve().parent  # shipped with the model so the API can load it


# ---------------------------------------------------------------- candidate models
# Each entry is one experiment. The "tuned" settings come from the INSY 674 random search.
CANDIDATES = [
    {
        "name": "dummy_baseline",
        "family": "baseline",
        "model": DummyClassifier(strategy="prior"),
        "log_transform": True,
        "min_freq": 0.01,
    },
    {
        "name": "logreg_default",
        "family": "logistic_regression",
        "model": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "log_transform": True,
        "min_freq": 0.01,
    },
    {
        "name": "logreg_tuned",
        "family": "logistic_regression",
        "model": LogisticRegression(max_iter=3000, C=0.034),
        "log_transform": True,
        "min_freq": 0.05,
    },
    {
        "name": "random_forest_default",
        "family": "random_forest",
        "model": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "log_transform": True,
        "min_freq": 0.01,
    },
    {
        "name": "random_forest_tuned",
        "family": "random_forest",
        "model": RandomForestClassifier(
            n_estimators=260,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "log_transform": False,
        "min_freq": 0.02,
    },
    {
        "name": "gradient_boosting_default",
        "family": "gradient_boosting",
        "model": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "log_transform": True,
        "min_freq": 0.01,
    },
    {
        "name": "gradient_boosting_tuned",
        "family": "gradient_boosting",
        "model": HistGradientBoostingClassifier(
            learning_rate=0.019,
            max_iter=241,
            max_leaf_nodes=27,
            min_samples_leaf=42,
            l2_regularization=2.14,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "log_transform": False,
        "min_freq": 0.02,
    },
]


# ---------------------------------------------------------------- step 1: data
def prepare_data(raw_dir: Path = config.RAW_DIR, download: bool = True) -> dict:
    """Download (once), load, filter, split and label. Returns model inputs and labels."""
    if download:
        data.download_raw_data(raw_dir)
    games = data.load_games(raw_dir)
    hits, steps = data.select_hits(games)
    train, test = data.split(hits)
    labeler = LastingLabeler().fit(train)
    prepared = {
        "X_train": data.to_model_input(train),
        "y_train": labeler.label(train),
        "X_test": data.to_model_input(test),
        "y_test": labeler.label(test),
        "filter_steps": steps,
        "expected_players_slope": labeler.slope,
    }
    log.info(
        "Data ready: %d training and %d test games (%.1f%% lasting)",
        len(train),
        len(test),
        100 * prepared["y_train"].mean(),
    )
    return prepared


# ---------------------------------------------------------------- step 2: experiments
def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(pipeline, X_train, y_train, X_test, y_test) -> tuple[dict, float, object]:
    """5-fold CV on the training set, then refit on all of it and score the test set once."""
    folds = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    oof = np.zeros(len(y_train))
    fold_ap, fold_auc = [], []
    for fit_idx, val_idx in folds.split(X_train, y_train):
        model = clone(pipeline).fit(X_train.iloc[fit_idx], y_train[fit_idx])
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        fold_ap.append(average_precision_score(y_train[val_idx], oof[val_idx]))
        fold_auc.append(roc_auc_score(y_train[val_idx], oof[val_idx]))

    threshold = _best_f1_threshold(
        y_train, oof
    )  # chosen on training CV only, never on the test set
    fitted = clone(pipeline).fit(X_train, y_train)
    test_proba = fitted.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    metrics = {
        "cv_pr_auc_mean": float(np.mean(fold_ap)),
        "cv_pr_auc_std": float(np.std(fold_ap)),
        "cv_roc_auc_mean": float(np.mean(fold_auc)),
        "test_pr_auc": average_precision_score(y_test, test_proba),
        "test_roc_auc": roc_auc_score(y_test, test_proba),
        "test_f1": f1_score(y_test, test_pred, zero_division=0),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred, zero_division=0),
    }
    return metrics, threshold, fitted


def train_candidates(prepared: dict, candidates: list[dict] = CANDIDATES) -> pd.DataFrame:
    """Log one MLflow run per candidate. Returns a table of runs and their metrics."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    X_train, y_train = prepared["X_train"], prepared["y_train"]
    X_test, y_test = prepared["X_test"], prepared["y_test"]

    rows = []
    for cand in candidates:
        pipeline = build_pipeline(
            clone(cand["model"]), log_transform=cand["log_transform"], min_freq=cand["min_freq"]
        )
        with mlflow.start_run(run_name=cand["name"]) as run:
            log.info("Training %s ...", cand["name"])
            metrics, threshold, fitted = evaluate(pipeline, X_train, y_train, X_test, y_test)

            model_params = {
                f"model__{k}": v
                for k, v in cand["model"].get_params().items()
                if k in _explicit_params(cand["model"])
            }
            mlflow.log_params(
                {
                    "model_family": cand["family"],
                    "estimator": type(cand["model"]).__name__,
                    "log_transform": cand["log_transform"],
                    "min_freq": cand["min_freq"],
                    **model_params,
                }
            )
            mlflow.log_params(
                {
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "share_lasting": round(float(y_train.mean()), 3),
                    "dataset_revision": config.DATASET_REVISION,
                    "window": f"{config.WINDOW_START}-{config.WINDOW_END}",
                    "hit_threshold": config.HIT_THRESHOLD,
                }
            )
            mlflow.log_metrics({**metrics, "decision_threshold": threshold})
            mlflow.set_tags({"model_family": cand["family"], "stage": "candidate"})

            example = X_train.head(5)
            info = mlflow.sklearn.log_model(
                fitted,
                name="model",
                signature=infer_signature(example, fitted.predict_proba(example)),
                input_example=example,
                code_paths=[str(PACKAGE_DIR)],  # lets the API unpickle our custom code
                pyfunc_predict_fn="predict_proba",  # pyfunc returns probabilities, not labels
                skops_trusted_types=_types_to_trust(fitted),
            )
            rows.append(
                {
                    "name": cand["name"],
                    "family": cand["family"],
                    "run_id": run.info.run_id,
                    "model_uri": info.model_uri,
                    "decision_threshold": threshold,
                    **metrics,
                }
            )
            log.info(
                "  %s: CV PR-AUC %.3f +/- %.3f, test PR-AUC %.3f",
                cand["name"],
                metrics["cv_pr_auc_mean"],
                metrics["cv_pr_auc_std"],
                metrics["test_pr_auc"],
            )
    return pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)


def _types_to_trust(model) -> list[str]:
    """MLflow saves scikit-learn models with skops, which only loads types it is told to trust.

    We trained this model ourselves in this run, so we trust exactly the types it contains:
    our own feature code plus scikit-learn internals such as decision trees.
    """
    return get_untrusted_types(data=dumps(model))


def _explicit_params(estimator) -> set[str]:
    """Hyperparameters that differ from the estimator's defaults (keeps the MLflow run readable)."""
    defaults = type(estimator)().get_params()
    return {k for k, v in estimator.get_params().items() if defaults.get(k) != v}


# ---------------------------------------------------------------- step 3: registry
def register_best(results: pd.DataFrame) -> dict:
    """Register the best non-baseline run (by CV PR-AUC) and point the 'champion' alias at it."""
    best = results[results["family"] != "baseline"].iloc[0]
    version = mlflow.register_model(best["model_uri"], config.REGISTERED_MODEL_NAME)
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    client.set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS, version.version
    )
    for key in ["decision_threshold", "cv_pr_auc_mean", "test_pr_auc", "test_roc_auc"]:
        client.set_model_version_tag(
            config.REGISTERED_MODEL_NAME, version.version, key, f"{best[key]:.4f}"
        )
    client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME, version.version, "candidate", best["name"]
    )
    client.set_tag(best["run_id"], "stage", "registered")
    log.info(
        "Registered %s v%s as @%s (CV PR-AUC %.3f)",
        config.REGISTERED_MODEL_NAME,
        version.version,
        config.CHAMPION_ALIAS,
        best["cv_pr_auc_mean"],
    )
    return {
        "name": config.REGISTERED_MODEL_NAME,
        "version": version.version,
        "alias": config.CHAMPION_ALIAS,
        "candidate": best["name"],
        "decision_threshold": float(best["decision_threshold"]),
    }


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(
        description="Train, track and register the staying-power model."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=config.RAW_DIR, help="folder with the raw data files"
    )
    parser.add_argument(
        "--no-download", action="store_true", help="fail instead of downloading missing files"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    prepared = prepare_data(args.raw_dir, download=not args.no_download)
    results = train_candidates(prepared)
    print(
        "\n"
        + results[["name", "cv_pr_auc_mean", "cv_pr_auc_std", "test_pr_auc", "test_roc_auc"]]
        .round(3)
        .to_string(index=False)
    )
    registered = register_best(results)
    print(
        f"\nRegistered {registered['name']} v{registered['version']} (@{registered['alias']}): "
        f"{registered['candidate']}"
    )
    return registered


if __name__ == "__main__":
    main()
