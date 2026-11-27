# Training pipeline and MLflow

This part of the repo turns the INSY 674 notebook into reusable Python code that trains the model, tracks every
experiment in MLflow, and registers the best model for the API to serve.

## Run it

Needs Python 3.13 (the version CI uses, pinned in `.python-version`).

```bash
python3.13 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # project dependencies + pytest and ruff
pip install -e .                      # makes `steam_mlops` importable
pytest -m "not integration"           # the same unit tests CI runs (no download needed)
python -m steam_mlops.train           # downloads the data once (~600 MB), trains, logs, registers
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://localhost:5000 to browse runs
```

When MLflow runs in docker compose, point the code at it instead of the local file:
`MLFLOW_TRACKING_URI=http://localhost:5000 python -m steam_mlops.train`

A full run takes about a minute after the download.

## What each file does

| File | Job |
|---|---|
| `src/steam_mlops/config.py` | Every setting in one place: pinned dataset version, release window, hit threshold, MLflow names |
| `src/steam_mlops/data.py` | Downloads the pinned data, loads it (fixing the CSV header bug), keeps the 2019–2023 hits, splits 80/20 by release year, converts rows to the model's input schema |
| `src/steam_mlops/target.py` | The "lasting" label: more current players than expected for a game's size and release year (top 25% of its year), fitted on training data only |
| `src/steam_mlops/features.py` | The input schema, feature engineering (`make_features`) and the scikit-learn pipeline |
| `src/steam_mlops/train.py` | Trains 7 candidate models, logs each as an MLflow run, registers the best as `steam-staying-power@champion` |
| `scripts/make_sample_data.py` | Writes `tests/data/sample_games.csv` (300 rows) so tests and CI never need the big download |
| `tests/test_training_code.py` | Starter unit tests: data validation, preprocessing, model vs. baseline, valid probabilities |

## The three pipeline steps

`train.py` is split into three functions so they can become Prefect tasks in the training flow:

1. **`prepare_data()`**: download → load → select hits → split → label. Returns model inputs and labels.
2. **`train_candidates()`**: one MLflow run per candidate, each with parameters, 5-fold CV metrics, test
   metrics, the decision threshold and the fitted model.
3. **`register_best()`**: registers the run with the best cross-validated PR-AUC (never the baseline) and
   points the `champion` alias at it. The API always loads `models:/steam-staying-power@champion`, so a
   retrained model goes live by moving the alias.

## Experiments tracked in MLflow

| Run | CV PR-AUC (mean ± std) | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| **random_forest_tuned (registered)** | **0.597 ± 0.032** | **0.586** | **0.770** |
| random_forest_default | 0.584 ± 0.035 | 0.574 | 0.764 |
| gradient_boosting_tuned | 0.578 ± 0.019 | 0.584 | 0.773 |
| gradient_boosting_default | 0.573 ± 0.025 | 0.592 | 0.767 |
| logreg_tuned | 0.562 ± 0.029 | 0.527 | 0.733 |
| logreg_default | 0.555 ± 0.029 | 0.524 | 0.741 |
| dummy_baseline | 0.250 ± 0.001 | 0.258 | 0.500 |

The "tuned" settings come from the INSY 674 random search. Random guessing scores 0.25, because 25% of
games are lasting by definition.

## Design decisions worth knowing

- **A form-friendly input schema.** The model takes 13 fields a user can type into a web form (title,
  release date, price, genres, store features, counts…). Genres and categories are comma-separated text,
  which keeps API requests simple JSON. The notebook's developer and publisher history features were
  dropped, because a user can't fill them in; accuracy barely changed (CV PR-AUC 0.597 vs. 0.60).
- **Feature engineering lives inside the model.** `make_features` is the first step of the pipeline, so the
  API sends raw fields and gets exactly the same processing as in training.
- **The model carries its own code.** `log_model(code_paths=...)` ships `steam_mlops` with the model, so the
  API container can load it without installing this package.
- **The API gets probabilities.** `pyfunc_predict_fn="predict_proba"` makes `mlflow.pyfunc` return
  `[P(not lasting), P(lasting)]` instead of a bare 0/1.
- **Safe model files.** MLflow saves scikit-learn models with skops, which refuses to load types it hasn't
  been told to trust. Because we train these models ourselves, `train.py` trusts exactly the types each
  model contains (our feature code plus scikit-learn internals such as decision trees).
- **Decision threshold.** Chosen to maximise F1 on cross-validated training predictions (never the test
  set) and saved as a tag on the registered model version (`decision_threshold`, about 0.28). The API can
  read it to turn a probability into "likely to last / unlikely to last".
- **No leakage.** Post-launch columns are never inputs, store labels added after launch (trading cards,
  the 2025 accessibility labels) are removed, and the target's regression and cutoffs are fitted on
  training data only.

## For teammates

- **Model API:** `mlflow.pyfunc.load_model("models:/steam-staying-power@champion")`, then
  `model.predict(df)[:, 1]` is P(lasting). Example inputs are in `tests/data/sample_games.csv`, and every
  logged model also stores an input example.
- **Prefect flow:** wrap `prepare_data`, `train_candidates` and `register_best` with `@task`, and call them
  from a `@flow`.
- **Traffic simulator / Evidently:** the held-out test set (`prepare_data()["X_test"]` and `["y_test"]`)
  is unseen data with known labels, so model performance can be measured on it.
- **Tests:** build on `tests/test_training_code.py`; load `tests/data/sample_games.csv` instead of the full data.
