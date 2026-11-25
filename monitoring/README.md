# Drift monitoring

Owner: Miles Koseleci-Vieira (`mileskv2`), branch `feat/drift-monitoring`.

This folder will become the `monitoring` service: on a schedule it reads recent
predictions from Postgres, runs Evidently against the training data, writes the
results to `drift_metrics`, and starts retraining when drift is detected. Grafana
dashboards and alerts for the API and for drift will also live here.

For the Nov 27 check-in it contains a prototype that picks the drift rule.

## Files

| File | What it is |
|---|---|
| `drift.py` | Drift check on the 13 model-contract fields: feature preparation, the drift rule, a drift injector for testing |
| `prototype_drift.py` | Runs the check on the real data: train vs. test, with and without injected drift, and over many small windows |
| `tests/test_drift.py` | Unit tests on synthetic games: no alert on same-distribution data, alert on injected drift |
| `requirements.txt` | Monitoring dependencies (Evidently 0.7.23 needs pandas below 3) |

## Run it

From the repo root:

```bash
pip install -r monitoring/requirements.txt
pytest monitoring                       # about 5 seconds, no download
python -m monitoring.prototype_drift    # downloads about 570 MB once, then about 3 minutes
```

The prototype writes Evidently HTML reports to `monitoring/reports/`. The data and
the reports are ignored by Git.

The loading code in `prototype_drift.py` copies the INSY 674 notebook so the
prototype runs before the training package is merged. When `src/steam_mlops` is
on `main`, it should import the data and split from there, so monitoring and
training use the same reference set.

## Monitored columns

The API receives 13 fields per game. Evidently compares 19 columns derived from them:

- numerical: `regular_price`, `required_age`, `achievements`, `n_languages`,
  `n_audio_languages`, `n_screenshots`, number of genres, number of categories
- categorical: `mac`, `linux`, `has_content_notes`, and a 0/1 flag for each of
  eight common genres (Action, Adventure, Casual, Indie, RPG, Simulation,
  Strategy, Free To Play)

`name` and `release_date` are left out. Names are unique, and new games always
have later release dates than the training set, so both would always register as drift.

## The drift rule

- Each column gets a p-value test: Kolmogorov-Smirnov for numerical columns,
  chi-square for categorical ones.
- A column has drifted when its p-value is below 0.05 / 19 (Bonferroni correction).
- A batch of requests has drifted when at least one column has drifted.

Evidently's defaults did not work on our data. With a reference set over 1,000
rows it switches to distance measures (Wasserstein and Jensen-Shannon) and flags
the whole batch only when half the columns drift. On windows of 100 requests with
no drift, a quarter of the columns still looked drifted. And the injected drift
(prices ×3, half of the non-Action games tagged Action) moves only 3 of 19
columns, well below the 50% the default rule needs.

Results from `python -m monitoring.prototype_drift --repeats 40` (train = 3,293
games as reference, test = 824 games):

| Comparison | Columns drifted | Alert |
|---|---|---|
| train vs. test | 0 of 19 | no |
| train vs. test with drift injected | 3 of 19 (`regular_price`, `genre_Action`, number of genres) | yes |

| Window of test traffic | Alert rate, no drift | Alert rate, drift injected |
|---|---|---|
| 50 requests | 5.0% | 100% |
| 100 requests | 2.5% | 100% |
| 200 requests | 0% | 100% |

Plan: check the last 100 predictions each time the monitoring job runs.

## Needed from the rest of the team

These are proposals. Each one changes someone else's part, so please reply on the PR.

1. **A key that links predictions to labels** (Irene, Ruilei). PR-AUC and F1 need
   each prediction matched to its true label. Proposal: `/predict` returns a
   `prediction_id` (UUID) and stores it in `predictions`; the simulator writes
   `ground_truth(prediction_id, lasting)` using that id.
2. **The predictions table stores the raw request** (Irene, Mohamed). Monitoring
   reads the same 13 fields the API received, either as columns or as one JSONB
   `features` column, plus `predicted_proba`, `predicted_label`, `model_version`
   and `created_at`.
3. **`drift_metrics` table** (Mohamed). Proposed columns:

   | Column | Type |
   |---|---|
   | `id` | serial primary key |
   | `run_at` | timestamptz |
   | `window_start`, `window_end` | timestamptz |
   | `n_rows` | integer |
   | `n_drifted_columns` | integer |
   | `share_drifted_columns` | real |
   | `dataset_drift` | boolean |
   | `column_p_values` | jsonb |
   | `pr_auc`, `f1` | real, null when there are no labels yet |
   | `model_version` | text |
   | `retrain_triggered` | boolean |

4. **Retraining entry point** (Ishaan). When drift is detected, the monitoring
   service calls Prefect's `run_deployment` with the deployment name of the
   training flow. Grafana shows the same alert so it appears in the screenshots.

## Still to do for the final submission

- [ ] Monitoring container: a loop that reads the last window from Postgres, runs `check_drift`, adds PR-AUC and F1 when labels exist, writes `drift_metrics`
- [ ] Retraining trigger through the Prefect deployment, with a cooldown so one drift episode starts one retrain
- [ ] Grafana provisioning in the repo: Postgres and Prometheus data sources
- [ ] Dashboard 1: request rate, latency, error rate (Prometheus)
- [ ] Dashboard 2: drift share, drifted columns, PR-AUC and F1 over time (Postgres)
- [ ] Alert rule on `dataset_drift`
- [ ] Screenshots before and after the simulator's drift switch, for the main README
- [ ] Replace the copied loading code with imports from `src/steam_mlops`
