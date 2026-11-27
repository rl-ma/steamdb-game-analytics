# Drift monitoring

Miles (`mileskv2`). This part checks whether the games sent to the API look different from the training data. Later it feeds the Grafana dashboards and the retraining alert.

For now this is the drift check and a prototype to pick the rule. The monitoring container and Grafana come after the check-in.

## Files

| File | What it does |
|---|---|
| `src/steam_mlops/drift.py` | Drift check on the 13 API input fields, plus `inject_drift()` for testing |
| `monitoring/prototype_drift.py` | Runs the check on the real data (train vs. test, with and without drift) |
| `tests/test_drift.py` | 4 tests on the sample CSV: no alert on normal data, alert on injected drift |
| `monitoring/requirements.txt` | Evidently, for the monitoring container later |

## Run it

```bash
pip install -r requirements-dev.txt -e . evidently==0.7.23 "pandas<3"
pytest tests/test_drift.py
python monitoring/prototype_drift.py   # downloads the data the first time
```

Evidently 0.7.23 is the latest version and breaks on pandas 3, so it needs `pandas<3`. Without Evidently installed the drift tests skip.

## What gets compared

The 13 input fields turned into 19 columns:

- numbers: price, required age, achievements, languages, audio languages, screenshots, number of genres, number of categories
- categories: mac, linux, content notes, and a 0/1 flag for 8 common genres

`name` and `release_date` are left out. Names are unique and new games always have newer dates, so they'd always look like drift.

## Drift rule

- KS test for number columns, chi-square for category columns
- a column drifts if p < 0.05 / 19 (Bonferroni, since we test 19 columns at once)
- a batch drifts if at least one column drifts

I tried Evidently's defaults first. With 3,000+ reference rows it uses distance measures and only flags a batch when half the columns drift. That gave a lot of false alarms on small batches, and the injected drift only moves 3 of 19 columns, so it was never flagged.

## Results

Reference is the training set (3,293 games). Fake drift is prices x3 and half the non-Action games tagged Action.

| Compared with | Columns drifted | Alert |
|---|---|---|
| test set (824 games) | 0 / 19 | no |
| test set with fake drift | 3 / 19 (price, Action, number of genres) | yes |

Random batches of test games, 40 tries each:

| Batch size | False alarms (no drift) | Caught (with drift) |
|---|---|---|
| 50 | 5% | 100% |
| 100 | 2.5% | 100% |
| 200 | 0% | 100% |

The plan is to check the last 100 predictions on each run.

## Need from others

1. Irene, Ruilei: a `prediction_id` returned by `/predict` and saved with each prediction, so the simulator can save the true label under the same id. Needed for PR-AUC and F1.
2. Irene, Mohamed: `predictions` should keep the 13 raw inputs (columns or one JSONB column) plus probability, label, model version and time.
3. Mohamed: `drift_metrics` table, roughly `run_at, n_rows, n_drifted, share_drifted, dataset_drift, p_values (jsonb), pr_auc, f1, model_version`.
4. Ishaan: name of the Prefect retraining deployment. The diagram has the Grafana alert starting retraining, either through a Grafana webhook or by the monitoring job calling Prefect. Can decide together.

## Still to do

- [ ] monitoring container: every few minutes read the last 100 predictions, run the check, add PR-AUC and F1 when labels exist, write to `drift_metrics`
- [ ] Grafana as code: Postgres and Prometheus data sources
- [ ] dashboard for API traffic, latency and errors
- [ ] dashboard for drift and model performance over time
- [ ] alert on drift that starts retraining (one retrain per drift episode)
- [ ] before/after drift screenshots for the main README
