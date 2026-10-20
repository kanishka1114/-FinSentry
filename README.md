# FinSentry
**Explainable machine learning for mobile-money fraud monitoring, evaluated honestly over time**

FinSentry compares a transparent rule baseline with Logistic Regression and Random Forest for ranking mobile-money transactions by fraud risk. It uses the synthetic PaySim dataset and presents everything in a Streamlit risk-analytics dashboard.

> PaySim is synthetic. Results are not evidence of real-world detection performance, have not been validated at any bank, and this project must not be used to make financial decisions.

## The problem
Fraud is rare: about 0.13% of the transactions in the 1M-row sample used here. That makes common shortcuts misleading:
- Accuracy rewards flagging nothing.
- Random train/test splits mix past and future.
- One headline number hides how much results move over time.

FinSentry focuses on the questions a fraud team actually faces:
- How well are fraud cases ranked above legitimate ones?
- How many false alerts must reviewers handle?
- How much fraud is missed?
- How stable is all of this across time and across data samples?

## Key features
- **Leakage-aware evaluation:** chronological split by simulation step, models fit on the past only, thresholds chosen on validation only, and the test period scored once. A test flips every test label to prove nothing upstream depends on it.
- **Operational metrics:** precision, recall, false alerts, missed fraud, alerts per 1,000 and top-K review queues, plus PR-AUC compared with its prevalence baseline.
- **Two threshold strategies:** max F1 and a fixed alert budget, plus a cost explorer for choosing between them.
- **Walk-forward temporal backtest** over 5 later time windows, with drift diagnostics.
- **Uncertainty:** row and moving-block bootstrap intervals, calibration (Brier, ECE, reliability, Platt scaling) and score drift (PSI).
- **Per-transaction explanations:** exact log-odds contributions for Logistic Regression and tree-path contributions for Random Forest.
- **Three-seed robustness study** that repeats the full experiment on independent samples.
- **10-page Streamlit dashboard** and an auto-generated Markdown report. Both read saved results only and never train at startup.
- **92 automated tests** on small synthetic data.

## Tech stack
Python 3.10+ · pandas · NumPy · scikit-learn · joblib · Streamlit · Plotly · pytest. There is no database, no external API and nothing beyond `requirements.txt`.

## Dashboard screenshots
> Screenshots coming soon. To see the dashboard now, follow the quick start below: the saved results in this repository let it run without downloading the dataset.

<!-- Add images here, e.g. ![Overview](docs/screenshots/overview.png) -->

## Project structure
```text
app.py                  Streamlit dashboard (10 pages)
.streamlit/config.toml  Dashboard theme
src/
  config.py             Paths, seeds, features and default settings
  data_loader.py        Memory-safe chunked loading and label-free sampling
  splits.py             Chronological and random splits by simulation step
  rules.py              Transparent rule baseline
  train.py              Training, threshold selection, evaluation, saved artifacts
  evaluation.py         Shared evaluation for the single split and the backtest
  metrics.py            Precision/recall/PR-AUC, thresholds, top-K, curves
  uncertainty.py        Row and moving-block bootstrap
  calibration.py        Brier, ECE, reliability bins, Platt scaling
  drift.py              PSI and score-distribution summaries
  explain.py            Global importance and per-transaction contributions
  protocol.py           Final-fit protocol and threshold transfer checks
  backtest.py           Walk-forward temporal backtest
  experiment.py         The single reproducible experiment workflow
  robustness.py         Multi-seed robustness study
  report.py             Report generation and output consistency checks
  ui.py                 Dashboard presentation helpers (formatting, cards, chart style)
tests/                  92 pytest tests on synthetic PaySim-like data
outputs/                Saved results of the primary experiment and robustness study
models/                 Saved primary-experiment models used by the dashboard
docs/                   Demo guide and viva guide
report/                 Report template
data/                   Put the PaySim CSV here (not included; see data/README.md)
```

## Dataset and license
Download `PS_20174392719_1491204439457_log.csv` from [Kaggle PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) and put it in `data/`. The raw CSV (~470 MB, 6.36M rows) is **not** included in this repository. The dataset listing identifies the license as CC BY-SA 4.0. Attribute the original creators and review the license before redistribution.

Citation: Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. EMSS.

## Quick start (safe on an 8 GB laptop)
Python 3.10+ recommended. The repository includes the saved results and the two primary models, so the dashboard runs without the dataset.

```bash
git clone https://github.com/kanishka1114/-FinSentry.git FinSentry
cd FinSentry
python -m venv .venv
source .venv/bin/activate                       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                             # 92 tests on synthetic data (~35 s)
streamlit run app.py                            # open http://localhost:8501
```

To reproduce the results yourself, first download the dataset (above). Running the experiment **overwrites** the saved files in `outputs/` and `models/`:

```bash
python -m src.experiment --sample-rows 1000000  # recommended experiment: ~5.2 min, 0.73 GB peak RSS (measured)
```

The per-seed robustness models (`outputs/robustness/seed_*/models/`) are not published, because the dashboard does not use them. `python -m src.robustness` recreates them along with everything else in the study.

### The recommended experiment
`python -m src.experiment` is the single reproducible workflow behind the headline results. It uses a fixed seed (42), a **label-free uniform** sample of 1,000,000 rows, a chronological 70/15/15 split by whole steps, the train-only final fit, validation-only thresholds, row and block bootstrap, and a 5-window expanding backtest on the same sample. It writes:
- `outputs/experiment.json`: experiment id, full configuration, dataset SHA-256, split boundaries, train/backtest run ids, timings, library versions, status;
- `outputs/metrics.json`, `outputs/backtest.json`, `outputs/test_preview.csv`, `models/*.joblib`, all tagged with the same experiment id / run id;
- `outputs/report_summary.md`, built only after both stages finish.

The report and dashboard run a **consistency check**: they flag outputs from an older schema, metrics and backtest from different configurations or experiments, an unfinished experiment, run ids that don't match `experiment.json`, a preview from another run, and stratified (label-informed) sampling. Options: `--sample-rows N`, `--seed N`, `--skip-backtest` (deletes any old `backtest.json` so it cannot be mixed in), `--full` (never automatic).

`src.train` and `src.backtest` remain available for development runs. Every entry point refuses to use more than 1,000,000 rows unless `--full` is passed; full-data results have **not** been measured in this repository.

### Measured results (experiment `exp-0781f6ac1d`, uniform 1M-row sample, seed 42)
Test period steps 378–742: 150,442 rows, 627 fraud (prevalence 0.42% = PR-AUC baseline). Training prevalence was 0.08% and validation 0.05%, so the test period has strong prevalence shift.

| Model (max-F1 threshold from validation) | PR-AUC (block 95% CI) | Precision | Recall | False alerts | Missed fraud | Alerts / 1,000 |
|---|---|---|---|---|---|---|
| Rule baseline | n/a (no score) | 0.016 | 0.667 | 26,120 | 209 | 176.4 |
| Logistic Regression | 0.153 (0.105–0.223) | 0.356 | 0.100 | 114 | 564 | 1.18 |
| Random Forest | 0.350 (0.301–0.411) | 0.652 | 0.260 | 87 | 464 | 1.66 |

The Random Forest alert budget (5 per 1,000 chosen on validation) gave 4.55 per 1,000 on test, with precision 0.349 and recall 0.381. In the backtest the Random Forest PR-AUC ranged from 0.077 to 0.359 across 5 windows (the Logistic Regression range was 0.005–0.186). Full tables are in `outputs/report_summary.md`. These are results on a synthetic sample, not real-bank performance.

### Multi-seed robustness study (study `rob-de5ef78fad`, measured)
```bash
python -m src.robustness --sample-rows 1000000 --seeds 42 123 2026   # measured: 905 s total, 0.75 GB peak RSS
python -m src.robustness --aggregate-only --sample-rows 1000000 --seeds 42 123 2026   # rebuild summary only
```
The study runs the unchanged recommended experiment three times, each time on an independent uniform 1M-row sample with its own seed. Every run goes into `outputs/robustness/seed_<seed>/`, with its own models, `experiment.json` and report, so the primary outputs are never touched. A run is aggregated only if it finished and passed the consistency checks; incomplete or failed runs are listed and excluded. Results are in `outputs/robustness/robustness.json` and `outputs/robustness_summary.md`.

All three runs completed.

| Measured on test period (max-F1 threshold) | seed 42 | seed 123 | seed 2026 | Mean (SD) |
|---|---|---|---|---|
| Fraud prevalence (PR-AUC baseline) | 0.42% | 0.44% | 0.42% | 0.42% |
| Random Forest PR-AUC | 0.350 | 0.348 | 0.343 | 0.347 (0.004) |
| Logistic Regression PR-AUC | 0.153 | 0.145 | 0.140 | 0.146 (0.006) |
| Random Forest precision / recall | 0.649 / 0.260 | 0.775 / 0.200 | 0.564 / 0.301 | 0.663 / 0.254 |
| Random Forest false alerts / missed fraud | 88 / 464 | 38 / 524 | 146 / 438 | 90.7 / 475.3 |
| Random Forest alerts per 1,000 | 1.67 | 1.13 | 2.24 | 1.68 (0.56) |
| Logistic Regression precision / recall | 0.356 / 0.100 | 0.818 / 0.027 | 0.585 / 0.061 | 0.586 / 0.063 |

Findings:
- **The ranking quality is stable across samples.** Random Forest beat Logistic Regression on test PR-AUC in all 3 seeds, and in all 15 backtest seed-windows. Both models beat each window's fraud-rate baseline in 15 of 15 seed-windows.
- **Threshold-dependent numbers are much less stable than PR-AUC.** Random Forest max-F1 precision ranged 0.564–0.775 and false alerts 38–146. Logistic Regression precision ranged 0.356–0.818 at recall of only 0.03–0.10. The max-F1 threshold is chosen from a validation period with fewer than 100 fraud cases, so it moves between samples. Headline precision and false-alert counts should be read with that spread in mind.
- **The operational trade-off is consistent.** In every seed the alert-budget threshold traded precision for recall relative to max F1. For Random Forest, alert-budget recall was 0.381–0.402 at 4.6–6.7 alerts per 1,000.
- **Time matters more than the sample.** The typical across-window SD of backtest PR-AUC was 0.156 for Random Forest and 0.071 for Logistic Regression. The between-seed SD of the window mean was 0.019 and 0.002. Random Forest window PR-AUC ranged 0.024–0.388 over all seed-windows.

With three seeds, SDs are descriptive, not confidence intervals, and per-run bootstrap intervals are not pooled. All results come from 1M-row samples of synthetic PaySim data; they are neither full-data results nor real-bank performance.

**Reproducibility note.** Re-running seed 42 reproduced the primary experiment's sample, split, all Logistic Regression results and both PR-AUCs exactly. The one difference was a single Random Forest max-F1 false alert (88 vs 87): Random Forest prediction with `n_jobs=2` sums tree probabilities in thread order, and one transaction sat exactly at the threshold. Runs with `n_jobs=1` are bit-identical, which the tests check.

### Regenerating outputs
- Everything: `python -m src.experiment --sample-rows 1000000` (overwrites `outputs/` and `models/`).
- Report only, from the saved JSON: `python -m src.report`. The dashboard's Report page always rebuilds the report from the current JSON and offers to save it if the file on disk differs.
- `FINSENTRY_OUTPUT_DIR` / `FINSENTRY_MODEL_DIR` redirect outputs, for example to keep a second experiment separate.

## Evaluation protocol (leakage controls)
1. **Split by whole simulation steps**: train < validation < test (boundaries chosen by row share and saved).
2. **Fit on train only.**
3. **Select on validation only**: thresholds (max F1; alert budget) and the Platt calibrator.
4. **Final fit**
   - `--final-fit train_only` (default): the train-only model is final; no threshold transfer needed.
   - `--final-fit refit`: refit on train+validation. Thresholds are transferred by matching the train-only model's *in-sample* alert rate on the refit model's in-sample scores, and only if the train-only model's in-sample and validation alert rates agree within a factor of 2. Otherwise the refit is **rejected** and the reason recorded. This mode is **experimental** and is not used for headline results, because its assumptions cannot be checked without unseen pre-test data. In an earlier development run on 300k rows, this check rejected both models' refits.
5. **Score the untouched test period once.** Test labels/scores are used only for reporting.

Additional leakage decisions:
- Raw `step` is never a model input (only hour-of-day encodings), to avoid time shortcuts.
- Identifiers, balances and `isFlaggedFraud` are never loaded.
- `--sampling uniform` (default) never looks at labels. `--sampling stratified` keeps rare fraud but sizes each class from global label counts, which uses test-period labels to choose rows. It is for development only and is flagged in the report and dashboard.
- Backtest window merging (`--merge-sparse`) is **off** by default because it would let test-label counts move later windows' training boundaries.
- `tests/test_leakage.py` flips every test-period label and checks that thresholds, validation metrics, calibrator parameters, model predictions, explanation backgrounds and refit decisions are identical; it does the same for the last backtest window.

## Thresholds and alert strategy
- **Max F1** (validation): treats a missed fraud and a false alert as equally costly.
- **Alert budget** (validation): about `--alerts-per-1000` alerts per 1,000 transactions; fixes reviewer workload; labels not needed.

Reported for validation and test: precision, recall, F1, false positives, false negatives, alerts per 1,000, plus top-K / top-X% precision and recall. The dashboard's cost explorer picks the lowest-cost threshold on validation for a chosen cost ratio and shows its test outcome. **No threshold is universally best**; it depends on the cost of a missed fraud relative to a false alert and on review capacity. Score drift can change the realised test alert rate.

## Uncertainty
| Method | What it resamples | Captures | Misses |
|---|---|---|---|
| Stratified row bootstrap | fraud and legitimate rows separately (fixed fraud count) | which rows were observed | time dependence, drift, retraining |
| Moving-block bootstrap | contiguous blocks of whole steps (default length ≈ cube root of test steps) | short-range time dependence, varying fraud count | drift beyond the test period; unreliable with < 8 blocks or < 10 fraud rows (then skipped) |

Both use fixed seeds and bounded replicates (`--bootstrap`, `--block-bootstrap`, `--block-steps`). They are **estimates, not guarantees**, and both are conditional on the fitted model and the chosen thresholds. The dashboard and report show them side by side with a width ratio. In the recommended experiment, block intervals were 0.9–2.9× the width of row intervals for Logistic Regression and 1.1–2.5× for Random Forest. The row bootstrap understates uncertainty when fraud clusters in time. Replicates without fraud make recall/PR-AUC undefined; `n_valid` records how many replicates contributed.

## Calibration and drift
- **Calibration**: Brier score (with a constant-rate baseline), expected calibration error, and reliability bins on test; Platt calibrator fit on validation only (monotonic, so rankings and PR-AUC are unchanged). Class-weighted models are expected to be badly calibrated before calibration.
- **Drift**: population stability index (PSI) of scores from validation to test, score quantiles, fraud prevalence, alerts per 1,000 and PR-AUC baseline — per single split and per backtest window.
- These measure different things: **prevalence shift** (fraud rate changes), **ranking quality** (PR-AUC vs baseline, ROC-AUC), **calibration** (score values vs observed rates) and **threshold drift** (same threshold, different alert volume).

## Per-transaction explanations
- **Logistic Regression**: exact log-odds contributions `coef × (x − training mean)`.
- **Random Forest**: tree-path (Saabas) contributions; they sum exactly to the predicted score but can over-credit features split near leaves compared with TreeSHAP. SHAP is not required.
- Contributions are grouped into the three real inputs (amount, hour of day, transaction type). They describe model behaviour, **not causes**. Explanations are computed on demand for one row (Transaction Explorer, Try a Transaction).

## Temporal backtest
```bash
python -m src.backtest --sample-rows 1000000                     # 5 expanding windows
python -m src.backtest --sample-rows 1000000 --scheme rolling --train-window-frac 0.3
python -m src.backtest --sample-rows 300000                      # quick; some windows flagged sparse
```
The first `--initial-frac` (50%) of rows is history; the rest is cut into `--windows` successive test windows. Validation is the `--val-frac` (10%) of rows just before each test window; training is everything earlier (expanding) or a recent block (rolling). Each window reports step boundaries, rows, fraud, fraud rate, both threshold strategies, top-K, PR-AUC vs its own baseline, row and block intervals, calibration and drift, with warnings when fraud is scarce (`--min-test-fraud`, `--min-val-fraud`). No models are saved.

## Training options (main)
| Flag | Default | Meaning |
|---|---|---|
| `--sample-rows N` / `--sampling` | none / uniform | Sample size and method (label-free `uniform`, or `stratified` for development) |
| `--max-step N`, `--full` | none, off | Time limit; allow > 1M rows |
| `--split` | chronological | or `random` (comparison only) |
| `--final-fit` | train_only | or `refit` with checked threshold transfer |
| `--alerts-per-1000`, `--top-k`, `--top-pct`, `--alert-budget` | 5, 100, 0.5 1.0, 100 | Alert strategies |
| `--bootstrap`, `--block-bootstrap`, `--block-steps` | 300, =bootstrap, auto | Intervals |
| `--rf-trees`, `--rf-max-depth`, `--rf-min-leaf`, `--rf-max-samples`, `--n-jobs` | 100, 12, 20, 0.5, 2 | Random Forest sized for 8 GB |

## Dashboard pages
Overview · Transaction Explorer (with explanations) · Model Comparison · Thresholds & Alerts · Temporal Backtest (with drift) · Robustness Study · Reliability & Calibration · Explainability · Try a Transaction (with explanations) · Report Summary. Everything is read from saved outputs; startup does no training or bootstrapping.

The dashboard uses a dark theme (`.streamlit/config.toml`) and a small set of presentation helpers in `src/ui.py`: KPI cards with hover definitions, consistent number formatting (undefined values are shown as "undefined", never as 0), a shared Plotly template and chart subtitles. Every page labels the data as synthetic PaySim. The Robustness Study page reads `outputs/robustness/robustness.json` and is kept separate from the primary experiment; if no study exists it shows the command to create one.

## Outputs
- `outputs/experiment.json`: configuration, fingerprint, boundaries, run ids, timings, versions (recommended workflow only)
- `outputs/robustness/seed_<seed>/`: one complete, isolated experiment per seed; `outputs/robustness/robustness.json` and `outputs/robustness_summary.md` hold the aggregated study
- `models/*.joblib`: final pipeline, thresholds, run id, final-fit mode, explanation background
- `outputs/metrics.json`: run metadata, dataset fingerprint, sampling, protocol and split boundaries, per-model strategies, intervals (row and block), calibration, drift, warnings
- `outputs/backtest.json`: per-window boundaries, counts, metrics, intervals, calibration, drift; cross-window summary
- `outputs/test_preview.csv`: natural random test rows plus labelled fraud examples and top alerts
- `outputs/report_summary.md`: measured results (section A) separated from explanatory text (section B)

## Demo walkthrough (viva)
See [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) for launch commands, a timed 3–5 minute walkthrough, plain-language metric explanations, interview questions and troubleshooting. [`docs/VIVA_GUIDE.md`](docs/VIVA_GUIDE.md) gives a project overview with likely questions and answers. [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) lists the checks to repeat before sharing the project.

1. **Overview**: synthetic-data note, KPI cards, ranking quality vs baseline, operational errors, prevalence shift, final test table.
2. **Thresholds & Alerts**: max F1 vs alert budget; why max-F1 numbers are unstable; cost explorer.
3. **Temporal Backtest**: PR-AUC per window vs that window's baseline; drift diagnostics.
4. **Robustness Study**: the same experiment on three samples; time varies results more than the sample does.
5. **Try a Transaction**: score a transfer with both models and explain the contributions.
6. **Report Summary**: download the generated summary.

## Limitations
PaySim is synthetic. Sample runs contain few fraud cases. Intervals are estimates under stationarity. Thresholds may not transfer under drift. Only amount, type and hour are inputs, by design. Explanations are not causal. Alerts need human review; not suitable for production.
