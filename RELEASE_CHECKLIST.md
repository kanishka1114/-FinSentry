# FinSentry release checklist

Repeat before sharing, submitting or demoing the project. Run every command from the repository root inside the virtual environment.

## 1. Code health
- [ ] `python -m pytest -q`: all tests pass. The last recorded run was 92 passed.
- [ ] `python -m compileall -q src app.py tests` prints nothing.
- [ ] `requirements.txt` has no new or unused dependencies.

## 2. Regenerate results
- [ ] The dataset is at `data/PS_20174392719_1491204439457_log.csv`.
- [ ] Run `python -m src.experiment --sample-rows 1000000`. Last measured: 310 s and 0.73 GB peak RSS on an 8 GB Mac.
- [ ] `outputs/experiment.json` has `"status": "complete"`.
- [ ] The report shows **Consistency check: passed**, and all three sources share one experiment id.
- [ ] The report says the data was a "SAMPLE (uniform)". No full-data claims are made anywhere.

## 2b. Robustness study (optional, about 15 min)
- [ ] Run `python -m src.robustness --sample-rows 1000000 --seeds 42 123 2026`. Last measured: 905 s and 0.75 GB peak RSS.
- [ ] `outputs/robustness/robustness.json` has `"status": "complete"` and `n_aggregated` equals the number of seeds.
- [ ] `outputs/metrics.json` and the other primary outputs are unchanged. The study writes only under `outputs/robustness/` and `outputs/robustness_summary.md`.
- [ ] The README, report template and viva guide quote the current study id (`rob-de5ef78fad`).

## 3. Dashboard
- [ ] `streamlit run app.py` starts in a few seconds with no training on startup.
- [ ] All 10 pages open without errors: Overview, Transaction Explorer, Model Comparison, Thresholds & Alerts, Temporal Backtest, Robustness Study, Reliability & Calibration, Explainability, Try a Transaction, Report Summary.
- [ ] The header shows the "Synthetic PaySim data" badge and the primary experiment id; the sidebar shows "Outputs consistent".
- [ ] The Robustness Study page shows the study id and "3 of 3 seeds" without warnings. Its numbers match `outputs/robustness_summary.md`.
- [ ] Visual check (manual): the active sidebar item is highlighted, KPI cards show a definition on hover, and no chart legend overlaps an axis title. Narrow the browser window to about 800 px: KPI cards should wrap and nothing should scroll sideways.
- [ ] The sidebar shows sampling "uniform (label-free)" and the experiment id.
- [ ] Scoring a transaction under Try a Transaction shows scores plus a contribution chart for each model.
- [ ] Report Summary downloads the Markdown file, and there is no "older than the current outputs" notice.
- [ ] With an empty output folder, the app shows the commands to run instead of crashing. Check with `FINSENTRY_OUTPUT_DIR=/tmp/empty streamlit run app.py`.

## 4. Documentation
- [ ] The README numbers match `outputs/report_summary.md` for the current experiment id. If you re-ran the experiment, update the README table.
- [ ] `docs/VIVA_GUIDE.md` quotes the same numbers.
- [ ] Every document says that PaySim is synthetic and that the results are not real-bank performance.
- [ ] Dataset attribution and the license note are present.

## 5. Do not ship
- [ ] The raw dataset in `data/` (license, size).
- [ ] `.venv/`, `__pycache__/`, `.pytest_cache/`.
- [ ] Outputs from stratified or mixed runs. The consistency check flags them.
