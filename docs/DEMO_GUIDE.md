# FinSentry demo guide

This guide is for presenting the dashboard: how to start it, what to show in 3–5 minutes, how to explain the metrics to a non-specialist, and how to answer common questions honestly.

All numbers below come from saved outputs:
- the primary experiment `exp-0781f6ac1d` (`outputs/metrics.json`, `outputs/backtest.json`);
- the robustness study `rob-de5ef78fad` (`outputs/robustness/robustness.json`).

Both use 1,000,000-row uniform samples of the **synthetic** PaySim dataset. Nothing here describes a real bank, real customers or a deployed system. If you re-run the experiment, the dashboard updates automatically, but the numbers in this guide will not.

## 1. Setup and launch

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints, normally <http://localhost:8501>. Startup reads saved JSON and models only and takes a few seconds; it never trains anything.

If `outputs/` is empty (for example on a fresh clone), the dashboard shows the commands to create results instead of crashing:

```bash
# needs data/PS_20174392719_1491204439457_log.csv from Kaggle (see README)
python -m src.experiment --sample-rows 1000000                       # ~5 min, ~0.73 GB peak RAM
python -m src.robustness --sample-rows 1000000 --seeds 42 123 2026   # optional, ~15 min
```

Before a live demo:
- Run `python -m pytest -q`. The last recorded run was 92 passed.
- Open every page once so caches are warm.

## 2. A 3–5 minute walkthrough

| Time | Page | What to show and say |
|---|---|---|
| 0:00 | **Overview** | Point at the amber "Synthetic PaySim data" badge and the note under it. Say: "This is a research prototype on simulated mobile-money data, evaluated on a 1M-row sample." In the KPI row, fraud is **0.42%** of the 150,442 test transactions. Random Forest PR-AUC is **0.350**, about **84×** the 0.0042 random baseline. Hover any card for its definition. |
| 0:45 | **Overview** charts | *Ranking quality*: both models are far above the dashed baseline. *Operational errors*: the rule catches more fraud but raises **26,120** false alerts, against **87** for Random Forest. *Prevalence shift*: fraud is 0.08% in train, 0.05% in validation and 0.42% in test, so precision is not comparable between periods. |
| 1:30 | **Thresholds & Alerts** | Two ways to turn scores into alerts, both chosen on validation only. Random Forest at **max F1**: precision 65.2%, recall 26.0%, 1.66 alerts per 1,000. With an **alert budget** of about 5 per 1,000: precision 34.9%, recall 38.1%, 4.55 per 1,000. Read the amber note: max-F1 numbers moved a lot between samples. Optionally move the cost-ratio input in the cost explorer. |
| 2:30 | **Temporal Backtest** | Five later time windows, each trained only on the past. Random Forest PR-AUC ranged **0.077–0.359** across windows, always above that window's baseline. "One headline number hides how much performance moves over time." |
| 3:15 | **Robustness Study** | The same experiment on three independent samples (seeds 42, 123, 2026). PR-AUC barely moved (Random Forest 0.343–0.350), but max-F1 precision ranged 56.4–77.5% and false alerts 38–146. *Where the variation comes from*: across time windows the SD was 0.156, between samples 0.019. "Time matters more than which sample we drew." |
| 4:00 | **Try a Transaction** | Keep TRANSFER and 150,000, then press **Analyze transaction**. The cards show each model's score against its thresholds, and the bar charts show which inputs pushed the score up or down. "These explain the model's behaviour, not the cause of fraud." |
| 4:40 | **Report Summary** | The report is rebuilt from the saved JSON every time, with download buttons for the primary report and the robustness summary. |

If you only have 3 minutes, skip Temporal Backtest and say one sentence about it on the Robustness Study page.

## 3. The metrics in plain language

Use the Random Forest's test results as the running example. The test period has 150,442 transactions, of which 627 are fraud. At the max-F1 threshold the model raised 250 alerts; 163 of them were fraud.

- **Precision**: of the transactions we alerted on, how many were really fraud? 163 / 250 = **65.2%**. Low precision means reviewers waste time on false alerts.
- **Recall**: of all the fraud, how much did we catch? 163 / 627 = **26.0%**. Low recall means fraud slips through.
- **False alerts** (false positives): legitimate transactions we flagged. **87** here, each costing a reviewer's time and possibly annoying a customer.
- **Missed fraud** (false negatives): fraud we did not flag. **464** here, each a potential loss.
- **Alert volume**: alerts per 1,000 transactions (**1.66** here). This is the reviewer workload.
- **PR-AUC**: precision and recall averaged over *every* possible threshold, which measures how well the model ranks fraud above legitimate transactions. A model that guesses randomly scores the fraud rate (0.0042 here), not 0.5. So always compare PR-AUC with that baseline: 0.350 is about 84× better than random. We don't use accuracy: flagging nothing would score 99.6% accurate on this test period and catch no fraud.
- **Threshold selection**: a model outputs a score, and a threshold turns it into "alert / no alert". A higher threshold gives fewer alerts, higher precision and lower recall. FinSentry picks thresholds on the validation period only, in two ways:
  - **max F1** balances precision and recall equally;
  - an **alert budget** fixes the workload (about 5 alerts per 1,000).
  Neither is "correct"; the right one depends on how costly a missed fraud is compared with a false alert, and how many reviewers you have.
- **Why max F1 can be unstable**: the validation period contains fewer than 100 fraud cases, so the max-F1 threshold depends on a handful of examples. In the robustness study, the same method on three samples gave Random Forest precision anywhere from 56.4% to 77.5%. Treat threshold-dependent numbers as one sample's outcome, not a fixed property of the model.
- **Temporal variation**: fraud patterns and fraud rates change over time. In the walk-forward backtest, Random Forest PR-AUC ranged from 0.077 to 0.359 depending on the time window, while each window's random baseline ranged from 0.0005 to 0.0058. A model evaluated on one period can look much better or worse on the next.

## 4. Interview questions and honest answers

**Is this deployed, or validated at a bank?**
No. PaySim is a synthetic simulator, and every result comes from a 1M-row sample of it. Nothing has been tested on real transactions, and the full 6.36M-row dataset has not been used.

**Why is precision only 65% if the model is "84× better than random"?**
Fraud is extremely rare (0.42%), so even a good ranker produces false alerts once you alert on more than a few hundred transactions. On ranking alone it does well: among its top 100 scores, 96 were fraud.

**Why does the rule baseline look so bad?**
It catches 66.7% of fraud, but it alerts on about 176 of every 1,000 transactions (26,120 false alerts). No review team could handle that. It is there as a transparent reference point.

**Why is the test fraud rate so different from training?**
PaySim's fraud rate changes over simulated time: 0.08% in train, 0.05% in validation, 0.42% in test. The split is chronological on purpose, so that this shift is visible instead of averaged away.

**How confident are you in these numbers?**
- The model ranking is stable: Random Forest beat Logistic Regression in all 3 samples and all 15 backtest windows.
- Test PR-AUC had a between-sample SD of 0.004 for Random Forest and 0.006 for Logistic Regression.
- Threshold-dependent numbers varied much more.
- Bootstrap intervals are reported too, but they are estimates and assume rough stationarity.
- Three seeds are a sanity check, not a confidence interval.

**Why does seed 42 in the robustness study differ from the primary run?**
By one Random Forest false alert (88 vs 87). Parallel tree prediction adds probabilities in thread order, and one transaction sat exactly on the threshold. Everything else matched exactly, and single-threaded runs are bit-identical (this is tested).

**Could the model be cheating by seeing the future?**
The protocol is built to prevent that:
- data is split by time;
- models are fit on the past only;
- thresholds and calibration use validation only;
- the test period is scored once;
- a test flips every test label and checks that nothing the model or thresholds depend on changes.

**Why not use the balance columns? They would boost performance.**
In PaySim, balance fields nearly reveal the label, so results would look great and mean little. They are excluded by design. Testing them properly would need a separate leakage analysis.

**What do the explanations tell you?**
How each input moved this model's score away from its baseline. They describe the model's behaviour, not why fraud happens. Random Forest starts from 50% because balanced class weights make its root node 50/50.

**What would you do next?**
- More seeds and a more stable threshold rule.
- Nested temporal validation.
- Per-period recalibration.
- A TreeSHAP comparison.
- Full-data runs on bigger hardware.

## 5. Startup troubleshooting

| Symptom | Fix |
|---|---|
| `streamlit: command not found` or `ModuleNotFoundError` | Activate the environment (`source .venv/bin/activate`) or run `.venv/bin/streamlit run app.py`. Reinstall with `pip install -r requirements.txt` if needed. |
| `Port 8501 is already in use` | Another Streamlit is running. Stop it (Ctrl+C in its terminal) or use `streamlit run app.py --server.port 8502`. |
| Dashboard says "No experiment results were found" and shows commands | `outputs/metrics.json` is missing. Run `python -m src.experiment --sample-rows 1000000` (needs the dataset in `data/`). |
| Light theme or default colours | Start Streamlit from the repository root so it finds `.streamlit/config.toml`. |
| Red consistency error in the sidebar or on Overview | Outputs come from different runs or an older version. Re-run `python -m src.experiment --sample-rows 1000000`; do not mix files from different runs. |
| "belongs to a different training run" on Transaction Explorer | `outputs/test_preview.csv` is from another run. Re-run the experiment. |
| Robustness Study page shows only a command | No study in `outputs/robustness/`. Run the robustness command above (~15 min), or skip that page. |
| "File change" banner with a **Rerun** button | Code was edited while the app was running. Click **Rerun**, or restart Streamlit if a change in `src/` does not appear. |
| Message suggesting the Watchdog module | Harmless; it only affects how fast file changes are detected. |
| Blank page or stuck spinner | Refresh the browser tab. If it persists, stop the server, then start it again from the repository root. |
| Using a different output folder | `FINSENTRY_OUTPUT_DIR=/path/to/outputs FINSENTRY_MODEL_DIR=/path/to/models streamlit run app.py` |
