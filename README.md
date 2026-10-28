# FinSentry-Mini

A small machine-learning project on fraud detection in **PaySim**, a *simulated* mobile-money dataset. It presents one saved experiment (`exp-0781f6ac1d`) from the original FinSentry project: a fixed rule, Logistic Regression and Random Forest, compared on a time-based test period. Nothing here trains a model.

## What it does
- **`finsentry_mini.py`** rebuilds the experiment's data: a 1,000,000-row random sample (seed 42), split by time into train (steps 1–323), validation (324–377) and test (378–742). It then scores the test period with the two saved models at their saved thresholds and checks every figure against the saved `outputs/metrics.json`.
- **Dashboard, three pages:**
  - **Results:** the data split and the final test results for all three approaches.
  - **How the threshold works:** how each model's threshold was chosen on the validation period.
  - **Try a transaction:** scores one example you enter with a saved model.

  The first two pages only read saved results. The third uses copies of the two saved models in `models/`.

Thresholds were chosen by maximising F1 on the **validation** period. The **test** period was used once, only to report the final results.

## What is included
- `outputs/metrics.json`: the saved results of the experiment, an exact copy of the original project's file.
- `models/`: the two saved models, exact copies of the original project's files.
- The PaySim CSV (about 471 MB) is **not** included.

Provenance: every figure on the dashboard comes from experiment `exp-0781f6ac1d`, run `a7545b9ebf9d` (2026-10-20), as saved in `outputs/metrics.json`. The dashboard does not train, re-score or re-tune anything except the single transaction you enter on the Try a transaction page.

## Run the dashboard
The dashboard needs only the files in this repository.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # versions match the saved models
streamlit run Results.py               # Streamlit prints the local address
```

## Verify the saved results (needs the PaySim CSV)
`finsentry_mini.py` is a verification script. It does not train anything. It rebuilds the test period from the PaySim CSV, scores it with the saved models and checks every figure against `outputs/metrics.json`. Download the CSV from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1), then either place it in `data/` or pass its path:

```bash
python finsentry_mini.py path/to/PS_20174392719_1491204439457_log.csv
```

It ends with "every figure above matches" or lists any mismatch and exits with an error. The script scores the Random Forest with `n_jobs=1`, set in memory only. One test transaction's score sits exactly on the threshold, and parallel scoring can move it by one bit, so serial scoring is used to make the result repeatable.

## Try a transaction
Open **Try a transaction** in the sidebar. Choose a model, enter an amount, a transaction type and an hour of day (0–23), then click **Check transaction**. The page shows what you entered, the model score, the saved threshold and whether an alert is raised. The model raises an alert when its score is at or above the threshold; a score below the threshold gets no alert.

To see an alert straight away, click **Load Logistic Regression example** or **Load Random Forest example**. Each fills in a TRANSFER from the test period that the saved model alerts on (Logistic Regression: 9,725,837.08 at hour 14, score about 0.999929 against threshold 0.997993; Random Forest: 461,369.99 at hour 7, score about 0.795842 against threshold 0.766811). Any input that scores below the threshold gets no alert; the thresholds are never changed to make an input alert.

The score is a model output used for ranking, not a guaranteed probability of fraud. This is a classroom demonstration with models trained on simulated data, so an alert does not prove a transaction is fraud. Hour of day is the PaySim simulation step modulo 24, as in training; it is not verified clock time.

## What the metrics mean
- **Precision:** share of alerts that were fraud.
- **Recall:** share of fraud that was alerted.
- **F1:** harmonic mean of precision and recall.
- **Fraud caught / false alerts / missed fraud:** true positives, false positives and false negatives.
- **PR-AUC:** summarises precision and recall over all thresholds. Fraud is rare (0.42% of test transactions), so random guessing scores about 0.0042, not 0.5.

Accuracy is not reported, because flagging nothing would already be over 99% accurate.

## Main limitation
PaySim is simulated, and only three raw inputs are used (amount, transaction type and hour). The results show how these models behaved on one sample of simulated data, not how they would perform on real transactions. Results at a single threshold are also sensitive to the few fraud cases in the validation period (79), which is what the thresholds were chosen from.
