# FinSentry-Mini

A small machine-learning project that flags fraudulent transactions in **PaySim**, a *simulated* mobile-money dataset. It compares a **simple rule** with a **Random Forest** model on later transactions that the model never saw.

## Results (test period, simulated data)

| Approach | Fraud caught (of 627) | False alerts | Missed fraud | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Simple rule | 418 | 26,120 | 209 | 0.02 | 0.67 | 0.03 |
| Random Forest | 405 | 24 | 222 | 0.94 | 0.65 | 0.77 |

- **Simple rule:** alert on a TRANSFER or CASH_OUT of at least 200,000, or on any transaction of at least 1,000,000.
- **Random Forest:** many decision trees vote; it alerts when the score is at least 0.741.
- **Precision:** of the alerts, how many were fraud. **Recall:** of the fraud, how much was caught. **F1:** balances the two (0 to 1).
- Accuracy is not used: fraud is so rare that never alerting would already be over 99% accurate.

## How it works
1. **Data:** a random sample of 1,000,000 PaySim transactions (seed 42); 1,279 are fraud.
2. **Inputs:** amount, transaction type, hour of day, and the sender's balance before and after the transaction. No names or account numbers.
3. **Time split:** train (hours 1–323) to teach the model, validation (324–377) to choose the alert threshold, test (378–742) used once for the final results.
4. **Threshold:** the score cut-off with the best F1 on the validation period.

## What changed from the first version
The first version used only amount, type and hour, and reached F1 0.37 (163 fraud caught, 87 false alerts). Adding the sender's balance before and after raised F1 to 0.77 (405 caught, 24 false alerts), with the same sample, split, model settings and threshold method. In PaySim, fraud often empties the sender's account, which the amount alone cannot show. Logistic Regression was dropped to keep the comparison simple.

The first version's experiment (`exp-0781f6ac1d`, run `a7545b9ebf9d`, 2026-10-20) is kept in this repository's git history and in [FinSentry-original-archive](https://github.com/kanishka1114/FinSentry-original-archive).

## Files
- `finsentry.py`: shared code for loading the data, the time split, the model inputs, the rule and the metrics.
- `train.py`: trains the Random Forest, chooses the threshold and saves `models/random_forest.joblib` and `outputs/results.json`.
- `verify.py`: re-scores the test period with the saved model and checks every saved figure. It trains nothing.
- `Results.py` and `pages/`: the dashboard. It only reads the saved model and results.
- The PaySim CSV (about 471 MB) is **not** included. Download it from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) to train or verify.

## Run the dashboard
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # versions match the saved model
streamlit run Results.py
```

Pages: **Results**, **How it works** and **Try a transaction**. On Try a transaction, enter an amount, type, hour and the sender's balances, or click **Load fraud example (alert)** or **Load normal example (no alert)**, both real test-period transactions. A score at or above the threshold raises an alert; a score below it gets no alert.

## Train and verify (needs the PaySim CSV)
```bash
python train.py path/to/PS_20174392719_1491204439457_log.csv    # about a minute
python verify.py path/to/PS_20174392719_1491204439457_log.csv   # ends with "Check passed"
```

## Limitation
PaySim is simulated. These results show how the model did on this simulated data, not how it would perform on real bank transactions. The threshold was chosen on only 79 validation frauds, so results at that threshold can vary.
