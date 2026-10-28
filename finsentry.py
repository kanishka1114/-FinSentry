"""Shared FinSentry code: data, time split, model inputs, the simple rule and the metrics.

Used by train.py, verify.py and the dashboard, so every part builds the inputs the same way.
"""
import numpy as np
import pandas as pd

CSV_NAME = "PS_20174392719_1491204439457_log.csv"
COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud"]

# Experiment settings; changing them changes every result.
SAMPLE_ROWS = 1_000_000
SEED = 42
TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FEATURES = ["amount", "log_amount", "hour_sin", "hour_cos", "type",
            "sender_balance_before", "sender_balance_after"]
NUMERIC_FEATURES = [f for f in FEATURES if f != "type"]

RULE_TYPES = ("TRANSFER", "CASH_OUT")
RULE_LARGE_AMOUNT = 200_000
RULE_VERY_LARGE_AMOUNT = 1_000_000
RULE_DESCRIPTION = ("Alert on a TRANSFER or CASH_OUT of at least 200,000, or on any "
                    "transaction of at least 1,000,000.")


def load_paysim(csv_path):
    """Read the columns used here and drop rows with invalid values."""
    df = pd.read_csv(csv_path, usecols=COLUMNS)
    valid = (df["type"].isin(TRANSACTION_TYPES) & (df["amount"] >= 0) & (df["step"] >= 1)
             & (df["oldbalanceOrg"] >= 0) & (df["newbalanceOrig"] >= 0)
             & df["isFraud"].isin([0, 1]))
    return df[valid].reset_index(drop=True)


def uniform_sample(df, n_rows, seed):
    """Pick n_rows at random without looking at the fraud label, then sort by time."""
    chosen = np.sort(np.random.default_rng(seed).choice(len(df), n_rows, replace=False))
    return df.iloc[chosen].sort_values("step", kind="stable").reset_index(drop=True)


def chronological_split(steps, train_share, validation_share):
    """Split by whole hourly steps so train < validation < test in time.

    Each boundary is the step whose cumulative row share is closest to the target share.
    """
    unique_steps, counts = np.unique(steps, return_counts=True)
    cumulative_share = np.cumsum(counts) / counts.sum()

    def last_step_index(target):
        i = int(np.searchsorted(cumulative_share, target, side="left"))
        if i > 0 and (i >= len(cumulative_share)
                      or target - cumulative_share[i - 1] <= cumulative_share[i] - target):
            i -= 1
        return i

    train_end = min(last_step_index(train_share), len(unique_steps) - 3)
    validation_end = min(max(last_step_index(train_share + validation_share), train_end + 1),
                         len(unique_steps) - 2)
    train_last, validation_last = unique_steps[train_end], unique_steps[validation_end]
    return {"train": steps <= train_last,
            "validation": (steps > train_last) & (steps <= validation_last),
            "test": steps > validation_last}


def build_features(amount, kind, hour, balance_before, balance_after):
    """The model inputs. Each argument may be a single value or a pandas Series.

    Hour of day is the PaySim step modulo 24, written as a point on a circle so that
    hour 23 sits next to hour 0.
    """
    frame = pd.DataFrame({"amount": amount, "type": kind, "hour": hour,
                          "before": balance_before, "after": balance_after},
                         index=getattr(amount, "index", [0]))
    amount = frame["amount"].astype(float).clip(lower=0)
    hour = frame["hour"].astype(float)
    return pd.DataFrame({
        "amount": amount,
        "log_amount": np.log1p(amount),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "type": frame["type"].astype(str),
        "sender_balance_before": frame["before"].astype(float).clip(lower=0),
        "sender_balance_after": frame["after"].astype(float).clip(lower=0),
    })[FEATURES]


def features_from_paysim(df):
    return build_features(df["amount"], df["type"], df["step"] % 24,
                          df["oldbalanceOrg"], df["newbalanceOrig"])


def rule_alerts(df):
    amount = df["amount"].astype(float)
    risky = df["type"].astype(str).isin(RULE_TYPES) & (amount >= RULE_LARGE_AMOUNT)
    return (risky | (amount >= RULE_VERY_LARGE_AMOUNT)).to_numpy()


def alert_metrics(y_true, alerts):
    """Counts and rates for a set of yes/no alerts."""
    y_true, alerts = np.asarray(y_true).astype(bool), np.asarray(alerts).astype(bool)
    tp = int((alerts & y_true).sum())
    fp = int((alerts & ~y_true).sum())
    fn = int((~alerts & y_true).sum())
    tn = int((~alerts & ~y_true).sum())
    return {
        "fraud_caught": tp, "false_alerts": fp, "missed_fraud": fn, "correctly_ignored": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
        "accuracy": (tp + tn) / len(y_true),
    }
