import numpy as np
import pandas as pd
import pytest


def make_paysim_like(n=20_000, steps=200, fraud_rate=0.01, seed=0):
    rng = np.random.default_rng(seed)
    kinds = np.array(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    df = pd.DataFrame({
        "step": np.sort(rng.integers(1, steps + 1, n)),
        "type": rng.choice(kinds, n),
        "amount": np.round(rng.lognormal(10, 1.5, n), 2),
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": 1.0, "newbalanceOrig": 0.0,
        "nameDest": [f"M{i}" for i in range(n)],
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "isFraud": 0, "isFlaggedFraud": 0,
    })
    risky = df["type"].isin(["TRANSFER", "CASH_OUT"]).to_numpy()
    fraud = risky & (rng.random(n) < fraud_rate * 2.5)
    df.loc[fraud, "isFraud"] = 1
    df.loc[fraud, "amount"] *= 5
    return df


@pytest.fixture
def paysim_csv(tmp_path):
    path = tmp_path / "paysim_small.csv"
    make_paysim_like().to_csv(path, index=False)
    return path
