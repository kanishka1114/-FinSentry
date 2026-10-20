import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import (DATA_FILE, USECOLS, CSV_DTYPES, READ_CHUNKSIZE,
                        TRANSACTION_TYPES, FEATURES, SEED)

TYPE_DTYPE = pd.CategoricalDtype(TRANSACTION_TYPES)


def resolve_path(path=None):
    path = Path(path) if path else DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found: {path}. Download from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 and place it in data/."
        )
    return path


def file_fingerprint(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block_size):
            digest.update(chunk)
    return {"file_name": Path(path).name, "size_bytes": Path(path).stat().st_size,
            "sha256": digest.hexdigest()}


def load_data(path=None, max_step=None, chunksize=READ_CHUNKSIZE):
    """Read only the needed columns in chunks with compact dtypes.

    Returns (df, info) where info records how many rows were read, dropped and kept.
    """
    path = resolve_path(path)
    header = pd.read_csv(path, nrows=0).columns
    missing = set(USECOLS) - set(header)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {sorted(missing)}")

    parts, rows_read, rows_invalid = [], 0, 0
    reader = pd.read_csv(path, usecols=USECOLS, chunksize=chunksize,
                         dtype={"type": "string"})
    for chunk in reader:
        rows_read += len(chunk)
        chunk = _clean(chunk)
        rows_invalid += chunk.attrs.pop("invalid", 0)
        if max_step is not None:
            chunk = chunk[chunk["step"] <= max_step]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        raise ValueError("No valid rows left after cleaning/max_step filtering.")
    df = pd.concat(parts, ignore_index=True)
    info = {"rows_in_file": int(rows_read), "rows_invalid_dropped": int(rows_invalid),
            "max_step": max_step, "rows_after_filter": int(len(df)),
            "fraud_after_filter": int(df["isFraud"].sum())}
    return df, info


def _clean(chunk):
    kind = chunk["type"].astype("string").str.strip()
    out = pd.DataFrame({
        "step": pd.to_numeric(chunk["step"], errors="coerce"),
        "type": kind.where(kind.isin(TRANSACTION_TYPES)).astype(TYPE_DTYPE),
        "amount": pd.to_numeric(chunk["amount"], errors="coerce"),
        "isFraud": pd.to_numeric(chunk["isFraud"], errors="coerce"),
    })
    valid = (out[["step", "type", "amount", "isFraud"]].notna().all(axis=1)
             & out["isFraud"].isin([0, 1]) & (out["amount"] >= 0) & (out["step"] >= 1))
    out = out[valid].astype({"step": CSV_DTYPES["step"], "amount": CSV_DTYPES["amount"],
                             "isFraud": CSV_DTYPES["isFraud"]})
    out.attrs["invalid"] = int((~valid).sum())
    return out


def stratified_sample(df, n_rows, seed=SEED):
    """Reproducible random sample across all steps that keeps the natural fraud rate.

    Each class is sampled proportionally; at least one fraud row is kept when the
    source has fraud. Output stays sorted by step so time order is preserved.
    """
    if n_rows is None or n_rows >= len(df):
        return df.sort_values("step", kind="stable").reset_index(drop=True)
    if n_rows <= 0:
        raise ValueError("sample_rows must be positive.")
    rng = np.random.default_rng(seed)
    fraud_idx = df.index[df["isFraud"] == 1].to_numpy()
    legit_idx = df.index[df["isFraud"] == 0].to_numpy()
    n_fraud = int(round(n_rows * len(fraud_idx) / len(df)))
    if len(fraud_idx):
        n_fraud = min(max(n_fraud, 1), len(fraud_idx), n_rows)
    n_legit = min(n_rows - n_fraud, len(legit_idx))
    chosen = np.concatenate([rng.choice(fraud_idx, n_fraud, replace=False),
                             rng.choice(legit_idx, n_legit, replace=False)])
    chosen.sort()
    return df.loc[chosen].sort_values("step", kind="stable").reset_index(drop=True)


def uniform_sample(df, n_rows, seed=SEED):
    """Reproducible random sample that never looks at labels (fraud count is random)."""
    if n_rows is None or n_rows >= len(df):
        return df.sort_values("step", kind="stable").reset_index(drop=True)
    if n_rows <= 0:
        raise ValueError("sample_rows must be positive.")
    chosen = np.sort(np.random.default_rng(seed).choice(len(df), n_rows, replace=False))
    return df.iloc[chosen].sort_values("step", kind="stable").reset_index(drop=True)


SAMPLERS = {"stratified": stratified_sample, "uniform": uniform_sample}


def build_features(df):
    """Model inputs from step/type/amount. Raw step is not an output column."""
    hour = df["step"].astype("int64") % 24
    amount = df["amount"].astype("float64").clip(lower=0)
    return pd.DataFrame({
        "amount": amount,
        "log_amount": np.log1p(amount),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "type": df["type"].astype(TYPE_DTYPE).astype(str),
    }, index=df.index)[FEATURES]


def prepare_features(df):
    return build_features(df), df["isFraud"].astype(int)
