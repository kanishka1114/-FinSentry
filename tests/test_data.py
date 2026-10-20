import numpy as np
import pandas as pd
import pytest
from src.config import FEATURES
from src.data_loader import load_data, stratified_sample, build_features, prepare_features


def test_load_reads_only_needed_columns_with_compact_dtypes(paysim_csv):
    df, info = load_data(paysim_csv, chunksize=3_000)
    assert list(df.columns) == ["step", "type", "amount", "isFraud"]
    assert df["isFraud"].dtype == np.int8
    assert df["step"].dtype == np.int32
    assert isinstance(df["type"].dtype, pd.CategoricalDtype)
    assert info["rows_in_file"] == info["rows_after_filter"] == len(df) == 20_000


def test_load_drops_invalid_rows(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"step": [1, 2, 3, "x"], "type": ["PAYMENT", "BOGUS", "TRANSFER", "PAYMENT"],
                  "amount": [10, 20, -5, 1], "isFraud": [0, 0, 1, 0]}).to_csv(path, index=False)
    df, info = load_data(path)
    assert len(df) == 1 and info["rows_invalid_dropped"] == 3


def test_load_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"step": [1], "amount": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_data(path)


def test_max_step_filter(paysim_csv):
    df, info = load_data(paysim_csv, max_step=50)
    assert df["step"].max() <= 50 and info["max_step"] == 50


def test_sample_is_reproducible_stratified_and_spread(paysim_csv):
    df, _ = load_data(paysim_csv)
    a = stratified_sample(df, 4_000, seed=7)
    b = stratified_sample(df, 4_000, seed=7)
    c = stratified_sample(df, 4_000, seed=8)
    assert len(a) == 4_000
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    expected_fraud = round(4_000 * df["isFraud"].mean())
    assert abs(int(a["isFraud"].sum()) - expected_fraud) <= 1
    # Spread across the whole timeline, not just the first rows.
    assert a["step"].min() <= 5 and a["step"].max() >= 195
    assert a["step"].is_monotonic_increasing


def test_sample_keeps_rare_fraud():
    df = pd.DataFrame({"step": np.arange(1, 10_001), "type": "TRANSFER",
                       "amount": 1.0, "isFraud": 0})
    df.loc[5_000, "isFraud"] = 1
    assert stratified_sample(df, 100, seed=1)["isFraud"].sum() == 1


def test_sample_larger_than_data_returns_all(paysim_csv):
    df, _ = load_data(paysim_csv)
    assert len(stratified_sample(df, 10**9)) == len(df)


def test_features_exclude_step_and_ids():
    df = pd.DataFrame({"step": [1, 13, 25], "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
                       "amount": [0.0, 100.0, 50.0], "isFraud": [0, 1, 0]})
    X, y = prepare_features(df)
    assert list(X.columns) == FEATURES
    assert "step" not in X.columns
    assert y.tolist() == [0, 1, 0]
    # Steps 1 and 25 share hour-of-day 1, so their hour encodings match.
    assert np.isclose(X.loc[0, "hour_sin"], X.loc[2, "hour_sin"])
    assert np.isclose(X.loc[1, "log_amount"], np.log1p(100.0))


def test_build_features_handles_single_manual_row():
    X = build_features(pd.DataFrame([{"step": 300, "type": "TRANSFER", "amount": 1.5e5}]))
    assert X.shape == (1, len(FEATURES)) and X.loc[0, "type"] == "TRANSFER"
