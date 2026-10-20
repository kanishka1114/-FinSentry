import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = Path(os.environ.get("FINSENTRY_MODEL_DIR", ROOT / "models"))
OUTPUT_DIR = Path(os.environ.get("FINSENTRY_OUTPUT_DIR", ROOT / "outputs"))
DATA_FILE = DATA_DIR / "PS_20174392719_1491204439457_log.csv"

SEED = 42
# Bump when the structure of outputs/*.json changes; the dashboard and report flag older files.
SCHEMA_VERSION = 4
# Uniform sampling never looks at labels, so it is the default for chronological experiments.
DEFAULT_SAMPLING = "uniform"

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
# Only these columns are read from the CSV; IDs, balances and isFlaggedFraud are never loaded.
USECOLS = ["step", "type", "amount", "isFraud"]
CSV_DTYPES = {"step": "int32", "amount": "float64", "isFraud": "int8"}
READ_CHUNKSIZE = 500_000

# Raw `step` is used only for splitting, never as a model input.
NUMERIC_FEATURES = ["amount", "log_amount", "hour_sin", "hour_cos"]
CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_FIELDS = ["step (split only)", "nameOrig", "nameDest", "oldbalanceOrg",
                   "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
ALERT_BUDGET = 100

# Defaults sized for an 8 GB laptop: shallow-ish trees, per-tree row subsampling, few workers.
RF_DEFAULTS = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 20,
    "max_samples": 0.5,
    "n_jobs": 2,
}
LR_DEFAULTS = {"max_iter": 1000, "C": 1.0}

EVAL_DEFAULTS = {
    "alerts_per_1000": 5.0,   # alert-budget strategy: target alert rate on validation
    "top_k": [100],           # first value is the headline Precision@K
    "top_pct": [0.5, 1.0],    # also report top X% of test rows
    "bootstrap": 300,         # row-bootstrap replicates (0 disables)
    "block_bootstrap": None,  # block-bootstrap replicates; None = same as bootstrap
    "block_steps": None,      # steps per block; None = round(n_steps ** (1/3))
    "min_blocks": 8,          # skip block intervals with fewer independent blocks
    "min_ci_fraud": 10,       # skip intervals when the test period has fewer fraud rows
    "seed": SEED,
}
FINAL_FIT_CHOICES = ("train_only", "refit")
THRESHOLD_TRANSFER_TOLERANCE = 2.0

BACKTEST_DEFAULTS = {
    "windows": 5,
    "scheme": "expanding",    # or "rolling"
    "initial_frac": 0.5,      # share of rows before the first test window
    "val_frac": 0.10,         # share of rows used as validation right before each test window
    "train_window_frac": 0.30,  # rolling scheme only: share of rows in each training window
    "window_by": "rows",      # test windows hold equal row shares ("rows") or step spans ("steps")
    "min_test_fraud": 30,
    "min_val_fraud": 10,
    # Off by default: merging uses test-period label counts to move window boundaries,
    # which would let test labels change later windows' training data.
    "merge_sparse": False,
    "bootstrap": 100,
}

# Above this many rows, training requires an explicit --full flag.
FULL_DATA_GUARD_ROWS = 1_000_000
