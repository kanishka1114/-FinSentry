"""The recommended, reproducible FinSentry experiment.

One fixed configuration: label-free uniform sampling, chronological split, train-only final
fit, validation-only thresholds, row and block bootstrap, then a walk-forward backtest on the
same sample settings. Both outputs carry the same experiment_id; outputs/experiment.json
records the configuration, data fingerprint, run ids, timings and environment.
"""
import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from src.config import (OUTPUT_DIR, MODEL_DIR, SEED, TRAIN_FRAC, VAL_FRAC, ALERT_BUDGET,
                        RF_DEFAULTS, LR_DEFAULTS, EVAL_DEFAULTS, BACKTEST_DEFAULTS,
                        SCHEMA_VERSION)
from src.train import run_training, library_versions
from src.backtest import run_backtest
from src.report import generate_report

DEFAULT_SAMPLE_ROWS = 1_000_000


def experiment_config(sample_rows=DEFAULT_SAMPLE_ROWS, seed=SEED):
    return {
        "sample_rows": sample_rows, "sampling": "uniform", "seed": seed,
        "split": "chronological", "train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC,
        "final_fit": "train_only", "alert_budget": ALERT_BUDGET,
        "rf_settings": dict(RF_DEFAULTS), "lr_settings": dict(LR_DEFAULTS),
        "evaluation": {k: v for k, v in EVAL_DEFAULTS.items() if k != "seed"},
        "backtest": {**BACKTEST_DEFAULTS, "merge_sparse": False},
    }


def run_experiment(data_path=None, sample_rows=DEFAULT_SAMPLE_ROWS, seed=SEED,
                   allow_full=False, skip_backtest=False, rf_settings=None,
                   eval_overrides=None, backtest_overrides=None,
                   model_dir=MODEL_DIR, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    cfg = experiment_config(sample_rows, seed)
    if rf_settings:
        cfg["rf_settings"].update(rf_settings)
    if eval_overrides:
        cfg["evaluation"].update(eval_overrides)
    if backtest_overrides:
        cfg["backtest"].update(backtest_overrides)
    exp_id = "exp-" + uuid.uuid4().hex[:10]
    record = {
        "experiment_id": exp_id, "schema_version": SCHEMA_VERSION,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "config": cfg, "versions": library_versions(), "status": "running",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_path = output_dir / "experiment.json"
    exp_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if skip_backtest:
        # A backtest from an earlier experiment must not be read as part of this one.
        (output_dir / "backtest.json").unlink(missing_ok=True)

    t0 = time.time()
    train_res = run_training(
        data_path, sample_rows, None, cfg["split"], seed, cfg["train_frac"], cfg["val_frac"],
        cfg["alert_budget"], cfg["rf_settings"], cfg["lr_settings"], allow_full,
        model_dir, output_dir, eval_settings=cfg["evaluation"], final_fit=cfg["final_fit"],
        sampling_method=cfg["sampling"], write_report=False, experiment_id=exp_id)
    record["train"] = {"run_id": train_res["run"]["run_id"], "seconds": round(time.time() - t0, 2)}
    record["dataset"] = train_res["dataset"]
    record["split_boundaries"] = train_res["split"]["step_boundaries"]

    if not skip_backtest:
        t1 = time.time()
        bt = dict(cfg["backtest"])
        bt_res = run_backtest(
            data_path, sample_rows, None, seed, bt, cfg["rf_settings"], cfg["lr_settings"],
            eval_settings={k: v for k, v in cfg["evaluation"].items() if k != "bootstrap"},
            allow_full=allow_full, output_dir=output_dir, sampling_method=cfg["sampling"],
            write_report=False, experiment_id=exp_id)
        record["backtest"] = {"run_id": bt_res["run"]["run_id"],
                              "seconds": round(time.time() - t1, 2),
                              "windows": [{k: w[k] for k in ("window", "train_steps",
                                                             "validation_steps", "test_steps")}
                                          for w in bt_res["windows"]]}
    record.update({"status": "complete", "seconds": round(time.time() - t0, 2),
                   "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    exp_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    generate_report(output_dir)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the recommended reproducible FinSentry experiment.")
    p.add_argument("--data", default=None)
    p.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS,
                   help=f"Uniform sample size (default {DEFAULT_SAMPLE_ROWS:,})")
    p.add_argument("--full", action="store_true",
                   help="Use every row instead of a sample (not run by default; needs time and RAM)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-backtest", action="store_true")
    args = p.parse_args(argv)
    rec = run_experiment(args.data, None if args.full else args.sample_rows, args.seed,
                         allow_full=args.full, skip_backtest=args.skip_backtest)
    print(json.dumps({k: rec.get(k) for k in ("experiment_id", "status", "seconds", "train",
                                              "split_boundaries")} |
                     {"backtest_run_id": rec.get("backtest", {}).get("run_id")}, indent=2))


if __name__ == "__main__":
    main()
