import numpy as np
from src.uncertainty import (block_bootstrap_indices, block_bootstrap_ci, stratified_bootstrap_ci,
                             interval_comparison, default_block_length)


def clustered_data(seed=0, n_steps=120, rows_per_step=60):
    """Fraud rate and score level vary by step, so rows within a step are dependent."""
    rng = np.random.default_rng(seed)
    steps = np.repeat(np.arange(1, n_steps + 1), rows_per_step)
    step_rate = rng.choice([0.0, 0.0, 0.01, 0.08], n_steps)
    step_shift = rng.normal(0, 0.15, n_steps)
    y = (rng.random(len(steps)) < step_rate[steps - 1]).astype(int)
    score = np.clip(0.3 + 0.35 * y + step_shift[steps - 1] + rng.normal(0, 0.1, len(steps)), 0, 1)
    return y, score, steps


def test_indices_keep_whole_steps_and_step_count():
    _, _, steps = clustered_data(n_steps=40, rows_per_step=7)
    idx = block_bootstrap_indices(steps, 3, np.random.default_rng(1))
    drawn, counts = np.unique(steps[idx], return_counts=True)
    assert (counts % 7 == 0).all()            # each drawn step contributes all of its rows
    assert counts.sum() // 7 == 40            # same number of steps as the original period


def test_indices_draw_contiguous_blocks():
    steps = np.repeat(np.arange(10, 30), 2)
    idx = block_bootstrap_indices(steps, 4, np.random.default_rng(5))
    seq = steps[idx][::2]                     # one entry per drawn step, in draw order
    for i in range(0, len(seq) - 4, 4):
        block = seq[i:i + 4]
        assert (np.diff(block) == 1).all()


def test_block_ci_reproducible_and_reports_metadata():
    y, s, steps = clustered_data()
    a = block_bootstrap_ci(y, s, steps, {"t": 0.6}, n_boot=150, seed=3)
    b = block_bootstrap_ci(y, s, steps, {"t": 0.6}, n_boot=150, seed=3)
    assert a == b and a["computed"]
    assert a["block_steps"] == default_block_length(120) and a["n_blocks"] >= 8
    assert 0 <= a["replicate_fraud"]["zero_fraud_share"] <= 1
    assert "not guarantees" in a["note"]


def test_block_interval_wider_than_row_interval_under_step_clustering():
    y, s, steps = clustered_data(seed=2)
    row = stratified_bootstrap_ci(y, s, {"t": 0.6}, n_boot=300, seed=1)
    block = block_bootstrap_ci(y, s, steps, {"t": 0.6}, n_boot=300, seed=1)
    comp = {r["metric"]: r for r in interval_comparison(row, block)}
    assert comp["t.recall"]["width_ratio_block_vs_row"] > 1
    assert comp["pr_auc"]["width_ratio_block_vs_row"] > 1


def test_block_ci_skips_with_too_few_blocks_or_fraud():
    y, s, steps = clustered_data(n_steps=12)
    out = block_bootstrap_ci(y, s, steps, {"t": 0.5}, block_steps=3, min_blocks=8)
    assert out["computed"] is False and "blocks" in out["reason"]
    y0 = np.zeros_like(y)
    assert block_bootstrap_ci(y0, s, steps, {"t": 0.5})["computed"] is False
    assert block_bootstrap_ci(y, s, steps, {"t": 0.5}, n_boot=0)["computed"] is False
    empty = block_bootstrap_ci(np.array([], dtype=int), np.array([]), np.array([]), {"t": 0.5})
    assert empty["computed"] is False


def test_replicates_without_fraud_are_counted_not_zeroed():
    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(1, 101), 20)
    y = np.zeros(len(steps), dtype=int)
    y[(steps == 50)] = 1                      # all fraud inside one step
    s = rng.random(len(steps)) * 0.5 + 0.5 * y
    out = block_bootstrap_ci(y, s, steps, {"t": 0.6}, n_boot=200, seed=0, min_fraud=5)
    assert out["replicate_fraud"]["zero_fraud_share"] > 0
    assert out["intervals"]["t.recall"]["n_valid"] < 200
    assert out["intervals"]["pr_auc"]["n_valid"] < 200
