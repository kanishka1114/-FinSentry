import numpy as np
import pytest
from src.splits import chronological_split, random_split


def test_chronological_split_never_mixes_time():
    rng = np.random.default_rng(0)
    steps = np.sort(rng.integers(1, 744, 50_000))
    train, val, test, bounds = chronological_split(steps, 0.7, 0.15)
    assert (train | val | test).all()
    assert not (train & val).any() and not (val & test).any() and not (train & test).any()
    assert steps[train].max() < steps[val].min()
    assert steps[val].max() < steps[test].min()
    assert bounds["train"][1] < bounds["validation"][0] <= bounds["validation"][1] < bounds["test"][0]
    assert abs(train.mean() - 0.7) < 0.02 and abs(val.mean() - 0.15) < 0.02


def test_chronological_split_follows_row_share_for_front_loaded_data():
    steps = np.concatenate([np.repeat(np.arange(1, 11), 1_000), np.arange(11, 101)])
    train, val, test, _ = chronological_split(steps, 0.7, 0.15)
    assert steps[train].max() <= 7
    assert val.sum() > 0 and test.sum() > 0


def test_chronological_split_works_on_unsorted_input():
    steps = np.array([5, 1, 3, 2, 4, 6, 9, 8, 7, 10])
    train, val, test, _ = chronological_split(steps, 0.6, 0.2)
    assert steps[train].max() < steps[val].min() and steps[val].max() < steps[test].min()


def test_chronological_split_needs_three_steps():
    with pytest.raises(ValueError):
        chronological_split(np.array([1, 1, 2, 2]))


def test_random_split_is_reproducible_and_disjoint():
    y = np.array([0] * 950 + [1] * 50)
    a = random_split(y, 0.7, 0.15, seed=3)
    b = random_split(y, 0.7, 0.15, seed=3)
    for m1, m2 in zip(a[:3], b[:3]):
        assert (m1 == m2).all()
    train, val, test, bounds = a
    assert bounds is None
    assert (train.astype(int) + val + test == 1).all()
    assert y[test].sum() > 0
