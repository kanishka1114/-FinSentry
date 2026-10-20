import numpy as np
from sklearn.model_selection import train_test_split
from src.config import TRAIN_FRAC, VAL_FRAC, SEED


def chronological_split(steps, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Split by whole simulation steps: every train step < every val step < every test step.

    Boundaries are chosen so each period holds roughly the requested share of rows
    (PaySim activity is front-loaded, so equal step ranges would be very unbalanced).
    Returns boolean masks (train, val, test) and the step boundaries.
    """
    steps = np.asarray(steps)
    unique, counts = np.unique(steps, return_counts=True)
    if len(unique) < 3:
        raise ValueError("Chronological split needs at least 3 distinct steps.")
    cum = np.cumsum(counts) / counts.sum()
    # Index of the last step in each period; keep at least one step per period.
    train_end = min(closest_index(cum, train_frac), len(unique) - 3)
    val_end = min(max(closest_index(cum, train_frac + val_frac), train_end + 1), len(unique) - 2)

    train_max, val_max = unique[train_end], unique[val_end]
    train = steps <= train_max
    val = (steps > train_max) & (steps <= val_max)
    test = steps > val_max
    bounds = {
        "train": [int(unique[0]), int(train_max)],
        "validation": [int(unique[train_end + 1]), int(val_max)],
        "test": [int(unique[val_end + 1]), int(unique[-1])],
    }
    return train, val, test, bounds


def closest_index(cum, target):
    """Index whose cumulative row share is nearest to target."""
    i = int(np.searchsorted(cum, target, side="left"))
    if i > 0 and (i >= len(cum) or target - cum[i - 1] <= cum[i] - target):
        i -= 1
    return i


def random_split(y, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED):
    """Stratified random split (comparison only; it ignores time order)."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    strat = y if np.bincount(y, minlength=2).min() >= 3 else None
    train_idx, rest = train_test_split(idx, train_size=train_frac, random_state=seed,
                                       stratify=strat)
    rest_strat = y[rest] if strat is not None and np.bincount(y[rest], minlength=2).min() >= 2 else None
    val_share = val_frac / (1 - train_frac)
    val_idx, test_idx = train_test_split(rest, train_size=val_share, random_state=seed,
                                         stratify=rest_strat)
    masks = []
    for chosen in (train_idx, val_idx, test_idx):
        m = np.zeros(len(y), dtype=bool)
        m[chosen] = True
        masks.append(m)
    return (*masks, None)
