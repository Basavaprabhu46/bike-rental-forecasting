"""Chronological splits + time-series CV helpers (Sec 6)."""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def chronological_train_val_test(n, train_frac=0.7, val_frac=0.1):
    """Indices for train/val/test in time order. Test = remainder.

    Default 70/10/20 keeps last 20% as test (same as old 80/20 test window)
    so results stay comparable; val = 10% before test for model selection.
    """
    assert 0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return range(0, train_end), range(train_end, val_end), range(val_end, n)


def split_df(df, train_frac=0.7, val_frac=0.1):
    tr, va, te = chronological_train_val_test(len(df), train_frac, val_frac)
    return df.iloc[list(tr)].copy(), df.iloc[list(va)].copy(), df.iloc[list(te)].copy()


def tscv_splits(n, n_splits=5):
    return list(TimeSeriesSplit(n_splits=n_splits).split(range(n)))
