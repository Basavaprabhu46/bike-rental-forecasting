"""Deterministic preprocessing for bike-sharing dataset (day.csv / hour.csv).

Design: Option A from proposal.
- preprocess_data() does deterministic cleaning + feature engineering only.
- No learned statistics here (no scaler/imputer/encoder fitting).
- Learned transforms are fitted in train.py inside sklearn
  Pipelines / ColumnTransformers (leakage-safe, fit on train only).

Expected columns (UCI bike-sharing):
  instant, dteday, season, yr, mnth, [hr], holiday, weekday,
  workingday, weathersit, temp, atemp, hum, windspeed,
  casual, registered, cnt (target)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "cnt"
LEAKAGE_COLS = {"casual", "registered", "cnt"}
ID_COLS = {"instant", "dteday"}
CATEGORICAL_FEATURES = ["season", "weathersit"]


def _validate(df: pd.DataFrame) -> None:
    required = {"dteday", "season", "yr", "mnth", "holiday", "weekday",
                "workingday", "weathersit", "temp", "atemp", "hum",
                "windspeed", "cnt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if (df["cnt"] < 0).any():
        raise ValueError("Target cnt contains negative values")
    if {"casual", "registered"}.issubset(df.columns):
        bad = (df["cnt"] != df["casual"] + df["registered"]).sum()
        if bad:
            raise ValueError(f"cnt != casual+registered in {bad} rows")


def add_history_features(df: pd.DataFrame, lags=(1, 24)) -> pd.DataFrame:
    """OPTIONAL lag features with strict shifting (no leakage).

    Uses only past values: shift(lag) so row t never sees target at t.
    Requires chronological sort by dteday (+hr if present).
    Default OFF in preprocess_data(); enable only for true forecasting
    with chronological split. Drops rows with NaN lags.
    """
    df = df.copy()
    sort_keys = ["dteday"] + (["hr"] if "hr" in df.columns else [])
    df = df.sort_values(sort_keys).reset_index(drop=True)
    for lag in lags:
        df[f"lag_{lag}"] = df["cnt"].shift(lag)
    return df


def preprocess_data(df: pd.DataFrame, add_history: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Process raw dataset into model-ready X, y.

    Deterministic only. season/weathersit stay as integers here;
    one-hot encoding happens in build_tabular_preprocessor() (fit on train).
    """
    df = df.copy()
    _validate(df)
    if add_history:
        df = add_history_features(df).dropna().reset_index(drop=True)

    df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")
    if df["dteday"].isna().any():
        raise ValueError("Invalid timestamps in dteday")
    has_hour = "hr" in df.columns

    df["weekend"] = (df["weekday"] >= 5).astype(int)
    if has_hour:
        df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
        df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
        df["mnth_sin"] = np.sin(2 * np.pi * (df["mnth"] - 1) / 12)
        df["mnth_cos"] = np.cos(2 * np.pi * (df["mnth"] - 1) / 12)
        is_wd = df["workingday"] == 1
        df["peak_weekday"] = (is_wd & df["hr"].isin([7, 8, 9, 17, 18, 19])).astype(int)
        df["peak_weekend"] = ((~is_wd) & df["hr"].between(10, 18)).astype(int)
    else:
        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
        df["mnth_sin"] = np.sin(2 * np.pi * (df["mnth"] - 1) / 12)
        df["mnth_cos"] = np.cos(2 * np.pi * (df["mnth"] - 1) / 12)

    # temp vs atemp r ~ 0.99 -> keep atemp only (feels-like, per ref paper)
    df = df.drop(columns=["temp"])

    drop = [c for c in (ID_COLS | LEAKAGE_COLS) if c in df.columns]
    y = df["cnt"].copy()
    X = df.drop(columns=drop)
    assert len(X) == len(y)
    return X, y


def build_tabular_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Leakage-safe learned transforms. MUST be fit on train only.

    - numeric: SimpleImputer(median, fit train) [+ StandardScaler if scale_numeric]
    - categoricals (season, weathersit): SimpleImputer(most_frequent) + OneHotEncoder(ignore unknown)
    Trees: scale_numeric=False. Linear: True.
    """
    cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    num_pipe = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_pipe.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_pipe), num)]
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    return ColumnTransformer(transformers)
