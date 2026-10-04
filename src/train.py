"""Train with 3-way chronological split (Sec 6) + val-based selection.

Split: 70% train / 10% val / 20% test (test = same last-20% window as before).
- Fit on train, select on val, report final on test.
- All imputers/scalers/encoders are ColumnTransformers fit on train only.

Usage:
    python3 src/train.py --data dataset/bike-sharing-dataset/hour.csv
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error)
from sklearn.pipeline import Pipeline
from preprocessing import build_tabular_preprocessor, preprocess_data
from splits import split_df


def rmsle(y_true, y_pred):
    y_pred = np.maximum(np.asarray(y_pred), 0)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_models(X_train: pd.DataFrame):
    return {
        "baseline_mean": Pipeline([
            ("prep", build_tabular_preprocessor(X_train, scale_numeric=False)),
            ("m", DummyRegressor(strategy="mean"))]),
        "linear": Pipeline([
            ("prep", build_tabular_preprocessor(X_train, scale_numeric=True)),
            ("m", LinearRegression())]),
        "random_forest": Pipeline([
            ("prep", build_tabular_preprocessor(X_train, scale_numeric=False)),
            ("m", RandomForestRegressor(
                n_estimators=200, max_depth=None, min_samples_leaf=2,
                n_jobs=-1, random_state=42))]),
        "gradient_boosting": Pipeline([
            ("prep", build_tabular_preprocessor(X_train, scale_numeric=False)),
            ("m", GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.08,
                subsample=0.9, random_state=42))]),
    }


def scores(y, pred):
    pred = np.maximum(pred, 0)
    return (float(mean_absolute_error(y, pred)), rmse(y, pred), rmsle(y, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/bike-sharing-dataset/hour.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    gran = "hour" if "hr" in df.columns else "day"
    df_train, df_val, df_test = split_df(df, 0.7, 0.1)
    print(f"[SPLIT] {gran}: train {len(df_train)} / val {len(df_val)} / test {len(df_test)} (chronological 70/10/20)")

    X_train, y_train = preprocess_data(df_train)
    X_val, y_val = preprocess_data(df_val)
    X_test, y_test = preprocess_data(df_test)
    print(f"[FEATURES] {X_train.shape[1]} cols")

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/results", exist_ok=True)
    val_rows, test_rows = [], []
    for name, pipe in build_models(X_train).items():
        pipe.fit(X_train, y_train)
        mae_v, rmse_v, rmsle_v = scores(y_val, pipe.predict(X_val))
        mae_t, rmse_t, rmsle_t = scores(y_test, pipe.predict(X_test))
        val_rows.append({"model": name, "MAE": mae_v, "RMSE": rmse_v, "RMSLE": rmsle_v})
        test_rows.append({"model": name, "MAE": mae_t, "RMSE": rmse_t, "RMSLE": rmsle_t})
        print(f"[VAL ] {name}: RMSE={rmse_v:.2f} RMSLE={rmsle_v:.4f}")
        print(f"[TEST] {name}: MAE={mae_t:.2f} RMSE={rmse_t:.2f} RMSLE={rmsle_t:.4f}")
        with open(f"models/{gran}_{name}.pkl", "wb") as f:
            pickle.dump({"pipeline": pipe, "granularity": gran}, f)
    pd.DataFrame(val_rows).sort_values("RMSE").to_csv(
        f"reports/results/{gran}_val_metrics.csv", index=False)
    res = pd.DataFrame(test_rows).sort_values("RMSE")
    res.to_csv(f"reports/results/{gran}_metrics.csv", index=False)
    print(f"[SAVED] reports/results/{gran}_val_metrics.csv + {gran}_metrics.csv")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
