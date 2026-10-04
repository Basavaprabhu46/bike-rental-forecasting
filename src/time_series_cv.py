"""Time-series CV (Sec 6, optional extension). Expanding-window, no shuffling.

Usage:
    python3 src/time_series_cv.py --data dataset/bike-sharing-dataset/hour.csv --splits 5
Outputs: reports/results/<gran>_tscv.csv + <gran>_tscv.png
Uses lighter estimators (100 trees) for speed; final models still trained in train.py.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from preprocessing import build_tabular_preprocessor, preprocess_data
from splits import tscv_splits


def rmsle(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/bike-sharing-dataset/hour.csv")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    gran = "hour" if "hr" in df.columns else "day"
    X, y = preprocess_data(df)
    print(f"[TSCV] {gran}: {len(X)} rows, {args.splits} expanding folds")

    def pipes():
        return {
            "baseline_mean": Pipeline([("prep", build_tabular_preprocessor(X, False)), ("m", DummyRegressor(strategy="mean"))]),
            "linear": Pipeline([("prep", build_tabular_preprocessor(X, True)), ("m", LinearRegression())]),
            "random_forest": Pipeline([("prep", build_tabular_preprocessor(X, False)), ("m", RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42))]),
            "gradient_boosting": Pipeline([("prep", build_tabular_preprocessor(X, False)), ("m", GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.9, random_state=42))]),
        }
    rows = []
    for fold, (tr, te) in enumerate(tscv_splits(len(X), args.splits)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        for name, pipe in pipes().items():
            pipe.fit(Xtr, ytr)
            r = rmsle(yte, pipe.predict(Xte))
            rows.append({"model": name, "fold": fold, "RMSLE": r, "n_train": len(tr), "n_test": len(te)})
            print(f"[FOLD {fold}] {name}: RMSLE={r:.4f} (train {len(tr)} test {len(te)})")
    res = pd.DataFrame(rows)
    os.makedirs("reports/results", exist_ok=True)
    res.to_csv(f"reports/results/{gran}_tscv.csv", index=False)
    piv = res.pivot_table(index="fold", columns="model", values="RMSLE")
    piv.plot(kind="bar", figsize=(9, 4), title=f"Expanding-window TSCV RMSLE ({gran})")
    plt.ylabel("RMSLE")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{gran}_tscv.png", dpi=150)
    plt.close()
    print(f"[SAVED] {gran}_tscv.csv + png")
    print(res.groupby("model").RMSLE.agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
