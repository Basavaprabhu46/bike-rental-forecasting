"""Evaluate saved models + generate plots.

Usage:
    python3 src/evaluate.py --data dataset/bike-sharing-dataset/hour.csv
Outputs:
    reports/figures/<gran>_*.png
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess_data


def rmsle(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/bike-sharing-dataset/hour.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    gran = "hour" if "hr" in df.columns else "day"
    cut = int(len(df) * (1 - args.test_size))
    df_test = df.iloc[cut:].copy()
    X_test, y_test = preprocess_data(df_test)

    os.makedirs("reports/figures", exist_ok=True)
    models = ["baseline_mean", "linear", "random_forest", "gradient_boosting"]
    preds = {}
    for m in models:
        p = f"models/{gran}_{m}.pkl"
        if not os.path.exists(p):
            print(f"[SKIP] {p} not found, run train.py first")
            continue
        with open(p, "rb") as f:
            obj = pickle.load(f)
        preds[m] = np.maximum(obj["pipeline"].predict(X_test), 0)

    if preds:
        rms = [rmsle(y_test, preds[m]) for m in preds]
        plt.figure(figsize=(7, 4))
        plt.bar(list(preds.keys()), rms)
        plt.ylabel("RMSLE (lower=better)")
        plt.title(f"Model comparison ({gran})")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"reports/figures/{gran}_model_comparison.png", dpi=150)
        plt.close()
        print(f"[SAVED] {gran}_model_comparison.png")

    if preds:
        best = min(preds, key=lambda m: rmsle(y_test, preds[m]))
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, preds[best], s=5, alpha=0.3)
        mx = float(max(y_test.max(), preds[best].max()))
        plt.plot([0, mx], [0, mx], "r--")
        plt.xlabel("Actual cnt")
        plt.ylabel("Predicted cnt")
        plt.title(f"Predicted vs actual: {best} ({gran})")
        plt.tight_layout()
        plt.savefig(f"reports/figures/{gran}_pred_vs_actual.png", dpi=150)
        plt.close()
        res = y_test.values - preds[best]
        plt.figure(figsize=(7, 4))
        plt.hist(res, bins=50)
        plt.xlabel("Residual (actual-pred)")
        plt.title(f"Residuals: {best} ({gran})")
        plt.tight_layout()
        plt.savefig(f"reports/figures/{gran}_residuals.png", dpi=150)
        plt.close()
        print(f"[SAVED] {gran}_pred_vs_actual.png + residuals, best={best}")


if __name__ == "__main__":
    main()
