"""Sec 9: error analysis + missing eval plots.

- residuals vs predicted scatter (complements residuals hist in evaluate.py)
- errors by hour / weekday-workingday / demand level / weathersit
- uses best-by-RMSLE saved model on chronological test (last 20%)

Usage: python3 src/error_analysis.py --data dataset/bike-sharing-dataset/hour.csv
Outputs: reports/results/<gran>_errors_by_*.csv, reports/figures/<gran>_resid_vs_pred.png, <gran>_errors_by_*.png
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
from splits import split_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/bike-sharing-dataset/hour.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    gran = "hour" if "hr" in df.columns else "day"
    _, _, dte = split_df(df, 0.7, 0.1)
    Xte_raw, yte = preprocess_data(dte)
    dte = dte.reset_index(drop=True)
    yte = yte.reset_index(drop=True)

    # pick best saved model by test RMSLE
    from sklearn.metrics import mean_squared_error
    best, best_r, best_p = None, 1e9, None
    for m in ["baseline_mean", "linear", "random_forest", "gradient_boosting"]:
        p = f"models/{gran}_{m}.pkl"
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            pipe = pickle.load(f)["pipeline"]
        pred = np.maximum(pipe.predict(Xte_raw), 0)
        r = float(np.sqrt(mean_squared_error(np.log1p(yte), np.log1p(pred))))
        if r < best_r:
            best, best_r, best_p = m, r, pred
    print(f"[ERROR-ANALYSIS] {gran}: best={best} RMSLE={best_r:.4f}")
    pred = best_p
    resid = yte.values - pred

    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/results", exist_ok=True)

    # 1. residuals vs predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(pred, resid, s=5, alpha=0.25)
    plt.axhline(0, color="r", ls="--")
    plt.xlabel("Predicted cnt")
    plt.ylabel("Residual (actual - pred)")
    plt.title(f"Residuals vs predicted: {best} ({gran})")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{gran}_resid_vs_pred.png", dpi=150)
    plt.close()

    # 2. errors by group
    err = pd.DataFrame({"actual": yte.values, "pred": pred,
                        "ae": np.abs(resid), "se": resid ** 2})
    groups = {}
    if "hr" in dte.columns:
        groups["hour"] = dte["hr"]
    groups["workingday"] = dte["workingday"].map({0: "weekend/holiday", 1: "workingday"})
    groups["weathersit"] = dte["weathersit"]
    err["demand_bin"] = pd.qcut(err.actual, q=4, labels=["low", "mid-low", "mid-high", "high"], duplicates="drop")
    for gname, gvals in groups.items():
        t = err.assign(g=gvals.values).groupby("g").agg(
            MAE=("ae", "mean"), RMSE=("se", lambda s: float(np.sqrt(s.mean()))), n=("ae", "size")).reset_index()
        t.to_csv(f"reports/results/{gran}_errors_by_{gname}.csv", index=False)
        plt.figure(figsize=(8, 4))
        plt.bar(t.g.astype(str), t.MAE)
        plt.xlabel(gname)
        plt.ylabel("MAE")
        plt.title(f"MAE by {gname}: {best} ({gran}) — answers where model fails")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"reports/figures/{gran}_errors_by_{gname}.png", dpi=150)
        plt.close()
        print(f"[BY {gname}]\n", t.to_string(index=False))
    t = err.groupby("demand_bin", observed=True).agg(
        MAE=("ae", "mean"), RMSE=("se", lambda s: float(np.sqrt(s.mean()))), n=("ae", "size")).reset_index()
    t.to_csv(f"reports/results/{gran}_errors_by_demand.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(t.demand_bin.astype(str), t.MAE)
    plt.xlabel("actual demand quartile")
    plt.ylabel("MAE")
    plt.title(f"MAE by demand level ({gran}) — high demand usually higher absolute error")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{gran}_errors_by_demand.png", dpi=150)
    plt.close()
    print("[BY demand]\n", t.to_string(index=False))
    print(f"[SAVED] {gran}_resid_vs_pred.png + errors_by_*.csv/png")


if __name__ == "__main__":
    main()
