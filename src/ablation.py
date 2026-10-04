"""Sec 8: ablation (with/without feature groups) + importance.

Groups (from preprocess_data output):
- peak: peak_weekday, peak_weekend
- cyclical: hr_sin/cos, weekday_sin/cos, mnth_sin/cos
- weather_env: atemp, hum, windspeed, season, weathersit
- calendar_low: holiday (tests ref-paper claim that holiday is noise)
Ablation model: GradientBoosting(n=100) on 70/10/20 chronological split.
Importance: RF built-in + permutation (sampled test) for best config.

Usage: python3 src/ablation.py --data dataset/bike-sharing-dataset/hour.csv
Outputs: reports/results/<gran>_ablation.csv/.png, <gran>_importance.csv/.png
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from preprocessing import build_tabular_preprocessor, preprocess_data
from splits import split_df

GROUPS = {
    "peak": ["peak_weekday", "peak_weekend"],
    "cyclical": ["hr_sin", "hr_cos", "weekday_sin", "weekday_cos", "mnth_sin", "mnth_cos"],
    "weather_env": ["atemp", "hum", "windspeed", "season", "weathersit"],
    "calendar_low": ["holiday"],
}


def rmsle(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0)))))


def fit_gb(X_sample):
    return Pipeline([
        ("prep", build_tabular_preprocessor(X_sample, scale_numeric=False)),
        ("m", GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                        learning_rate=0.08, subsample=0.9, random_state=42))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/bike-sharing-dataset/hour.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    gran = "hour" if "hr" in df.columns else "day"
    dtr, dva, dte = split_df(df, 0.7, 0.1)
    Xtr, ytr = preprocess_data(dtr)
    Xva, yva = preprocess_data(dva)
    Xte, yte = preprocess_data(dte)

    configs = {"full": []}
    for g, cols in GROUPS.items():
        configs[f"no_{g}"] = [c for c in cols if c in Xtr.columns]
    rows = []
    for cfg, drop in configs.items():
        keep = [c for c in Xtr.columns if c not in drop]
        pipe = fit_gb(Xtr[keep])
        pipe.fit(Xtr[keep], ytr)
        rv, rt = rmsle(yva, pipe.predict(Xva[keep])), rmsle(yte, pipe.predict(Xte[keep]))
        rows.append({"config": cfg, "dropped": len(drop), "val_RMSLE": rv, "test_RMSLE": rt})
        print(f"[ABLATION] {cfg} (drop {len(drop)}): val RMSLE={rv:.4f} test={rt:.4f}")
    res = pd.DataFrame(rows)
    os.makedirs("reports/results", exist_ok=True)
    res.to_csv(f"reports/results/{gran}_ablation.csv", index=False)
    plt.figure(figsize=(8, 4))
    plt.bar(res.config, res.test_RMSLE)
    plt.ylabel("Test RMSLE (lower=better)")
    plt.title(f"Ablation: with/without groups ({gran}, GB-100)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"reports/figures/{gran}_ablation.png", dpi=150)
    plt.close()

    # Importance on full config with RF (built-in + permutation on sampled test)
    rf = Pipeline([
        ("prep", build_tabular_preprocessor(Xtr, scale_numeric=False)),
        ("m", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42))])
    rf.fit(Xtr, ytr)
    prep = rf.named_steps["prep"]
    try:
        feat_names = prep.get_feature_names_out()
    except Exception:
        feat_names = np.array(Xtr.columns)
    imp = rf.named_steps["m"].feature_importances_
    samp = np.random.RandomState(0).choice(len(Xte), size=min(2000, len(Xte)), replace=False)
    perm = permutation_importance(rf, Xte.iloc[samp], yte.iloc[samp], n_repeats=5, random_state=0, scoring="neg_mean_squared_error")
    rf_df = pd.DataFrame({"feature": list(feat_names), "rf_importance": imp}).sort_values("rf_importance", ascending=False)
    perm_df = pd.DataFrame({"feature": list(Xtr.columns), "perm_mean": perm.importances_mean,
                            "perm_std": perm.importances_std}).sort_values("perm_mean", ascending=False)
    imp_df = perm_df  # primary ranking by permutation (original columns)
    imp_df.to_csv(f"reports/results/{gran}_importance.csv", index=False)
    rf_df.to_csv(f"reports/results/{gran}_importance_rf.csv", index=False)
    top = imp_df.head(12).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top.feature.astype(str), top.perm_mean, xerr=top.perm_std)
    plt.xlabel("Permutation importance (MSE drop, sampled test n<=2000)")
    plt.title(f"Top-12 permutation importance ({gran}, RF)")
    plt.tight_layout()
    plt.savefig(f"reports/figures/{gran}_importance.png", dpi=150)
    plt.close()
    print(f"[SAVED] {gran}_ablation.csv/png + importance.csv/png")
    print("[TOP5 importance]\n", imp_df.head(5).to_string(index=False))
    print("[NOTE] Importance = predictive contribution, not causation. Low linear corr features kept (e.g. holiday) can still matter non-linearly.")


if __name__ == "__main__":
    main()
