"""Load best saved pipeline and predict.

Example:
    python3 src/predict.py --gran hour --hr 8 --weekday 1 --workingday 1 \
        --weathersit 1 --atemp 0.5 --hum 0.6 --windspeed 0.2 --yr 1 --mnth 6 --season 2
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gran", default="hour", choices=["hour", "day"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--hr", type=int, default=8)
    ap.add_argument("--weekday", type=int, default=1)
    ap.add_argument("--workingday", type=int, default=1)
    ap.add_argument("--weathersit", type=int, default=1)
    ap.add_argument("--season", type=int, default=2)
    ap.add_argument("--yr", type=int, default=1)
    ap.add_argument("--mnth", type=int, default=6)
    ap.add_argument("--holiday", type=int, default=0)
    ap.add_argument("--atemp", type=float, default=0.5)
    ap.add_argument("--hum", type=float, default=0.6)
    ap.add_argument("--windspeed", type=float, default=0.2)
    args = ap.parse_args()

    name = args.model
    if name is None:
        for cand in ["gradient_boosting", "random_forest", "linear"]:
            if os.path.exists(f"models/{args.gran}_{cand}.pkl"):
                name = cand
                break
    path = f"models/{args.gran}_{name}.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. Run train.py first.")
    with open(path, "rb") as f:
        obj = pickle.load(f)

    from preprocessing import preprocess_data
    row = {"instant": 0, "dteday": "2012-06-01", "season": args.season,
           "yr": args.yr, "mnth": args.mnth, "holiday": args.holiday,
           "weekday": args.weekday, "workingday": args.workingday,
           "weathersit": args.weathersit, "temp": 0.5, "atemp": args.atemp,
           "hum": args.hum, "windspeed": args.windspeed,
           "casual": 0, "registered": 0, "cnt": 0}
    if args.gran == "hour":
        row["hr"] = args.hr
    X, _ = preprocess_data(pd.DataFrame([row]))
    pred = float(np.maximum(obj["pipeline"].predict(X), 0)[0])
    print(f"[PREDICT] {args.gran}/{name}: {pred:.1f} bikes")


if __name__ == "__main__":
    main()
