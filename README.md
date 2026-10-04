# Bike Rental Demand Forecasting

Predict hourly / daily bike rentals from Capital Bikeshare (Washington D.C. 2011-2012) using time + weather features.

## Dataset
Source: `dataset/bike-sharing-dataset/` (UCI, Fanaee-T & Gama 2013)
- `hour.csv`: 17379 rows, 17 cols (has `hr`)
- `day.csv`: 731 rows, 16 cols
- Target: `cnt` (= casual + registered, verified). No missing, no duplicates.
- Docs: see `dataset/bike-sharing-dataset/Readme.txt`

## Setup
Python 3.10+, install:
```
pip install -r requirements.txt
```

## Run (chronological 70/10/20 + TSCV, leakage-safe)
```
python3 notebooks/01_eda.py
python3 src/train.py --data dataset/bike-sharing-dataset/hour.csv  # + day.csv
python3 src/time_series_cv.py --data dataset/bike-sharing-dataset/hour.csv
python3 src/ablation.py --data dataset/bike-sharing-dataset/hour.csv
python3 src/evaluate.py --data dataset/bike-sharing-dataset/hour.csv
python3 src/error_analysis.py --data dataset/bike-sharing-dataset/hour.csv
python3 src/train.py --data dataset/bike-sharing-dataset/hour.csv
python3 src/train.py --data dataset/bike-sharing-dataset/day.csv
python3 src/evaluate.py --data dataset/bike-sharing-dataset/hour.csv
python3 src/evaluate.py --data dataset/bike-sharing-dataset/day.csv
python3 src/predict.py --gran hour --hr 8 --workingday 1
```

## Results (verified Oct 2026)
Hour test (3476 rows): GB RMSE 72.7 / RF RMSLE 0.41 / Linear RMSE 122.6 / Baseline 232.6
Day test (147 rows): GB RMSE 1015 / RMSLE 0.50 / Linear 1206 / Baseline 2560
Key insights: hr most important (peaks 8h, 17-18h workingday 498 vs 149 off-peak);
fall > spring; clear weather > bad; temp +ve, hum -ve; temp/atemp r=0.99 (keep atemp).

## Structure
```
src/preprocessing.py  # preprocess_data(df)->X,y, deterministic only
src/train.py          # chronological split, Pipelines, saves models/*.pkl
src/evaluate.py       # metrics + reports/figures/*.png
src/predict.py        # single prediction via saved pipeline
notebooks/01_eda.py   # EDA script
reports/figures/      # EDA + evaluation plots
reports/results/      # *_metrics.csv
models/               # saved pipelines (gitignored if large)
```

## Limitations
Chronological not random split; ws_4 has only 3 hour rows; hum==0 / windspeed==0 kept as-is;
history/rolling features omitted to avoid leakage; tuning minimal for 1-week scope.
