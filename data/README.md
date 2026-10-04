# Data location

Actual CSVs (UCI Bike Sharing, Fanaee-T & Gama 2013):
- `dataset/bike-sharing-dataset/hour.csv` — 17379 rows, hourly (has `hr`)
- `dataset/bike-sharing-dataset/day.csv` — 731 rows, daily
- `dataset/bike-sharing-dataset/Readme.txt` — column docs, license, citation

Target: `cnt` (= `casual` + `registered`). Do NOT use `casual`/`registered`/`cnt` as inputs.

Place new downloads here if re-fetched from UCI. Do NOT commit large CSVs if license/size requires — keep locally and document source. Code loads via `pd.read_csv("dataset/bike-sharing-dataset/hour.csv")`.
Example rows are illustrative only; see `notebooks/01_eda.py` for real inspection.
