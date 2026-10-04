"""EDA for both day.csv and hour.csv. Generates figures + prints stats.

Run:
    python3 notebooks/01_eda.py
Outputs:
    reports/figures/eda_*.png
Covers proposal Sec 2.1 + 3.1 + 3.2 (all essential + optional plots).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DAY = "dataset/bike-sharing-dataset/day.csv"
HOUR = "dataset/bike-sharing-dataset/hour.csv"


def main():
    os.makedirs("reports/figures", exist_ok=True)
    day = pd.read_csv(DAY, parse_dates=["dteday"])
    hour = pd.read_csv(HOUR, parse_dates=["dteday"])
    print(f"[SHAPE] day {day.shape}, hour {hour.shape}")
    print(f"[COLS] day {list(day.columns)}")
    print(f"[COLS] hour {list(hour.columns)}")
    print(f"[DTYPES]\n{day.dtypes.to_string()}")
    print(f"[MISSING] day total {day.isnull().sum().sum()} {day.isnull().sum().to_dict()}")
    print(f"[MISSING] hour total {hour.isnull().sum().sum()}")
    print(f"[DUPLICATES] day {day.duplicated().sum()}, hour {hour.duplicated().sum()}")

    # Sec 2.1: categorical uniques + min/max dates + granularity + leakage
    for name, df in [("day", day), ("hour", hour)]:
        print(f"[{name} uniques] season {sorted(df.season.unique())} "
              f"weathersit {sorted(df.weathersit.unique())} "
              f"weekday {sorted(df.weekday.unique())} mnth {sorted(df.mnth.unique())} "
              f"holiday {sorted(df.holiday.unique())} workingday {sorted(df.workingday.unique())} "
              + (f"hr {df.hr.min()}-{df.hr.max()}" if "hr" in df.columns else "no-hr(daily)"))
    print(f"[DATE RANGE] day {day.dteday.min()} to {day.dteday.max()}, "
          f"hour {hour.dteday.min()} to {hour.dteday.max()}")
    print(f"[LEAKAGE] casual/registered/cnt must be excluded from X; "
          f"instant/dteday are IDs. cnt==casual+registered: "
          f"day {(day.cnt == day.casual + day.registered).all()}, "
          f"hour {(hour.cnt == hour.casual + hour.registered).all()}")
    print("[DAY describe]\n", day.describe().round(2).to_string())
    print("[HOUR describe]\n", hour.describe().round(2).to_string())
    print(f"[SKEW cnt] day {day.cnt.skew():.3f}, hour {hour.cnt.skew():.3f}")
    print(f"[CORR temp-atemp] day {day.temp.corr(day.atemp):.4f}, hour {hour.temp.corr(hour.atemp):.4f}")
    print("[CORR with cnt hour]\n", hour.select_dtypes(include=np.number).corr()["cnt"].sort_values(ascending=False).round(3).to_string())
    print(f"[OUTLIERS] hum==0 day {(day.hum == 0).sum()} hour {(hour.hum == 0).sum()}, "
          f"windspeed==0 hour {(hour.windspeed == 0).sum()}, ws_4 day {(day.weathersit == 4).sum()} hour {(hour.weathersit == 4).sum()}")

    # 1. demand distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(day.cnt, bins=30)
    plt.title("Daily cnt (symmetric)")
    plt.subplot(1, 2, 2)
    plt.hist(hour.cnt, bins=50)
    plt.title("Hourly cnt (right-skew 1.27)")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_demand_dist.png", dpi=150)
    plt.close()

    # 2. demand over time + monthly trend
    plt.figure(figsize=(10, 4))
    plt.plot(day.dteday, day.cnt, lw=0.8)
    plt.title("Daily rentals 2011-2012 (growth + seasonality)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_demand_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    day.assign(ym=day.dteday.dt.to_period("M").astype(str)).groupby("ym").cnt.mean().plot(kind="line", marker="o", ms=3)
    plt.title("Monthly mean daily rentals (seasonality + YoY growth)")
    plt.xticks(rotation=60, fontsize=7)
    plt.tight_layout()
    plt.savefig("reports/figures/eda_monthly_trend.png", dpi=150)
    plt.close()

    # 3. demand by hour
    plt.figure(figsize=(9, 4))
    hour.groupby("hr").cnt.mean().plot(kind="bar")
    plt.title("Mean rentals by hour (peaks 8h, 17-18h)")
    plt.xlabel("Hour")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_by_hour.png", dpi=150)
    plt.close()

    # 3b. demand by weekday + hour x weekday
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    hour.groupby("weekday").cnt.mean().plot(kind="bar", ax=ax[0], title="Mean by weekday (0=Sun..6=Sat)")
    piv = hour.pivot_table(index="hr", columns="workingday", values="cnt", aggfunc="mean")
    piv.plot(kind="line", ax=ax[1], title="Hour profile: workingday=1 vs 0")
    ax[1].set_xlabel("Hour")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_weekday_hourxweekday.png", dpi=150)
    plt.close()

    # 4. season + weather + boxplots (outliers)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    day.groupby("season").cnt.mean().plot(kind="bar", ax=ax[0], title="Daily mean by season (1-4)")
    hour.groupby("weathersit").cnt.mean().plot(kind="bar", ax=ax[1], title="Hourly mean by weather (1-3,4 rare n=3)")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_season_weather.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    day.boxplot(column="cnt", by="season", ax=ax[0])
    ax[0].set_title("Daily cnt boxplot by season")
    hour[hour.weathersit.isin([1, 2, 3])].boxplot(column="cnt", by="weathersit", ax=ax[1])
    ax[1].set_title("Hourly cnt boxplot by weather")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_boxplots.png", dpi=150)
    plt.close()

    # 5. weather scatter
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].scatter(hour.temp, hour.cnt, s=2, alpha=0.2)
    ax[0].set_title("temp vs cnt (+ve)")
    ax[1].scatter(hour.hum, hour.cnt, s=2, alpha=0.2)
    ax[1].set_title("hum vs cnt (-ve)")
    ax[2].scatter(hour.windspeed, hour.cnt, s=2, alpha=0.2)
    ax[2].set_title("windspeed vs cnt")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_weather_scatter.png", dpi=150)
    plt.close()

    # 6. correlation heatmap (numeric)
    corr = hour.select_dtypes(include=np.number).corr()
    plt.figure(figsize=(9, 7))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=60, ha="right", fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    plt.title("Hour numeric correlation heatmap")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_corr_heatmap.png", dpi=150)
    plt.close()

    print("[SAVED] reports/figures/eda_*.png (9 figures)")
    print("[INTERP] hr dominates (commuter peaks stronger on workingday); "
          "weekday flat (~177-196) but hourxweekday differs; fall>spring; clear>bad; "
          "temp +ve hum -ve; temp/atemp r~0.99 keep atemp; hum==0/windspeed==0 kept; ws_4 n=3 rare.")


if __name__ == "__main__":
    main()
