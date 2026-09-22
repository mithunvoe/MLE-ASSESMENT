"""Figures for the report. Run after `python -m freight.train --backtest`.

    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freight import config, data, features, model  # noqa: E402
from freight.train import prepare_training_frame  # noqa: E402

FIG_DIR = config.REPORT_DIR / "figures"
BLUE, RED, GREY, GREEN = "#1f4e79", "#c0392b", "#7f8c8d", "#2e7d32"
EQUIP_COLORS = {"Dry Van": BLUE, "Flatbed": GREEN, "Reefer": RED}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG_DIR / name)


def fig_timeseries(train: pd.DataFrame, validation: pd.DataFrame, clean: pd.Series) -> None:
    rpm = (train["posted_rate"] / train["distance"])[clean].groupby(train["date"]).mean()
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    axes[0].plot(rpm.index, rpm, color=BLUE, lw=0.8, alpha=0.6, label="daily mean")
    axes[0].plot(rpm.index, rpm.rolling(7, center=True).mean(), color=BLUE, lw=1.8, label="7-day mean")
    for q_end in pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30"]):
        axes[0].axvline(q_end, color=GREY, lw=0.7, ls=":")
    axes[0].set_ylabel("rate per mile ($)")
    axes[0].set_title("Posted rate per mile, training loads (corrupted labels excluded)", loc="left")
    axes[0].legend(frameon=False, loc="upper left")
    for frame, color, label in ((train, BLUE, "training"), (validation, RED, "validation window")):
        mi = frame.groupby("date")["market_index"].mean()
        axes[1].plot(mi.index, mi, color=color, lw=1.0, label=label)
    axes[1].set_ylabel("market index")
    axes[1].set_title("Daily mean market_index (weekly cycle on a shifting level)", loc="left")
    axes[1].legend(frameon=False, loc="upper right")
    save(fig, "timeseries.png")


def fig_quote_signal(train: pd.DataFrame, validation: pd.DataFrame, clean: pd.Series) -> None:
    raw = pd.read_csv(config.TRAIN_PATH, usecols=["load_id", "quote_signal"])
    t = train.merge(raw, on="load_id")[clean.to_numpy()]
    t = t.assign(rpm=t["posted_rate"] / t["distance"], month=t["date"].dt.month)
    corr = pd.Series({m: g["quote_signal"].corr(g["rpm"]) for m, g in t.groupby("month")})
    v = pd.read_csv(config.VALIDATION_PATH, parse_dates=["date"]).assign(month=lambda d: d["date"].dt.month)
    both = pd.concat([t[["month", "equipment", "quote_signal"]], v[["month", "equipment", "quote_signal"]]])
    by_equip = both.groupby(["month", "equipment"])["quote_signal"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    colors = [BLUE if c > 0.5 else RED if c < -0.5 else GREY for c in corr]
    axes[0].bar(corr.index, corr, color=colors)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xticks(range(1, 11))
    axes[0].set_ylabel("Pearson correlation")
    axes[0].set_title("quote_signal vs rate per mile, by month", loc="left")
    for equip, series in by_equip.items():
        axes[1].plot(series.index, series, marker="o", ms=3, color=EQUIP_COLORS[equip], label=equip)
    axes[1].axvspan(10.5, 12.5, color=GREY, alpha=0.15, label="validation window")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_ylabel("mean quote_signal")
    axes[1].set_title("Mean quote_signal by equipment and month", loc="left")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower left")
    save(fig, "quote_signal.png")


def fig_label_noise(train: pd.DataFrame) -> None:
    ratio = np.log10(train["posted_rate"] / model.reference_rate(train))
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.hist(ratio, bins=160, color=BLUE)
    ax.set_yscale("log")
    for cut in config.LABEL_RATIO_BOUNDS:
        ax.axvline(np.log10(cut), color=RED, lw=1, ls="--")
    ax.set_xlabel("log10(posted_rate / reference rate)")
    ax.set_ylabel("loads (log scale)")
    ax.set_title("Posted rate relative to a simple reference fit; dashed lines are the exclusion cut-offs", loc="left")
    save(fig, "label_noise.png")


def fig_quarter_end(train: pd.DataFrame, clean: pd.Series) -> None:
    t = train[clean]
    daily = pd.DataFrame({
        "log_rpm": np.log(t["posted_rate"] / t["distance"]).groupby(t["date"]).mean(),
        "log_mi": np.log(t.groupby("date")["market_index_daily"].first()),
    })
    daily["trend"] = (daily.index - pd.Timestamp(config.EPOCH)).days / 365
    for k in range(1, 7):
        daily[f"dow{k}"] = (daily.index.dayofweek == k).astype(float)
    fit = LinearRegression().fit(daily.drop(columns="log_rpm"), daily["log_rpm"])
    daily["resid"] = daily["log_rpm"] - fit.predict(daily.drop(columns="log_rpm"))
    quarter = daily.index.to_period("Q")
    daily["days_to_end"] = (quarter.end_time.normalize() - daily.index).days
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for q, color in zip(("2025Q1", "2025Q2", "2025Q3"), (BLUE, GREEN, RED)):
        g = daily[quarter == q].sort_values("days_to_end")
        ax.plot(g["days_to_end"], 100 * g["resid"].rolling(3, center=True).mean(), color=color, lw=1.4, label=q[-2:])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(45, 0)
    ax.set_xlabel("days to quarter end")
    ax.set_ylabel("residual (%)")
    ax.set_title("Daily rate residual after market index, trend and weekday effects", loc="left")
    ax.legend(frameon=False)
    save(fig, "quarter_end.png")


def fig_city_effects(train: pd.DataFrame, validation: pd.DataFrame, clean: pd.Series) -> None:
    t = train[clean]
    resid = np.log(t["posted_rate"] / model.reference_rate(t))
    coords = data.city_coordinates(train, validation)
    effect = pd.DataFrame({
        "pickup": resid.groupby(t["pickup"]).mean(),
        "delivery": resid.groupby(t["delivery"]).mean(),
    }).join(coords)
    new = sorted(set(coords.index) - set(effect.index))
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.scatter(effect["lat"], 100 * effect["pickup"], s=14, color=BLUE, label="as pickup")
    ax.scatter(effect["lat"], 100 * effect["delivery"], s=14, color=RED, marker="^", label="as delivery")
    ax.scatter(coords.loc[new, "lat"], np.zeros(len(new)) - 5.2, marker="|", s=80, color=GREY,
               label="validation-only cities (latitude)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("latitude")
    ax.set_ylabel("city effect (%)")
    ax.set_title("City premium versus latitude; the eight unseen cities sit inside the covered range", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    save(fig, "city_effects.png")


def fig_backtest() -> None:
    path = config.REPORT_DIR / "backtest_results.csv"
    if not path.exists():
        print("skipping backtest figure, run `python -m freight.train --backtest` first")
        return
    results = pd.read_csv(path)
    order = list(dict.fromkeys(results["model"]))
    folds = list(dict.fromkeys(results["fold"]))
    labels = {
        "median rate per mile by equipment": "median rate\nper mile",
        "log-linear regression": "log-linear\nregression",
        "lightgbm: load + calendar": "LightGBM\nload + calendar",
        "lightgbm: + market index": "LightGBM\n+ market index",
        "lightgbm: + market index + day index": "LightGBM\n+ index + day index",
        "hybrid: linear time component + lightgbm": "hybrid\n(final)",
    }
    fig, ax = plt.subplots(figsize=(8, 3.2))
    width = 0.8 / len(folds)
    for i, fold in enumerate(folds):
        sub = results[results["fold"] == fold].set_index("model").loc[order]
        first, last = fold.split(" to ")
        months = f"{pd.Timestamp(first):%b}-{pd.Timestamp(last) - pd.DateOffset(months=1):%b}"
        ax.bar(np.arange(len(order)) + i * width, sub["MAE_clean"], width, label=f"test {months}")
    ax.set_xticks(np.arange(len(order)) + width)
    ax.set_xticklabels([labels.get(o, o) for o in order], fontsize=8)
    ax.set_ylabel("MAE on clean test loads ($)")
    ax.set_title("Rolling-origin backtest, each fold trained on all earlier months", loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "backtest.png")


def fig_holdout_tracking(train: pd.DataFrame, corrupted: pd.Series) -> None:
    start, end = config.BACKTEST_FOLDS[-1]
    fit_rows = train[train["date"] < start]
    fit_rows = fit_rows[~data.flag_corrupted_labels(fit_rows, model.reference_rate(fit_rows))]
    test = train[(train["date"] >= start) & ~corrupted].copy()
    test["predicted"] = model.HybridRateModel().fit(fit_rows).predict(test)
    daily = pd.DataFrame({
        "actual": (test["posted_rate"] / test["distance"]).groupby(test["date"]).mean(),
        "predicted": (test["predicted"] / test["distance"]).groupby(test["date"]).mean(),
    })
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(daily.index, daily["actual"], color=BLUE, lw=1.4, label="actual")
    ax.plot(daily.index, daily["predicted"], color=RED, lw=1.4, ls="--", label="predicted")
    ax.axvline(pd.Timestamp("2025-09-30"), color=GREY, lw=0.7, ls=":")
    ax.set_ylabel("mean rate per mile ($)")
    ax.set_title("Last backtest fold: model fitted on Jan-Aug, daily means over Sep-Oct", loc="left")
    ax.legend(frameon=False)
    save(fig, "holdout_tracking.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train, corrupted = prepare_training_frame()
    validation = data.load_validation()
    clean = ~corrupted
    fig_timeseries(train, validation, clean)
    fig_quote_signal(train, validation, clean)
    fig_label_noise(train)
    fig_quarter_end(train, clean)
    fig_city_effects(train, validation, clean)
    fig_backtest()
    fig_holdout_tracking(train, corrupted)


if __name__ == "__main__":
    main()
