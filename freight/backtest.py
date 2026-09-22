"""Rolling-origin backtests used for model selection and for the report."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from freight import config, data, model


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(100 * np.mean(np.abs(err) / actual)),
        "R2": float(1 - np.sum(err**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def candidate_models() -> dict[str, callable]:
    load = [f for f in config.LOAD_FEATURES if f not in ("dow", "days_to_quarter_end")]
    calendar = ["dow", "days_to_quarter_end", "day_of_quarter"]
    return {
        "median rate per mile by equipment": model.MedianRatePerMileModel,
        "log-linear regression": model.LinearRateModel,
        "lightgbm: load + calendar": lambda: model.LightGBMRateModel(load + calendar),
        "lightgbm: + market index": lambda: model.LightGBMRateModel(load + calendar + ["market_index_daily"]),
        "lightgbm: + market index + day index": lambda: model.LightGBMRateModel(load + calendar + ["market_index_daily", "t"]),
        "hybrid: linear time component + lightgbm": model.HybridRateModel,
    }


def _fold_masks(train: pd.DataFrame, start: str, end: str) -> tuple[pd.Series, pd.Series]:
    fit_mask = train["date"] < pd.Timestamp(start)
    test_mask = (train["date"] >= pd.Timestamp(start)) & (train["date"] < pd.Timestamp(end))
    return fit_mask, test_mask


def run_backtest(train: pd.DataFrame, corrupted: pd.Series, models: dict | None = None) -> pd.DataFrame:
    """Fit every candidate on each fold's training window and score the test window.

    Corrupted labels are excluded from fitting (flagged within the training
    window only) and metrics are reported both on clean test rows and on all
    test rows, since the hidden validation labels presumably carry the same
    corruption.
    """
    models = models or candidate_models()
    rows = []
    for start, end in config.BACKTEST_FOLDS:
        fit_mask, test_mask = _fold_masks(train, start, end)
        fit_rows = train[fit_mask]
        fit_rows = fit_rows[~data.flag_corrupted_labels(fit_rows, model.reference_rate(fit_rows))]
        test_rows = train[test_mask]
        actual = test_rows["posted_rate"].to_numpy()
        clean = ~corrupted[test_mask].to_numpy()
        for name, make in models.items():
            predicted = make().fit(fit_rows).predict(test_rows)
            row = {"model": name, "fold": f"{start[:7]} to {end[:7]}", "n_test": int(test_mask.sum())}
            row.update({f"{k}_clean": v for k, v in metrics(actual[clean], predicted[clean]).items()})
            row.update({f"{k}_all": v for k, v in metrics(actual, predicted).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["MAE_clean", "RMSE_clean", "MAPE_clean", "R2_clean", "MAE_all", "MAPE_all"]
    order = list(dict.fromkeys(results["model"]))
    return results.groupby("model")[cols].mean().loc[order]


def quote_signal_check(train: pd.DataFrame, corrupted: pd.Series) -> pd.DataFrame:
    """Show how the split choice changes the verdict on quote_signal.

    A random split makes the feature look useful. The time-based folds show
    the gain is not reliable: it helps in some test months and hurts in
    others, depending on which regime the month happens to be in, and in the
    prediction window the signal is indistinguishable from noise (see report).
    """
    quote = pd.read_csv(config.TRAIN_PATH, usecols=["load_id", "quote_signal"])
    frame = train.merge(quote, on="load_id", how="left")
    base = [f for f in config.LOAD_FEATURES if f not in ("dow", "days_to_quarter_end")]
    base += ["dow", "days_to_quarter_end", "day_of_quarter", "market_index_daily", "t"]
    variants = {"without quote_signal": base, "with quote_signal": base + ["quote_signal"]}
    clean_frame = frame[~corrupted.to_numpy()]
    table: dict[str, dict[str, float]] = {}
    for name, feats in variants.items():
        errors = []
        for fit_idx, test_idx in KFold(5, shuffle=True, random_state=0).split(clean_frame):
            m = model.LightGBMRateModel(feats, rounds=500).fit(clean_frame.iloc[fit_idx])
            test = clean_frame.iloc[test_idx]
            errors.append(metrics(test["posted_rate"].to_numpy(), m.predict(test))["MAE"])
        column = {"random 5-fold, all months": float(np.mean(errors))}
        for start, end in config.BACKTEST_FOLDS:
            fit_mask, test_mask = _fold_masks(clean_frame, start, end)
            m = model.LightGBMRateModel(feats, rounds=500).fit(clean_frame[fit_mask])
            test = clean_frame[test_mask].assign(predicted=lambda d: m.predict(d))
            for month, g in test.groupby(test["date"].dt.month):
                label = f"time-based, test {pd.Timestamp(2025, month, 1):%b}"
                column[label] = metrics(g["posted_rate"].to_numpy(), g["predicted"].to_numpy())["MAE"]
        table[name] = column
    return pd.DataFrame(table)


def unseen_city_check(train: pd.DataFrame, corrupted: pd.Series, n_cities: int = 6, seed: int = 1) -> pd.DataFrame:
    """Refit the final model on the last fold with a few cities removed and compare
    errors on test loads that touch them against the rest. The validation set has
    eight cities that never occur in training, so this matters."""
    start, end = config.BACKTEST_FOLDS[-1]
    fit_mask, test_mask = _fold_masks(train, start, end)
    cities = sorted(set(train["pickup"]))
    held = list(np.random.RandomState(seed).choice(cities, n_cities, replace=False))
    fit_rows = train[fit_mask & ~train["pickup"].isin(held) & ~train["delivery"].isin(held)]
    fit_rows = fit_rows[~data.flag_corrupted_labels(fit_rows, model.reference_rate(fit_rows))]
    test_rows = train[test_mask & ~corrupted]
    predicted = model.HybridRateModel().fit(fit_rows).predict(test_rows)
    touches = (test_rows["pickup"].isin(held) | test_rows["delivery"].isin(held)).to_numpy()
    actual = test_rows["posted_rate"].to_numpy()
    out = pd.DataFrame({
        "loads touching held-out cities": metrics(actual[touches], predicted[touches]),
        "other loads": metrics(actual[~touches], predicted[~touches]),
    }).T
    out["n"] = [int(touches.sum()), int((~touches).sum())]
    out.attrs["held_out_cities"] = held
    return out
