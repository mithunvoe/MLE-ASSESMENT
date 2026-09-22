"""Rate models. The final model is HybridRateModel; the others are backtest baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

from freight import config


def time_design(df: pd.DataFrame) -> pd.DataFrame:
    """Design matrix of the time component.

    Log market index and its square (the response is convex: rates react more
    strongly when the market is hot), a linear trend, day-of-week dummies and
    days-to-quarter-end buckets for the premium that builds up over the last
    month of every quarter.
    """
    log_mi = np.log(df["market_index_daily"].to_numpy())
    cols = {
        "log_market_index": log_mi,
        "log_market_index_sq": log_mi**2,
        "trend_years": df["t"].to_numpy() / 365.0,
    }
    dow = df["dow"].to_numpy()
    for k in range(1, 7):
        cols[f"dow_{k}"] = (dow == k).astype(float)
    dtqe = df["days_to_quarter_end"].to_numpy()
    lower = -1
    for edge in config.QUARTER_END_BUCKETS:
        cols[f"quarter_end_{lower + 1}_{edge}"] = ((dtqe > lower) & (dtqe <= edge)).astype(float)
        lower = edge
    return pd.DataFrame(cols, index=df.index)


def _fit_lgb(X: pd.DataFrame, y: np.ndarray, params: dict, rounds: int) -> lgb.Booster:
    categorical = [c for c in config.CATEGORICAL_FEATURES if c in X.columns]
    dataset = lgb.Dataset(X, y, categorical_feature=categorical, free_raw_data=False)
    return lgb.train(params, dataset, num_boost_round=rounds)


class HybridRateModel:
    """log(rate) = linear time component + gradient-boosted load component.

    The two parts are fitted by backfitting: the linear part is fitted on
    log(rate) minus the current tree prediction, the tree is then fitted on
    log(rate) minus the linear part, and this is repeated. The linear part is
    the only one that sees the trend, so the model extrapolates the level
    beyond the training window in a controlled way, while the tree captures
    the non-linear load pricing (distance curve, weight, geography).
    """

    def __init__(self, lgb_params: dict | None = None, rounds: int = config.LGB_ROUNDS,
                 iterations: int = config.BACKFIT_ITERATIONS):
        self.lgb_params = dict(config.LGB_PARAMS, **(lgb_params or {}))
        self.rounds = rounds
        self.iterations = iterations
        self.time_model: LinearRegression | None = None
        self.load_model: lgb.Booster | None = None

    def fit(self, df: pd.DataFrame) -> "HybridRateModel":
        y = np.log(df["posted_rate"].to_numpy())
        X_time = time_design(df)
        X_load = df[config.LOAD_FEATURES]
        tree_part = np.zeros(len(df))
        for _ in range(self.iterations):
            self.time_model = LinearRegression().fit(X_time, y - tree_part)
            self.load_model = _fit_lgb(X_load, y - self.time_model.predict(X_time), self.lgb_params, self.rounds)
            tree_part = self.load_model.predict(X_load)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        log_rate = self.time_model.predict(time_design(df)) + self.load_model.predict(df[config.LOAD_FEATURES])
        return np.exp(log_rate)

    def time_coefficients(self) -> pd.Series:
        names = time_design(pd.DataFrame({
            "market_index_daily": [1.0], "t": [0], "dow": [0], "days_to_quarter_end": [50],
        })).columns
        return pd.Series(self.time_model.coef_, index=names, name="coefficient")


class LightGBMRateModel:
    """Plain gradient boosting on log(rate) with a configurable feature list."""

    def __init__(self, features: list[str], lgb_params: dict | None = None, rounds: int = config.LGB_ROUNDS):
        self.features = list(features)
        self.lgb_params = dict(config.LGB_PARAMS, **(lgb_params or {}))
        self.rounds = rounds
        self.model: lgb.Booster | None = None

    def fit(self, df: pd.DataFrame) -> "LightGBMRateModel":
        self.model = _fit_lgb(df[self.features], np.log(df["posted_rate"].to_numpy()), self.lgb_params, self.rounds)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(df[self.features]))


class LinearRateModel:
    """Log-linear regression: the time design plus linear load terms."""

    def __init__(self) -> None:
        self.model: LinearRegression | None = None
        self.weight_fill: float = float("nan")

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        X = time_design(df)
        X["log_distance"] = df["log_distance"].to_numpy()
        X["log_distance_sq"] = X["log_distance"] ** 2
        X["flatbed"] = (df["equipment_code"] == 1).astype(float)
        X["reefer"] = (df["equipment_code"] == 2).astype(float)
        X["weight_10k"] = df["weight_abs"].fillna(self.weight_fill).to_numpy() / 1e4
        for c in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
            X[c] = df[c].to_numpy()
        return X

    def fit(self, df: pd.DataFrame) -> "LinearRateModel":
        self.weight_fill = float(df["weight_abs"].median())
        self.model = LinearRegression().fit(self._design(df), np.log(df["posted_rate"].to_numpy()))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(self._design(df)))


class MedianRatePerMileModel:
    """Naive baseline: median rate per mile by equipment, times distance."""

    def __init__(self) -> None:
        self.rate_per_mile: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "MedianRatePerMileModel":
        self.rate_per_mile = (df["posted_rate"] / df["distance"]).groupby(df["equipment"]).median()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (df["equipment"].map(self.rate_per_mile) * df["distance"]).to_numpy()


def reference_rate(df: pd.DataFrame) -> np.ndarray:
    """Rate implied by a deliberately simple log-linear fit.

    Only used to spot corrupted labels: fit once, drop the rows that are far
    off, refit on the rest so the corrupted rows do not pull the reference.
    """
    X = pd.DataFrame({
        "log_distance": df["log_distance"],
        "log_distance_sq": df["log_distance"] ** 2,
        "flatbed": (df["equipment_code"] == 1).astype(float),
        "reefer": (df["equipment_code"] == 2).astype(float),
        "log_market_index": np.log(df["market_index_daily"]),
        "trend_years": df["t"] / 365.0,
        "weight_10k": df["weight_abs"].fillna(df["weight_abs"].median()) / 1e4,
    })
    y = np.log(df["posted_rate"].to_numpy())
    keep = np.ones(len(df), dtype=bool)
    lo, hi = np.log(config.LABEL_RATIO_BOUNDS)
    for _ in range(2):
        fit = LinearRegression().fit(X[keep], y[keep])
        resid = y - fit.predict(X)
        keep = (resid > lo) & (resid < hi)
    return np.exp(fit.predict(X))
