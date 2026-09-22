"""Loading and cleaning of the assessment CSVs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from freight import config


def load_train() -> pd.DataFrame:
    return _clean(pd.read_csv(config.TRAIN_PATH, parse_dates=["date"]))


def load_validation() -> pd.DataFrame:
    return _clean(pd.read_csv(config.VALIDATION_PATH, parse_dates=["date"]))


def load_december() -> pd.DataFrame:
    return pd.read_csv(config.DECEMBER_PATH, parse_dates=["date"])


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the feature-level issues found during exploration.

    - Roughly 1% of weights are negative. Their magnitudes follow the same
      distribution as the positive weights and their rates match heavy loads,
      so the sign is a recording error and we take the absolute value.
    - Missing weights are left as NaN; LightGBM handles them natively and the
      affected loads are otherwise unremarkable.
    - quote_signal is dropped. Its relationship with the target flips sign from
      month to month and is absent in the prediction window (see report).
    """
    out = df.copy()
    out["weight_abs"] = out["weight"].abs()
    out = out.drop(columns=["quote_signal"], errors="ignore")
    return out


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per date, pooled over every frame that has one.

    market_index is a market-wide daily series observed with per-load noise
    (within-day SD 0.025, no correlation between a load's deviation from the
    daily mean and its rate). Averaging the ~150-200 loads posted on a day
    recovers the underlying value, fills the ~1% missing entries and gives
    a defensible value for the December chart rows, which carry no index.
    """
    stacked = pd.concat(
        [f[["date", "market_index"]] for f in frames if "market_index" in f], ignore_index=True
    )
    return stacked.groupby("date")["market_index"].mean().rename("market_index_daily")


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) lookup. Every city has exactly one coordinate pair."""
    parts = []
    for f in frames:
        parts.append(
            f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        )
        parts.append(
            f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        )
    coords = pd.concat(parts, ignore_index=True).drop_duplicates("city").set_index("city")
    return coords.sort_index()


def flag_corrupted_labels(train: pd.DataFrame, reference: np.ndarray) -> pd.Series:
    """Boolean mask of rows whose posted_rate is implausible given a reference fit.

    `reference` holds the reference model's predicted rate for each row. The
    ratio actual/reference is either within ~0.8-1.25 (clean) or in
    0.17-0.45 / 2.2-5.3 (corrupted), so any cutoff in the gaps works.
    """
    ratio = train["posted_rate"].to_numpy() / reference
    lo, hi = config.LABEL_RATIO_BOUNDS
    return pd.Series((ratio < lo) | (ratio > hi), index=train.index, name="corrupted")
