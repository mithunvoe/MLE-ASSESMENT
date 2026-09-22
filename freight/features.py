"""Feature construction. Every function returns a new frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from freight import config


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features that transfer to dates outside the training window.

    Month and day-of-year are deliberately excluded: November and December
    never occur in training, so a tree would map them to arbitrary leaves.
    Day-of-week and position within the quarter take the same values in Q4
    as in Q1-Q3, and the trend term `t` is only used by the linear component.
    """
    out = df.copy()
    date = out["date"]
    out["dow"] = date.dt.dayofweek
    out["t"] = (date - pd.Timestamp(config.EPOCH)).dt.days
    quarter = date.dt.to_period("Q")
    out["days_to_quarter_end"] = (quarter.dt.end_time.dt.normalize() - date).dt.days
    out["day_of_quarter"] = (date - quarter.dt.start_time).dt.days + 1
    return out


def add_load_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_distance"] = np.log(out["distance"])
    out["equipment_code"] = out["equipment"].map(config.EQUIPMENT_CODES)
    if out["equipment_code"].isna().any():
        unknown = sorted(set(out.loc[out["equipment_code"].isna(), "equipment"]))
        raise ValueError(f"unknown equipment values: {unknown}")
    if "weight_abs" not in out:
        out["weight_abs"] = out["weight"].abs()
    return out


def add_market_index(df: pd.DataFrame, index_by_date: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["market_index_daily"] = out["date"].map(index_by_date)
    if out["market_index_daily"].isna().any():
        missing = sorted(out.loc[out["market_index_daily"].isna(), "date"].dt.strftime("%Y-%m-%d").unique())
        raise ValueError(f"no market index for dates: {missing[:5]}")
    return out


def add_coordinates(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Attach pickup/delivery coordinates by city name (for the December rows)."""
    out = df.copy()
    for role in ("pickup", "delivery"):
        unknown = sorted(set(out[role]) - set(coords.index))
        if unknown:
            raise ValueError(f"no coordinates for {role} cities: {unknown}")
        out[f"{role}_lat"] = out[role].map(coords["lat"])
        out[f"{role}_lon"] = out[role].map(coords["lon"])
    return out


def build_features(df: pd.DataFrame, index_by_date: pd.Series) -> pd.DataFrame:
    return add_market_index(add_time_features(add_load_features(df)), index_by_date)
