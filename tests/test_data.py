import numpy as np
import pandas as pd
import pytest

from freight import data


def _frame(**overrides):
    base = dict(
        load_id=["A", "B", "C"],
        pickup=["Lexington", "Dallas", "Lexington"],
        delivery=["Fort Wayne", "Lexington", "Dallas"],
        pickup_lat=[37.0, 32.5, 37.0],
        pickup_lon=[-85.0, -97.0, -85.0],
        delivery_lat=[41.3, 37.0, 32.5],
        delivery_lon=[-85.4, -85.0, -97.0],
        distance=[360.0, 900.0, 900.0],
        equipment=["Dry Van", "Reefer", "Flatbed"],
        weight=[32000.0, -20000.0, np.nan],
        date=pd.to_datetime(["2025-03-01", "2025-03-01", "2025-03-02"]),
        market_index=[1.0, 1.2, np.nan],
        quote_signal=[2.0, 2.1, 2.2],
        posted_rate=[800.0, 2000.0, 1900.0],
    )
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_takes_absolute_weight_and_keeps_missing():
    out = data._clean(_frame())
    assert out["weight_abs"].tolist()[:2] == [32000.0, 20000.0]
    assert np.isnan(out["weight_abs"].iloc[2])


def test_clean_drops_quote_signal():
    assert "quote_signal" not in data._clean(_frame()).columns


def test_daily_market_index_pools_frames_and_ignores_missing():
    a = _frame()
    b = _frame(date=pd.to_datetime(["2025-03-02"] * 3), market_index=[0.9, 0.9, 0.9])
    idx = data.daily_market_index(a, b)
    assert idx.loc[pd.Timestamp("2025-03-01")] == pytest.approx(1.1)
    assert idx.loc[pd.Timestamp("2025-03-02")] == pytest.approx(0.9)


def test_city_coordinates_are_unique_per_city():
    coords = data.city_coordinates(_frame())
    assert coords.loc["Lexington", "lat"] == 37.0
    assert coords.loc["Fort Wayne", "lon"] == -85.4
    assert len(coords) == 3


def test_flag_corrupted_labels_uses_ratio_band():
    train = _frame(posted_rate=[800.0, 8000.0, 400.0])
    reference = np.array([800.0, 2000.0, 1900.0])
    flagged = data.flag_corrupted_labels(train, reference)
    assert flagged.tolist() == [False, True, True]
