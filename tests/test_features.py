import numpy as np
import pandas as pd
import pytest

from freight import features


def _frame(dates):
    n = len(dates)
    return pd.DataFrame(
        dict(
            pickup=["Lexington"] * n,
            delivery=["Fort Wayne"] * n,
            distance=[360.0] * n,
            equipment=["Dry Van"] * n,
            weight=[32000.0] * n,
            date=pd.to_datetime(dates),
        )
    )


def test_time_features_position_in_quarter():
    out = features.add_time_features(_frame(["2025-03-31", "2025-12-25", "2025-10-01"]))
    assert out["days_to_quarter_end"].tolist() == [0, 6, 91]
    assert out["day_of_quarter"].tolist() == [90, 86, 1]
    assert out["dow"].tolist() == [0, 3, 2]
    assert out["t"].tolist() == [89, 358, 273]


def test_load_features_encode_equipment_and_distance():
    out = features.add_load_features(_frame(["2025-01-01"]))
    assert out["equipment_code"].iloc[0] == 0
    assert out["log_distance"].iloc[0] == pytest.approx(np.log(360.0))
    assert out["weight_abs"].iloc[0] == 32000.0


def test_load_features_reject_unknown_equipment():
    frame = _frame(["2025-01-01"]).assign(equipment=["Tanker"])
    with pytest.raises(ValueError, match="Tanker"):
        features.add_load_features(frame)


def test_market_index_lookup_requires_every_date():
    idx = pd.Series({pd.Timestamp("2025-01-01"): 1.0})
    out = features.add_market_index(_frame(["2025-01-01"]), idx)
    assert out["market_index_daily"].iloc[0] == 1.0
    with pytest.raises(ValueError, match="2025-01-02"):
        features.add_market_index(_frame(["2025-01-02"]), idx)


def test_coordinates_attached_by_city():
    coords = pd.DataFrame({"lat": [37.0, 41.3], "lon": [-85.0, -85.4]}, index=["Lexington", "Fort Wayne"])
    out = features.add_coordinates(_frame(["2025-12-01"]), coords)
    assert out["pickup_lat"].iloc[0] == 37.0
    assert out["delivery_lon"].iloc[0] == -85.4
    with pytest.raises(ValueError, match="Nowhere"):
        features.add_coordinates(_frame(["2025-12-01"]).assign(delivery="Nowhere"), coords)
