import numpy as np
import pandas as pd
import pytest

from freight import config, data, features, model


def synthetic_loads(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Loads generated from a known log-linear structure for recovery tests."""
    rng = np.random.RandomState(seed)
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.randint(0, 270, n), unit="D")
    frame = pd.DataFrame(
        dict(
            load_id=[f"S-{i:05d}" for i in range(n)],
            pickup=rng.choice(["A", "B", "C"], n),
            delivery=rng.choice(["D", "E", "F"], n),
            pickup_lat=rng.uniform(30, 45, n),
            pickup_lon=rng.uniform(-120, -70, n),
            delivery_lat=rng.uniform(30, 45, n),
            delivery_lon=rng.uniform(-120, -70, n),
            distance=np.exp(rng.uniform(np.log(100), np.log(3000), n)),
            equipment=rng.choice(list(config.EQUIPMENT_CODES), n),
            weight=rng.uniform(5000, 47500, n),
            date=dates,
        )
    )
    frame = features.add_time_features(features.add_load_features(frame))
    daily_index = pd.Series(np.exp(rng.normal(0, 0.15, 400)), index=pd.date_range("2025-01-01", periods=400))
    frame = features.add_market_index(frame, daily_index)
    log_rate = (
        0.9 * frame["log_distance"]
        + 0.15 * np.log(frame["market_index_daily"])
        + 0.06 * frame["t"] / 365
        + 0.05 * (frame["days_to_quarter_end"] <= 3)
        + 0.1 * (frame["equipment_code"] == 2)
        + rng.normal(0, 0.01, n)
    )
    frame["posted_rate"] = np.exp(log_rate)
    return frame


def test_hybrid_recovers_time_component_and_predicts_well():
    frame = synthetic_loads()
    fitted = model.HybridRateModel(rounds=200).fit(frame)
    coef = fitted.time_coefficients()
    assert coef["log_market_index"] == pytest.approx(0.15, abs=0.03)
    assert coef["trend_years"] == pytest.approx(0.06, abs=0.03)
    predicted = fitted.predict(frame)
    mape = np.mean(np.abs(predicted - frame["posted_rate"]) / frame["posted_rate"])
    assert mape < 0.02
    # The quarter-end premium is shared between the linear part and the tree
    # (both see days_to_quarter_end), so check the combined effect: the same
    # load on the Monday before quarter end versus three Mondays earlier.
    pair = frame.iloc[[0, 0]].copy()
    pair["date"] = pd.to_datetime(["2025-09-29", "2025-09-08"])
    pair = features.add_time_features(pair)
    ratio = fitted.predict(pair)[0] / fitted.predict(pair)[1]
    assert ratio == pytest.approx(np.exp(0.05 + 0.06 * 21 / 365), abs=0.02)


def test_reference_rate_flags_planted_corruption():
    frame = synthetic_loads(n=2000)
    frame.loc[frame.index[:10], "posted_rate"] *= 4.0
    frame.loc[frame.index[10:20], "posted_rate"] /= 3.0
    flagged = data.flag_corrupted_labels(frame, model.reference_rate(frame))
    assert flagged.iloc[:20].all()
    assert flagged.iloc[20:].sum() == 0


def test_time_design_buckets_december_dates():
    frame = features.add_time_features(
        pd.DataFrame({"date": pd.to_datetime(["2025-12-25", "2025-12-31", "2025-11-03"])})
    ).assign(market_index_daily=1.0)
    design = model.time_design(frame)
    assert design["quarter_end_4_7"].tolist() == [1.0, 0.0, 0.0]
    assert design["quarter_end_0_3"].tolist() == [0.0, 1.0, 0.0]
    assert design.filter(like="quarter_end").iloc[2].sum() == 0.0


def test_median_baseline_scales_with_distance():
    frame = pd.DataFrame(
        dict(equipment=["Dry Van", "Dry Van", "Reefer"], distance=[100.0, 200.0, 100.0], posted_rate=[200.0, 400.0, 300.0])
    )
    baseline = model.MedianRatePerMileModel().fit(frame)
    assert baseline.predict(frame).tolist() == [200.0, 400.0, 300.0]
