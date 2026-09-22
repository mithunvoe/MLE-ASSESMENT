"""Predict the 12,000 validation loads and the fixed December series.

    python -m freight.predict

Writes outputs/validation_predictions.csv and outputs/december_chart_inputs.csv.
"""
from __future__ import annotations

import joblib
import pandas as pd

from freight import config, data, features


def december_frame(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """The chart template only has city names, so coordinates are looked up."""
    december = data.load_december()
    coords = data.city_coordinates(train, validation)
    december = features.add_coordinates(december, coords)
    december["weight_abs"] = december["weight"].abs()
    return december


def main() -> None:
    train = data.load_train()
    validation = data.load_validation()
    index_by_date = data.daily_market_index(train, validation)
    hybrid = joblib.load(config.MODEL_DIR / "hybrid.joblib")

    validation = features.build_features(validation, index_by_date)
    validation["predicted_rate"] = hybrid.predict(validation).round(2)
    template = pd.read_csv(config.TEMPLATE_PATH, usecols=["load_id"])
    submission = template.merge(validation[["load_id", "predicted_rate"]], on="load_id", how="left")
    if submission["predicted_rate"].isna().any():
        raise RuntimeError("template contains load_ids that are not in validation.csv")

    december = december_frame(train, validation)
    scored = features.build_features(december, index_by_date)
    december["predicted_rate"] = hybrid.predict(scored).round(2)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(config.OUTPUT_DIR / "validation_predictions.csv", index=False)
    december[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]].to_csv(
        config.OUTPUT_DIR / "december_chart_inputs.csv", index=False, date_format="%Y-%m-%d"
    )
    print(f"validation: {len(submission):,} rows, mean predicted rate {submission['predicted_rate'].mean():,.2f}")
    print(f"december: {len(december)} rows, range {december['predicted_rate'].min():,.2f} to {december['predicted_rate'].max():,.2f}")


if __name__ == "__main__":
    main()
