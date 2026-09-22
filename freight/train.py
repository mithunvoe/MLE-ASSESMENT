"""Fit the final model on all labelled data, optionally after running the backtests.

    python -m freight.train              # fit and save models/hybrid.joblib
    python -m freight.train --backtest   # also run the model comparison and checks
"""
from __future__ import annotations

import argparse
import time

import joblib
import pandas as pd

from freight import backtest, config, data, features, model


def prepare_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    train = data.load_train()
    validation = data.load_validation()
    index_by_date = data.daily_market_index(train, validation)
    train = features.build_features(train, index_by_date)
    corrupted = data.flag_corrupted_labels(train, model.reference_rate(train))
    return train, corrupted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backtest", action="store_true", help="run the rolling-origin model comparison first")
    args = parser.parse_args()
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)

    train, corrupted = prepare_training_frame()
    print(f"{len(train):,} training loads, {int(corrupted.sum())} flagged as corrupted labels")

    if args.backtest:
        config.REPORT_DIR.mkdir(exist_ok=True)
        started = time.time()
        results = backtest.run_backtest(train, corrupted)
        results.to_csv(config.REPORT_DIR / "backtest_results.csv", index=False)
        print("\nRolling-origin backtest, mean over folds:")
        print(backtest.summarize(results))
        quote = backtest.quote_signal_check(train, corrupted)
        quote.to_csv(config.REPORT_DIR / "quote_signal_check.csv")
        print("\nquote_signal under random vs time-based validation (clean MAE):")
        print(quote)
        cities = backtest.unseen_city_check(train, corrupted)
        cities.to_csv(config.REPORT_DIR / "unseen_city_check.csv")
        print(f"\nHeld-out cities {cities.attrs['held_out_cities']}:")
        print(cities)
        print(f"\nbacktests took {time.time() - started:.0f}s")

    final = model.HybridRateModel().fit(train[~corrupted])
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, config.MODEL_DIR / "hybrid.joblib")
    print("\nTime component coefficients (log scale):")
    print(final.time_coefficients())
    print(f"\nsaved {config.MODEL_DIR / 'hybrid.joblib'}")


if __name__ == "__main__":
    main()
