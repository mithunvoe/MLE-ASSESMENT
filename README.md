# Freight rate prediction

Solution for the Spotter machine learning engineer assessment: predict the posted
rate of 12,000 truckload shipments dated November and December 2025 from 48,000
labelled loads dated January to October 2025, plus a fixed 31-day December series
for one lane.

The write-up with the exploration, validation design and results is in
[`reports/report.pdf`](reports/report.pdf). The original brief is in `assessment/`.

## Approach in short

- The task is a forecast, not an interpolation: every load to predict is dated
  after the last training load. Model selection therefore uses rolling-origin
  backtests (train on months 1..k, test on the next two months), never a random split.
- Data issues handled: sign-flipped weights, missing weights and market index,
  1.4% of labels multiplied or divided by 2-6x, a `quote_signal` column whose
  relationship to the target flips sign by month and is pure noise in the
  prediction window, and eight validation cities that never appear in training.
- The model is a hybrid: a small linear time component (log market index and its
  square, a linear trend, weekday and quarter-end terms) plus LightGBM on the
  residual using load-level features (distance, equipment, weight, coordinates).
  The linear part is the only one that sees the trend, so the level extrapolates
  beyond October in a controlled way instead of being frozen at the last leaf.
- Backtest on the last fold (fit on Jan-Aug, test on Sep-Oct): MAE $39, MAPE 1.7%
  on clean loads. The mean over the three folds is MAE $38 against $51 for the
  best plain LightGBM and $167 for a rate-per-mile baseline.

## Layout

```
data/                    assessment inputs (train_test.csv, validation.csv, templates)
freight/
  config.py              paths, folds, feature lists, LightGBM settings
  data.py                loading, cleaning, daily market index, label screening
  features.py            calendar, load and geography features
  model.py               HybridRateModel and the backtest baselines
  backtest.py            rolling-origin backtests, quote_signal and unseen-city checks
  train.py               fit on all labelled data (and optionally run the backtests)
  predict.py             write the two prediction files
tests/                   unit tests (pytest)
outputs/                 validation_predictions.csv, december_chart_inputs.csv, scorer chart
reports/                 report.pdf, figures, backtest tables, training log
score.py                 Spotter's validator/chart script, unchanged
```

## Running it

Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests -q                 # unit tests, a few seconds
python -m freight.train --backtest        # backtests (~1 min) then fit the final model
python -m freight.predict                 # outputs/validation_predictions.csv and december_chart_inputs.csv
python score.py --predictions outputs/validation_predictions.csv \
                --december-predictions outputs/december_chart_inputs.csv \
                --output-dir outputs/scorer_results
```

`make all` runs the same sequence. Drop `--backtest` to only fit the final model
(a few seconds). Results are deterministic (fixed LightGBM seed).

## Outputs

- `outputs/validation_predictions.csv`: `load_id,predicted_rate` for the 12,000
  validation loads, in template order.
- `outputs/december_chart_inputs.csv`: the December template with `predicted_rate`
  filled. The template has no market index; each December day uses the mean
  `market_index` of the loads in `validation.csv` dated that day.
- `outputs/scorer_results/candidate_december.png`: chart produced by `score.py`.
