PY ?= .venv/bin/python

.PHONY: setup test train predict score all clean

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q -r requirements.txt

test:
	$(PY) -m pytest tests -q

train:
	$(PY) -m freight.train --backtest

predict:
	$(PY) -m freight.predict

score:
	$(PY) score.py --predictions outputs/validation_predictions.csv \
	    --december-predictions outputs/december_chart_inputs.csv \
	    --output-dir outputs/scorer_results

all: test train predict score

clean:
	rm -rf outputs models/*.joblib reports/backtest_results.csv reports/*_check.csv
