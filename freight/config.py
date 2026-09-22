"""Paths, constants and model settings shared across the package."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

# Day zero for the trend term. Training data starts here.
EPOCH = "2025-01-01"

# Rolling-origin backtest folds: (first test day, first day after the test window).
# Each fold trains on everything before the first test day. The last fold mirrors
# the real task: two months of forecasting immediately after the training window.
BACKTEST_FOLDS = (
    ("2025-05-01", "2025-07-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-09-01", "2025-11-01"),
)

# Rows whose posted_rate is outside this band relative to a simple reference model
# are treated as corrupted labels and excluded from training. The clean rows sit
# within about +/-25% of the reference, the corrupted ones are 2x-6x off.
LABEL_RATIO_BOUNDS = (0.5, 2.0)

EQUIPMENT_CODES = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}

# Load-level inputs for the gradient boosting stage.
LOAD_FEATURES = [
    "log_distance",
    "equipment_code",
    "weight_abs",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "dow",
    "days_to_quarter_end",
]
CATEGORICAL_FEATURES = ["equipment_code", "dow"]

# Boundaries (inclusive upper edge) of the days-to-quarter-end buckets in the
# time component. The quarter-end premium ramps up over the last month.
QUARTER_END_BUCKETS = (3, 7, 14, 21, 30)

LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 0,
}
LGB_ROUNDS = 1500
BACKFIT_ITERATIONS = 2
