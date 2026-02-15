"""
config.py — Central configuration for MSP welfare optimization pipeline

All parameters referenced in the paper are defined here so results are
reproducible without hunting through individual scripts. If you change
something here it propagates everywhere.

Author: Param Suneej Agarwal
Last updated: Feb 2026
"""

import os
from pathlib import Path

# ── paths ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
RESULTS     = ROOT / "results"
FIGURES     = ROOT / "figures"

for d in [DATA_RAW, DATA_PROC, DATA_SAMPLE, RESULTS, FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# ── data specifications ────────────────────────────────
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

# 10 APMC mandis across 5 states (actual market names from Agmarknet)
MARKETS = {
    "Punjab":         ["Amritsar", "Ludhiana"],
    "Haryana":        ["Karnal", "Hisar"],
    "Uttar Pradesh":  ["Lucknow", "Kanpur"],
    "Madhya Pradesh": ["Indore", "Bhopal"],
    "West Bengal":    ["Kolkata", "Burdwan"],
}

COMMODITIES = ["wheat", "rice"]

# temporal splits for train/val/test
TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
TEST_START = "2024-01-01"

# rolling CV folds (expanding window)
CV_FOLDS = [
    {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-06-30"},
    {"train_end": "2024-06-30", "test_start": "2024-07-01", "test_end": "2024-12-31"},
]

# ── feature engineering ────────────────────────────────
# total features = 68 (policy-conditioned) or 62 (policy-neutral)
POLICY_FEATURES = ["msp_current", "price_to_msp_ratio", "msp_change_pct"]
N_FEATURES_TOTAL = 68
N_FEATURES_NEUTRAL = 62  # 68 - 6 (3 policy features * 2 lag variants)

WEATHER_VARS = [
    "rainfall_7d_cum",
    "temp_anomaly",
    "extreme_heat_days",
    "drought_indicator",
    "max_temp",
    "min_temp",
    "humidity",
    "rainfall_daily",
    "temp_range",
]

# ── model hyperparameters ──────────────────────────────
# Random Forest
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 10,
    "random_state": 42,
    "n_jobs": -1,
}

# Gradient Boosting
GB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": 42,
}

# LSTM
LSTM_PARAMS = {
    "n_layers": 2,
    "hidden_units": 128,
    "lookback": 30,
    "epochs": 100,
    "batch_size": 32,
    "patience": 10,   # early stopping
}

# SARIMA — order selected by AIC on each fold, but the fixed-order variant is:
SARIMA_FIXED_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)  # weekly seasonality

# ── welfare optimization ───────────────────────────────
# elasticity calibration (from literature — see Table 2 in paper)
ELASTICITIES = {
    "wheat": {"supply": 0.45, "supply_se": 0.090,
              "demand": -0.32, "demand_se": 0.064},
    "rice":  {"supply": 0.38, "supply_se": 0.076,
              "demand": -0.30, "demand_se": 0.060},
}

KAPPA = 0.15  # procurement overhead fraction

# policy weight scenarios
SCENARIOS = {
    "farmer_centric":   {"alpha": 0.6, "beta": 0.2, "gamma": 0.2, "delta": 0.0},
    "consumer_centric": {"alpha": 0.2, "beta": 0.6, "gamma": 0.2, "delta": 0.0},
    "fiscal_conserv":   {"alpha": 0.33, "beta": 0.33, "gamma": 0.34, "delta": 0.0},
    "equal_weights":    {"alpha": 0.33, "beta": 0.33, "gamma": 0.33, "delta": 0.0},
}

# current MSP levels (2024-25 marketing season)
CURRENT_MSP = {"wheat": 2425, "rice": 2800}

# FCI procurement volumes 2024, million metric tonnes
PROCUREMENT_2024 = {"wheat": 26.64, "rice": 52.89}

# base market quantities (total production, MMT) — approx from Econ Survey
TOTAL_PRODUCTION = {"wheat": 112.0, "rice": 137.0}

# consumer-facing quantity = total production - procurement
Q_CONSUMER = {
    c: TOTAL_PRODUCTION[c] - PROCUREMENT_2024[c] for c in COMMODITIES
}

# ── Monte Carlo ────────────────────────────────────────
MC_N_DRAWS = 1000
MC_SEED = 2024
FORECAST_RMSE_FRACTION = 0.30  # SD of forecast perturbation = 30% of RMSE

# ── sensitivity analysis ───────────────────────────────
PROCUREMENT_ETAS = [0, 0.5, 1.0, 2.0, 3.0]  # MMT per Rs 100/qtl
ELASTICITY_MULTIPLIERS = [0.5, 1.0, 1.5]      # for stress test grid

# ── evaluation horizons (days) ─────────────────────────
HORIZONS = {"1-month": 30, "3-month": 90, "6-month": 180}

# ── plotting ───────────────────────────────────────────
FIGSIZE = (12, 5)
DPI = 300
COLORS = {
    "rf": "#2196F3",
    "gb": "#4CAF50",
    "lstm": "#FF9800",
    "sarima": "#F44336",
    "naive": "#9E9E9E",
    "ets": "#9C27B0",
    "current_msp": "#D32F2F",
    "optimal": "#1B5E20",
    "forecast": "#1565C0",
}
