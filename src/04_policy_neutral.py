"""
04_policy_neutral.py — Policy-neutral forecasting (Model B)

Same architectures as Model A but with MSP-derived features removed:
  - msp_current, price_to_msp_ratio, msp_change_pct (+ their lag7 variants)
  62 features retained out of 68.

Output: Model B predictions used as input to welfare optimization (05).

Usage:
    python src/04_policy_neutral.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import (eval_metrics, load_processed_data, train_test_split_temporal,
                        get_feature_cols, print_table)
import importlib
_forecasting = importlib.import_module("src.03_forecasting")
seasonal_naive = _forecasting.seasonal_naive
fit_sarima = _forecasting.fit_sarima
sarima_aic_select = _forecasting.sarima_aic_select
fit_ets = _forecasting.fit_ets
train_lstm = _forecasting.train_lstm
train_random_forest = _forecasting.train_random_forest
train_gradient_boosting = _forecasting.train_gradient_boosting

warnings.filterwarnings("ignore")
np.random.seed(42)


def run_model_b(crop, df):
    """
    Run all 7 models with MSP features excluded.
    This is the "policy-neutral" regime — predictions from this
    feed into the welfare optimizer because they don't embed
    the very policy instrument we're trying to optimize.
    """
    print(f"\n{'='*50}")
    print(f"  {crop.upper()} — Model B (policy-neutral)")
    print(f"{'='*50}")

    feat_cols = get_feature_cols(df, policy_neutral=True)
    print(f"  Features: {len(feat_cols)} (MSP-derived excluded)")

    train_df, test_df = train_test_split_temporal(df, config.TRAIN_END, config.TEST_START)

    X_train = train_df[feat_cols].values
    y_train = train_df["price_target"].values
    X_test = test_df[feat_cols].values
    y_test = test_df["price_target"].values
    n_test = len(y_test)

    # impute NaN
    col_means = np.nanmean(X_train, axis=0)
    for j in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, j]), j] = col_means[j]
        X_test[np.isnan(X_test[:, j]), j] = col_means[j]
    y_train = np.nan_to_num(y_train, nan=np.nanmean(y_train))
    y_test = np.nan_to_num(y_test, nan=np.nanmean(y_test))

    results = []
    predictions = {}

    # 1. Naive (same as Model A — doesn't use features)
    print("  [1/7] Seasonal Naive...")
    pred_naive = seasonal_naive(train_df, test_df)[:n_test]
    results.append(eval_metrics(y_test, pred_naive, "Seasonal Naive"))
    predictions["naive"] = pred_naive

    # 2. SARIMA AIC (same — univariate)
    print("  [2/7] SARIMA AIC...")
    try:
        ts = train_df.groupby("date")["price_modal"].mean().values[-365*2:]
        pred_sarima = sarima_aic_select(ts, n_test)[:n_test]
        results.append(eval_metrics(y_test, pred_sarima, "SARIMA (AIC)"))
        predictions["sarima_aic"] = pred_sarima
    except Exception as e:
        print(f"    Failed: {e}")
        results.append({"model": "SARIMA (AIC)", "rmse": np.nan, "mae": np.nan,
                        "mape": np.nan, "r2": np.nan, "dir_acc": np.nan})

    # 3. SARIMA fixed
    print("  [3/7] SARIMA(1,1,1)(1,1,1)_7...")
    try:
        ts = train_df.groupby("date")["price_modal"].mean().values[-365*2:]
        pred_sf = fit_sarima(ts, config.SARIMA_FIXED_ORDER,
                             config.SARIMA_SEASONAL_ORDER, n_test)[:n_test]
        results.append(eval_metrics(y_test, pred_sf, "SARIMA(1,1,1)(1,1,1)_7"))
        predictions["sarima_fix"] = pred_sf
    except Exception as e:
        print(f"    Failed: {e}")
        results.append({"model": "SARIMA(1,1,1)(1,1,1)_7", "rmse": np.nan,
                        "mae": np.nan, "mape": np.nan, "r2": np.nan, "dir_acc": np.nan})

    # 4. ETS (univariate, same as A)
    print("  [4/7] ETS...")
    try:
        ts = train_df.groupby("date")["price_modal"].mean().values[-365*2:]
        pred_ets = fit_ets(ts, n_test)[:n_test]
        results.append(eval_metrics(y_test, pred_ets, "ETS"))
        predictions["ets"] = pred_ets
    except:
        results.append({"model": "ETS", "rmse": np.nan, "mae": np.nan,
                        "mape": np.nan, "r2": np.nan, "dir_acc": np.nan})

    # 5. Random Forest (B) — this is the key one for welfare optimization
    print("  [5/7] Random Forest (Model B)...")
    pred_rf, rf_model = train_random_forest(X_train, y_train, X_test)
    results.append(eval_metrics(y_test, pred_rf, "Random Forest"))
    predictions["rf"] = pred_rf
    joblib.dump(rf_model, config.RESULTS / f"rf_model_b_{crop}.pkl")

    # feature importance for Model B
    fi = pd.DataFrame({
        "feature": feat_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(config.RESULTS / f"feature_importance_b_{crop}.csv", index=False)
    print(f"    Top feature: {fi.iloc[0]['feature']} "
          f"({fi.iloc[0]['importance']*100:.1f}%)")

    # 6. Gradient Boosting (B)
    print("  [6/7] Gradient Boosting (Model B)...")
    pred_gb, gb_model = train_gradient_boosting(X_train, y_train, X_test)
    results.append(eval_metrics(y_test, pred_gb, "Gradient Boosting"))
    predictions["gb"] = pred_gb

    # 7. LSTM (B)
    print("  [7/7] LSTM (Model B)...")
    try:
        pred_lstm, _ = train_lstm(X_train, y_train, X_test, y_test)
        pred_lstm = pred_lstm[:n_test]
        if len(pred_lstm) < n_test:
            pred_lstm = np.pad(pred_lstm, (n_test - len(pred_lstm), 0), mode="edge")
        results.append(eval_metrics(y_test, pred_lstm, "LSTM"))
        predictions["lstm"] = pred_lstm
    except Exception as e:
        print(f"    LSTM failed: {e}")
        results.append({"model": "LSTM", "rmse": np.nan, "mae": np.nan,
                        "mape": np.nan, "r2": np.nan, "dir_acc": np.nan})

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("mape")
    res_df["crop"] = crop
    print_table(res_df, f"Model B Results — {crop.upper()}")
    res_df.to_csv(config.RESULTS / f"model_b_results_{crop}.csv", index=False)

    # save predictions
    pred_df = pd.DataFrame({"date": test_df["date"].values[:n_test],
                             "actual": y_test})
    for name, vals in predictions.items():
        v = vals[:n_test] if len(vals) >= n_test else \
            np.pad(vals, (0, n_test - len(vals)), mode="edge")
        pred_df[name] = v
    pred_df.to_csv(config.RESULTS / f"predictions_b_{crop}.csv", index=False)

    # the forecast price used in welfare optimization is the RF mean prediction
    # over the full test period
    rf_mean_forecast = np.mean(pred_rf)
    print(f"\n  RF Model B mean forecast ({crop}): Rs {rf_mean_forecast:.0f}/quintal")

    return res_df, predictions, y_test, rf_mean_forecast


if __name__ == "__main__":
    print("=" * 60)
    print("  MSP Welfare Optimization — Model B (policy-neutral)")
    print("=" * 60)

    forecasts = {}

    for crop in config.COMMODITIES:
        df = load_processed_data(crop, config.DATA_PROC)
        res, preds, y_test, fc_mean = run_model_b(crop, df)
        forecasts[crop] = fc_mean

    # save the forecast prices for welfare optimization
    fc_df = pd.DataFrame([
        {"crop": crop, "forecast_price": price}
        for crop, price in forecasts.items()
    ])
    fc_df.to_csv(config.RESULTS / "model_b_forecasts.csv", index=False)

    print("\n" + "=" * 60)
    print("  Model B complete. Forecast prices saved for welfare optimization.")
    print("=" * 60)
