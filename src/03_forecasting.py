"""
03_forecasting.py — Seven-architecture forecast comparison (Model A)

Trains and evaluates all seven models on the test period (Jan-Dec 2024).
Implements rolling-origin cross-validation (3 folds) and Diebold-Mariano tests.

Model A = policy-conditioned (all 68 features including MSP-derived).
Model B (MSP features excluded) is in 04_policy_neutral.py.

Usage:
    python src/03_forecasting.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import (eval_metrics, eval_at_horizons, diebold_mariano_test,
                        load_processed_data, train_test_split_temporal,
                        get_feature_cols, print_table)

warnings.filterwarnings("ignore")
np.random.seed(42)


# ── 1. Seasonal Naive ─────────────────────────────────

def seasonal_naive(train, test, lag=7):
    """P_hat(t) = P(t - lag). Simple but surprisingly competitive for wheat."""
    preds = test["price_modal"].shift(lag)
    # fill first `lag` days from training tail
    tail = train["price_modal"].iloc[-lag:].values
    preds.iloc[:lag] = tail
    return preds.values


# ── 2-3. SARIMA variants ──────────────────────────────

def fit_sarima(train_prices, order, seasonal_order, steps):
    """Fit SARIMA and forecast `steps` ahead."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(train_prices, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    fc = fitted.forecast(steps=steps)
    return fc.values


def sarima_aic_select(train_prices, steps, max_p=3, max_q=3):
    """Select SARIMA order by AIC. Search over a small grid."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best_aic = np.inf
    best_order = (1, 1, 1)
    # only search non-seasonal order; fix seasonal at (1,1,1,7)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                m = SARIMAX(train_prices, order=(p, 1, q),
                            seasonal_order=(1, 1, 1, 7),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                res = m.fit(disp=False, maxiter=150)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, 1, q)
            except:
                continue

    print(f"    AIC-selected order: {best_order}, AIC={best_aic:.1f}")
    return fit_sarima(train_prices, best_order, (1, 1, 1, 7), steps)


# ── 4. Exponential Smoothing ──────────────────────────

def fit_ets(train_prices, steps):
    """ETS with damped trend. Handles the seasonal component via Holt-Winters."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    try:
        model = ExponentialSmoothing(train_prices, trend="add", damped_trend=True,
                                      seasonal="add", seasonal_periods=7)
        fitted = model.fit(optimized=True, use_brute=True)
        return fitted.forecast(steps).values
    except:
        # fallback: no seasonality
        model = ExponentialSmoothing(train_prices, trend="add", damped_trend=True)
        fitted = model.fit(optimized=True)
        return fitted.forecast(steps).values


# ── 5. Random Forest ──────────────────────────────────

def train_random_forest(X_train, y_train, X_test):
    rf = RandomForestRegressor(**config.RF_PARAMS)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return preds, rf


# ── 6. Gradient Boosting ──────────────────────────────

def train_gradient_boosting(X_train, y_train, X_test):
    gb = GradientBoostingRegressor(**config.GB_PARAMS)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_test)
    return preds, gb


# ── 7. LSTM ───────────────────────────────────────────

def build_lstm_model(n_features, params):
    """Build a 2-layer LSTM. Requires tensorflow."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    tf.random.set_seed(42)
    model = Sequential([
        LSTM(params["hidden_units"], return_sequences=True,
             input_shape=(params["lookback"], n_features)),
        Dropout(0.2),
        LSTM(params["hidden_units"], return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_lstm_data(X, y, lookback):
    """Reshape into (samples, lookback, features) for LSTM input."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def train_lstm(X_train, y_train, X_test, y_test_for_shape):
    """Full LSTM training with early stopping."""
    from tensorflow.keras.callbacks import EarlyStopping

    params = config.LSTM_PARAMS
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_tr_scaled = scaler_X.fit_transform(X_train)
    y_tr_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_te_scaled = scaler_X.transform(X_test)

    X_tr_seq, y_tr_seq = prepare_lstm_data(X_tr_scaled, y_tr_scaled, params["lookback"])

    model = build_lstm_model(X_train.shape[1], params)
    es = EarlyStopping(monitor="val_loss", patience=params["patience"],
                       restore_best_weights=True)
    model.fit(X_tr_seq, y_tr_seq, epochs=params["epochs"],
              batch_size=params["batch_size"], validation_split=0.15,
              callbacks=[es], verbose=0)

    # predict on test
    # need to prepend last `lookback` rows of training for context
    full_X = np.vstack([X_tr_scaled[-params["lookback"]:], X_te_scaled])
    X_te_seq, _ = prepare_lstm_data(full_X, np.zeros(len(full_X)), params["lookback"])
    preds_scaled = model.predict(X_te_seq, verbose=0).flatten()
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    # trim to match test length
    preds = preds[:len(X_test)]
    return preds, model


# ── run all models on one crop ────────────────────────

def run_all_models(crop, df):
    """Train all 7 models and return results table."""
    print(f"\n{'='*50}")
    print(f"  {crop.upper()} — Model A (policy-conditioned)")
    print(f"{'='*50}")

    feat_cols = get_feature_cols(df, policy_neutral=False)
    train_df, test_df = train_test_split_temporal(df, config.TRAIN_END, config.TEST_START)

    # also get val set for hyperparameter stuff (not used for final eval)
    val_df = df[(df["date"] > config.TRAIN_END) & (df["date"] <= config.VAL_END)]

    X_train = train_df[feat_cols].values
    y_train = train_df["price_target"].values
    X_test = test_df[feat_cols].values
    y_test = test_df["price_target"].values
    test_prices = test_df["price_modal"].values

    # handle NaN in features — fill with column means
    col_means = np.nanmean(X_train, axis=0)
    for j in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, j]), j] = col_means[j]
        X_test[np.isnan(X_test[:, j]), j] = col_means[j]

    y_train = np.nan_to_num(y_train, nan=np.nanmean(y_train))
    y_test = np.nan_to_num(y_test, nan=np.nanmean(y_test))

    n_test = len(y_test)
    results = []
    predictions = {}

    # --- 1. Seasonal Naive ---
    print("  [1/7] Seasonal Naive...")
    pred_naive = seasonal_naive(train_df, test_df, lag=7)[:n_test]
    results.append(eval_metrics(y_test, pred_naive, "Seasonal Naive"))
    predictions["naive"] = pred_naive

    # --- 2. SARIMA (AIC) ---
    print("  [2/7] SARIMA (AIC-selected)...")
    try:
        train_series = train_df.groupby("date")["price_modal"].mean()
        pred_sarima_aic = sarima_aic_select(train_series.values[-365*2:], n_test)
        if len(pred_sarima_aic) < n_test:
            pred_sarima_aic = np.pad(pred_sarima_aic, (0, n_test - len(pred_sarima_aic)),
                                      mode="edge")
        pred_sarima_aic = pred_sarima_aic[:n_test]
        results.append(eval_metrics(y_test, pred_sarima_aic, "SARIMA (AIC)"))
        predictions["sarima_aic"] = pred_sarima_aic
    except Exception as e:
        print(f"    SARIMA AIC failed: {e}")
        results.append({"model": "SARIMA (AIC)", "rmse": np.nan, "mae": np.nan,
                        "mape": np.nan, "r2": np.nan, "dir_acc": np.nan})

    # --- 3. SARIMA fixed order ---
    print("  [3/7] SARIMA(1,1,1)(1,1,1)_7...")
    try:
        train_series = train_df.groupby("date")["price_modal"].mean()
        pred_sarima_fix = fit_sarima(train_series.values[-365*2:],
                                     config.SARIMA_FIXED_ORDER,
                                     config.SARIMA_SEASONAL_ORDER, n_test)
        pred_sarima_fix = pred_sarima_fix[:n_test]
        results.append(eval_metrics(y_test, pred_sarima_fix, "SARIMA(1,1,1)(1,1,1)_7"))
        predictions["sarima_fix"] = pred_sarima_fix
    except Exception as e:
        print(f"    SARIMA fixed failed: {e}")
        results.append({"model": "SARIMA(1,1,1)(1,1,1)_7", "rmse": np.nan,
                        "mae": np.nan, "mape": np.nan, "r2": np.nan, "dir_acc": np.nan})

    # --- 4. ETS ---
    print("  [4/7] Exponential Smoothing (damped)...")
    try:
        train_series = train_df.groupby("date")["price_modal"].mean()
        pred_ets = fit_ets(train_series.values[-365*2:], n_test)
        pred_ets = pred_ets[:n_test]
        results.append(eval_metrics(y_test, pred_ets, "ETS"))
        predictions["ets"] = pred_ets
    except Exception as e:
        print(f"    ETS failed: {e}")
        results.append({"model": "ETS", "rmse": np.nan, "mae": np.nan,
                        "mape": np.nan, "r2": np.nan, "dir_acc": np.nan})

    # --- 5. Random Forest ---
    print("  [5/7] Random Forest (200 trees, depth 20)...")
    pred_rf, rf_model = train_random_forest(X_train, y_train, X_test)
    results.append(eval_metrics(y_test, pred_rf, "Random Forest"))
    predictions["rf"] = pred_rf

    # save the RF model
    joblib.dump(rf_model, config.RESULTS / f"rf_model_a_{crop}.pkl")

    # feature importance
    fi = pd.DataFrame({
        "feature": feat_cols,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(config.RESULTS / f"feature_importance_a_{crop}.csv", index=False)
    print(f"    Top 3 features: {', '.join(fi.head(3)['feature'].tolist())}")
    print(f"    Top feature explains {fi.iloc[0]['importance']*100:.1f}% of variance")

    # --- 6. Gradient Boosting ---
    print("  [6/7] Gradient Boosting (200 est, depth 8, lr=0.1)...")
    pred_gb, gb_model = train_gradient_boosting(X_train, y_train, X_test)
    results.append(eval_metrics(y_test, pred_gb, "Gradient Boosting"))
    predictions["gb"] = pred_gb
    joblib.dump(gb_model, config.RESULTS / f"gb_model_a_{crop}.pkl")

    # --- 7. LSTM ---
    print("  [7/7] LSTM (2 layers, 128 units, lookback=30)...")
    try:
        pred_lstm, lstm_model = train_lstm(X_train, y_train, X_test, y_test)
        pred_lstm = pred_lstm[:n_test]
        # pad if needed (lookback can eat a few rows)
        if len(pred_lstm) < n_test:
            pred_lstm = np.pad(pred_lstm, (n_test - len(pred_lstm), 0), mode="edge")
        results.append(eval_metrics(y_test, pred_lstm, "LSTM"))
        predictions["lstm"] = pred_lstm
    except Exception as e:
        print(f"    LSTM failed: {e}")
        results.append({"model": "LSTM", "rmse": np.nan, "mae": np.nan,
                        "mape": np.nan, "r2": np.nan, "dir_acc": np.nan})

    # results table
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("mape")
    print_table(res_df, f"Model A Results — {crop.upper()}")

    res_df["crop"] = crop
    res_df.to_csv(config.RESULTS / f"model_a_results_{crop}.csv", index=False)

    # save predictions for downstream use
    pred_df = pd.DataFrame({"date": test_df["date"].values[:n_test],
                             "actual": y_test})
    for name, vals in predictions.items():
        pred_df[name] = vals[:n_test] if len(vals) >= n_test else \
                        np.pad(vals, (0, n_test - len(vals)), mode="edge")
    pred_df.to_csv(config.RESULTS / f"predictions_a_{crop}.csv", index=False)

    return res_df, predictions, y_test


# ── rolling-origin cross-validation ───────────────────

def rolling_cv(crop, df):
    """3-fold expanding-window CV with DM tests vs naive."""
    print(f"\n  Rolling CV for {crop}...")
    feat_cols = get_feature_cols(df, policy_neutral=False)

    fold_results = {m: [] for m in ["Seasonal Naive", "Random Forest", "Gradient Boosting"]}

    for i, fold in enumerate(config.CV_FOLDS):
        print(f"    Fold {i+1}: train to {fold['train_end']}, test {fold['test_start']}–{fold['test_end']}")
        train = df[df["date"] <= fold["train_end"]]
        test = df[(df["date"] >= fold["test_start"]) & (df["date"] <= fold["test_end"])]

        if len(test) == 0:
            continue

        X_tr = train[feat_cols].values
        y_tr = train["price_target"].values
        X_te = test[feat_cols].values
        y_te = test["price_target"].values

        # fill NaN
        means = np.nanmean(X_tr, axis=0)
        for j in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, j]), j] = means[j]
            X_te[np.isnan(X_te[:, j]), j] = means[j]
        y_tr = np.nan_to_num(y_tr, nan=np.nanmean(y_tr))
        y_te = np.nan_to_num(y_te, nan=np.nanmean(y_te))

        n = len(y_te)

        # naive
        naive_preds = seasonal_naive(train, test, lag=7)[:n]
        fold_results["Seasonal Naive"].append(eval_metrics(y_te, naive_preds, f"fold{i}"))

        # RF
        rf_preds, _ = train_random_forest(X_tr, y_tr, X_te)
        fold_results["Random Forest"].append(eval_metrics(y_te, rf_preds[:n], f"fold{i}"))

        # GB
        gb_preds, _ = train_gradient_boosting(X_tr, y_tr, X_te)
        fold_results["Gradient Boosting"].append(eval_metrics(y_te, gb_preds[:n], f"fold{i}"))

    # aggregate
    cv_summary = []
    for model_name, fold_list in fold_results.items():
        mapes = [f["mape"] for f in fold_list if not np.isnan(f["mape"])]
        if mapes:
            cv_summary.append({
                "model": model_name,
                "mape_mean": np.mean(mapes),
                "mape_sd": np.std(mapes),
            })

    cv_df = pd.DataFrame(cv_summary)
    cv_df["crop"] = crop
    print_table(cv_df, f"Rolling CV — {crop}")
    cv_df.to_csv(config.RESULTS / f"rolling_cv_{crop}.csv", index=False)
    return cv_df


# ── multi-horizon evaluation ──────────────────────────

def multihorizon_eval(crop, predictions, y_test):
    """Evaluate at 1, 3, and 6 month horizons."""
    rows = []
    for model_name, preds in predictions.items():
        for hz_name, hz_days in config.HORIZONS.items():
            n = min(hz_days, len(y_test), len(preds))
            m = eval_metrics(y_test[:n], preds[:n], model_name)
            m["horizon"] = hz_name
            rows.append(m)

    hz_df = pd.DataFrame(rows)
    hz_df["crop"] = crop
    hz_df.to_csv(config.RESULTS / f"multihorizon_{crop}.csv", index=False)
    return hz_df


# ── main ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MSP Welfare Optimization — Forecasting (Model A)")
    print("=" * 60)

    all_results = []

    for crop in config.COMMODITIES:
        df = load_processed_data(crop, config.DATA_PROC)
        print(f"\nLoaded {crop}: {len(df)} rows, "
              f"{df['date'].min().date()} to {df['date'].max().date()}")

        # main evaluation
        res_df, predictions, y_test = run_all_models(crop, df)
        all_results.append(res_df)

        # rolling CV
        rolling_cv(crop, df)

        # multi-horizon
        hz = multihorizon_eval(crop, predictions, y_test)

    # combined table
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(config.RESULTS / "model_a_all.csv", index=False)

    print("\n" + "=" * 60)
    print("  Model A evaluation complete. Results in:", config.RESULTS)
    print("=" * 60)
