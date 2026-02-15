"""
06_weather_panel.py — Weather-price panel regressions

Estimates:
    P_it = mu_i + theta_m + phi*t + sum_k(beta_k * W_{it-l,k}) + sum_j(rho_j * P_{it-j}) + u_it

Market fixed effects, month fixed effects, linear trend, lagged weather,
autoregressive terms. Clustered SEs at market level.

Usage:
    python src/06_weather_panel.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import load_processed_data, print_table

warnings.filterwarnings("ignore")


def prepare_panel(df, crop):
    """
    Prepare the panel for regression. Need market and time identifiers,
    weather variables, lagged prices, and dummies.
    """
    panel = df.copy()

    # make sure we have entity and time indices
    if "market" not in panel.columns:
        # if aggregated, we can't do panel FE. fall back to OLS.
        print("  WARNING: no market column, using pooled OLS instead of panel FE")
        panel["market"] = "aggregate"

    # create market numeric ID for FE
    panel["market_id"] = pd.Categorical(panel["market"]).codes

    # time trend (days since start)
    panel["time_trend"] = (panel["date"] - panel["date"].min()).dt.days

    # month dummies
    panel["month"] = panel["date"].dt.month
    for m in range(2, 13):
        panel[f"month_{m}"] = (panel["month"] == m).astype(int)

    # lagged weather (7-day lag for causal identification)
    weather_vars = []
    for wv in ["rainfall_7d_cum", "temp_anomaly", "extreme_heat_days", "drought_indicator"]:
        if wv in panel.columns:
            lagged_col = f"{wv}_lag7"
            panel[lagged_col] = panel.groupby("market")[wv].shift(7)
            weather_vars.append(lagged_col)

    # autoregressive terms
    if "price_lag1" not in panel.columns:
        panel["price_lag1"] = panel.groupby("market")["price_modal"].shift(1)
    if "price_lag7" not in panel.columns:
        panel["price_lag7"] = panel.groupby("market")["price_modal"].shift(7)

    panel = panel.dropna(subset=weather_vars + ["price_lag1", "price_lag7"])

    return panel, weather_vars


def run_panel_regression(df, crop, weather_vars):
    """
    Panel regression with market fixed effects.
    Uses linearmodels if available, falls back to statsmodels with dummies.
    """
    # dependent variable
    y_col = "price_modal"

    # regressors
    month_dummies = [c for c in df.columns if c.startswith("month_")]
    x_cols = weather_vars + ["price_lag1", "price_lag7", "time_trend"] + month_dummies

    # drop any cols that don't exist
    x_cols = [c for c in x_cols if c in df.columns]

    try:
        from linearmodels.panel import PanelOLS
        # set multi-index for panel
        panel_df = df.set_index(["market", "date"])

        y = panel_df[y_col]
        X = panel_df[x_cols]

        model = PanelOLS(y, X, entity_effects=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)

        print(f"\n  Panel FE regression — {crop.upper()}")
        print(f"  Observations: {result.nobs:,}")
        print(f"  R² (within): {result.rsquared_within:.3f}")
        print(f"  Adj R²: {result.rsquared:.3f}")
        print(f"\n  Weather coefficients (market-clustered SE):")

        coef_rows = []
        for wv in weather_vars:
            if wv in result.params.index:
                coef = result.params[wv]
                se = result.std_errors[wv]
                pval = result.pvalues[wv]
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"    {wv:30s}: {coef:8.1f}{stars:4s} ({se:.1f})")
                coef_rows.append({
                    "variable": wv, "coefficient": coef,
                    "std_error": se, "p_value": pval,
                })

        return pd.DataFrame(coef_rows), result

    except ImportError:
        print("  linearmodels not installed, falling back to statsmodels OLS with dummies")
        return _run_ols_with_dummies(df, y_col, x_cols, weather_vars, crop)


def _run_ols_with_dummies(df, y_col, x_cols, weather_vars, crop):
    """Fallback: OLS with market dummies instead of proper panel FE."""
    import statsmodels.api as sm

    # add market dummies
    market_dummies = pd.get_dummies(df["market"], prefix="mkt", drop_first=True)
    X = pd.concat([df[x_cols], market_dummies], axis=1)
    X = sm.add_constant(X)
    y = df[y_col]

    model = sm.OLS(y, X, missing="drop")
    result = model.fit(cov_type="cluster", cov_kwds={"groups": df["market"]})

    print(f"\n  OLS with market dummies — {crop.upper()}")
    print(f"  Observations: {int(result.nobs):,}")
    print(f"  Adj R²: {result.rsquared_adj:.3f}")
    print(f"\n  Weather coefficients:")

    coef_rows = []
    for wv in weather_vars:
        if wv in result.params.index:
            coef = result.params[wv]
            se = result.bse[wv]
            pval = result.pvalues[wv]
            stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"    {wv:30s}: {coef:8.1f}{stars:4s} ({se:.1f})")
            coef_rows.append({
                "variable": wv, "coefficient": coef,
                "std_error": se, "p_value": pval,
            })

    return pd.DataFrame(coef_rows), result


if __name__ == "__main__":
    print("=" * 60)
    print("  MSP Welfare Optimization — Weather Panel Regressions")
    print("=" * 60)

    all_coefs = []

    for crop in config.COMMODITIES:
        df = load_processed_data(crop, config.DATA_PROC)
        panel, wvars = prepare_panel(df, crop)

        if not wvars:
            print(f"\n  No weather variables found for {crop}. Skipping.")
            continue

        coefs, result = run_panel_regression(panel, crop, wvars)
        coefs["crop"] = crop
        all_coefs.append(coefs)

    if all_coefs:
        combined = pd.concat(all_coefs, ignore_index=True)
        combined.to_csv(config.RESULTS / "weather_panel_results.csv", index=False)
        print_table(combined, "Weather Panel Results (all crops)")

    print("\n" + "=" * 60)
    print("  Panel regressions complete.")
    print("=" * 60)
