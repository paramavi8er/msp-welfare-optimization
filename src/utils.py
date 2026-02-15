"""
utils.py — Shared helper functions

Mostly metric calculations and data I/O wrappers that get called
from multiple scripts. Nothing fancy.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_mape(y_true, y_pred):
    """Mean Absolute Percentage Error. Handles zeros by adding epsilon."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true, y_pred):
    """% of correctly predicted day-over-day direction changes."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2:
        return np.nan
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return np.mean(actual_dir == pred_dir) * 100


def eval_metrics(y_true, y_pred, label=""):
    """Compute all 5 metrics used in the paper. Returns dict."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    results = {
        "model": label,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": calc_mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
    }
    return results


def eval_at_horizons(y_true, y_pred, horizons_dict):
    """
    Evaluate metrics at multiple horizons.
    horizons_dict: {"1-month": 30, "3-month": 90, ...}
    """
    rows = []
    for name, days in horizons_dict.items():
        n = min(days, len(y_true))
        m = eval_metrics(y_true[:n], y_pred[:n], label=name)
        m["horizon"] = name
        rows.append(m)
    return pd.DataFrame(rows)


def diebold_mariano_test(e1, e2, h=1, loss="squared"):
    """
    Diebold-Mariano test for equal predictive accuracy.
    e1, e2: forecast errors from model 1 and model 2.
    h: forecast horizon (for HAC bandwidth).
    loss: "squared" or "absolute".
    Returns (DM statistic, p-value).
    """
    from scipy import stats

    e1, e2 = np.array(e1), np.array(e2)
    if loss == "squared":
        d = e1**2 - e2**2
    else:
        d = np.abs(e1) - np.abs(e2)

    T = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC variance estimator
    bandwidth = int(np.floor(T ** (1/3)))
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, bandwidth + 1):
        w_k = 1 - k / (bandwidth + 1)  # Bartlett kernel
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += 2 * w_k * gamma_k

    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_bar / np.sqrt(var_d)
    p_val = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_val


def load_processed_data(crop, data_dir):
    """Load the processed daily panel data for a given crop."""
    fpath = data_dir / f"{crop}_daily_panel.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Processed data not found at {fpath}. Run 02_preprocessing.py first.")
    df = pd.read_csv(fpath, parse_dates=["date"])
    return df


def train_test_split_temporal(df, train_end, test_start=None):
    """Split by date. Returns (train, test) DataFrames."""
    train = df[df["date"] <= train_end].copy()
    if test_start:
        test = df[df["date"] >= test_start].copy()
    else:
        test = df[df["date"] > train_end].copy()
    return train, test


def get_feature_cols(df, policy_neutral=False):
    """
    Return list of feature column names.
    If policy_neutral=True, exclude MSP-derived features (Model B).
    """
    exclude = ["date", "market", "state", "price_modal", "price_target"]
    if policy_neutral:
        exclude += ["msp_current", "price_to_msp_ratio", "msp_change_pct",
                     "msp_current_lag7", "price_to_msp_ratio_lag7", "msp_change_pct_lag7"]

    return [c for c in df.columns if c not in exclude]


def print_table(df, title=""):
    """Pretty print a dataframe as a formatted table. For console output."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(df.to_string(index=False))
    print()
