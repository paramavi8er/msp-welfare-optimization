"""
02_preprocessing.py — Data cleaning, imputation, and feature engineering

Takes raw Agmarknet + IMD files and produces the analysis-ready panel
datasets: wheat_daily_panel.csv and rice_daily_panel.csv

If raw data is not available, generates a synthetic sample dataset matching
the paper's descriptive statistics for pipeline testing.

Usage:
    python src/02_preprocessing.py
    python src/02_preprocessing.py --synthetic   # use synthetic data for testing
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)


# ── loading raw data ───────────────────────────────────

def load_agmarknet_raw(commodity):
    """
    Load and combine raw Agmarknet CSV files for a given commodity.
    Expected format: columns like Date, Market, State, Modal_Price, ...
    """
    raw_dir = config.DATA_RAW / "agmarknet"
    if not raw_dir.exists():
        return None

    files = list(raw_dir.glob(f"*_{commodity}*.csv")) + list(raw_dir.glob(f"*_{commodity}*.xlsx"))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            if f.suffix == ".xlsx":
                df = pd.read_excel(f)
            else:
                df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    # standardize column names (Agmarknet exports vary)
    col_map = {}
    for c in combined.columns:
        cl = c.lower().strip()
        if "date" in cl or "arrival" in cl:
            col_map[c] = "date"
        elif "modal" in cl and "price" in cl:
            col_map[c] = "price_modal"
        elif "market" in cl:
            col_map[c] = "market"
        elif "state" in cl:
            col_map[c] = "state"

    combined = combined.rename(columns=col_map)
    combined["date"] = pd.to_datetime(combined["date"], dayfirst=True, errors="coerce")
    combined["price_modal"] = pd.to_numeric(combined["price_modal"], errors="coerce")
    combined = combined.dropna(subset=["date", "price_modal"])

    print(f"  Loaded {len(combined)} raw obs for {commodity}")
    return combined


def load_weather_raw():
    """Load weather data (IMD or POWER backup)."""
    imd_dir = config.DATA_RAW / "imd"
    # try IMD first, then POWER backup
    for fname in ["all_states_weather.csv", "all_states_weather_power.csv"]:
        fpath = imd_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath, parse_dates=["date"])
            print(f"  Loaded weather from {fname}: {len(df)} records")
            return df
    return None


# ── synthetic data generator ───────────────────────────
# Used for pipeline testing when real data isn't available.
# Matches the paper's descriptive statistics (Table 1).

def generate_synthetic_panel(commodity, n_markets=10, seed=42):
    """
    Generate synthetic daily price panel matching paper descriptive stats.
    NOT real data — for pipeline testing only.
    """
    rng = np.random.RandomState(seed if commodity == "wheat" else seed + 1)

    stats = {
        "wheat": {"mean": 2291, "std": 362, "min": 1650, "max": 3420},
        "rice":  {"mean": 3082, "std": 382, "min": 2150, "max": 4280},
    }
    s = stats[commodity]

    dates = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    n_days = len(dates)

    market_names = []
    for state, mkts in config.MARKETS.items():
        for m in mkts:
            market_names.append((state, m))

    rows = []
    for state, market in market_names:
        # base series with autocorrelation (AR(1) process)
        prices = np.zeros(n_days)
        prices[0] = s["mean"]
        for t in range(1, n_days):
            # rho ~0.98 for daily agricultural prices (high persistence)
            innovation = rng.normal(0, s["std"] * 0.14)
            prices[t] = 0.98 * prices[t-1] + 0.02 * s["mean"] + innovation
            # add slight seasonal pattern
            month = dates[t].month
            if commodity == "wheat" and month in [3, 4, 5]:  # harvest season
                prices[t] -= rng.uniform(10, 30)
            elif commodity == "rice" and month in [10, 11, 12]:
                prices[t] -= rng.uniform(10, 25)

        prices = np.clip(prices, s["min"], s["max"])

        # introduce ~3% missing values
        missing_mask = rng.random(n_days) < 0.03
        prices_with_na = prices.copy()
        prices_with_na[missing_mask] = np.nan

        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "state": state,
                "market": market,
                "price_modal": prices_with_na[i],
            })

    df = pd.DataFrame(rows)
    df["commodity"] = commodity
    print(f"  Generated synthetic {commodity} panel: {len(df)} obs, "
          f"{df['price_modal'].isna().mean()*100:.1f}% missing")
    return df


def generate_synthetic_weather(seed=42):
    """Generate synthetic weather matching rough IMD characteristics."""
    rng = np.random.RandomState(seed + 100)
    dates = pd.date_range(config.START_DATE, config.END_DATE, freq="D")

    rows = []
    for state in config.MARKETS.keys():
        # temperature baseline by latitude (rough)
        base_temp = {"Punjab": 25, "Haryana": 26, "Uttar Pradesh": 27,
                     "Madhya Pradesh": 28, "West Bengal": 28}
        bt = base_temp[state]

        for i, d in enumerate(dates):
            month = d.month
            # seasonal temp variation
            seasonal = 10 * np.sin(2 * np.pi * (month - 4) / 12)
            max_t = bt + seasonal + rng.normal(0, 3)
            min_t = max_t - rng.uniform(6, 14)

            # rainfall — monsoon concentrated Jun-Sep
            if month in [6, 7, 8, 9]:
                rain = rng.exponential(8) if rng.random() > 0.3 else 0
            elif month in [12, 1, 2]:
                rain = rng.exponential(1) if rng.random() > 0.85 else 0
            else:
                rain = rng.exponential(2) if rng.random() > 0.7 else 0

            rows.append({
                "date": d, "state": state,
                "max_temp": round(max_t, 1),
                "min_temp": round(min_t, 1),
                "rainfall_daily": round(max(0, rain), 1),
                "humidity": round(np.clip(rng.normal(60, 15), 20, 98), 1),
            })

    df = pd.DataFrame(rows)
    print(f"  Generated synthetic weather: {len(df)} obs")
    return df


# ── feature engineering ────────────────────────────────

def impute_prices(df):
    """Linear interpolation within each market, then forward/backward fill."""
    df = df.sort_values(["market", "date"])
    df["price_modal"] = df.groupby("market")["price_modal"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    # any remaining NAs (edges) — fill with market mean
    df["price_modal"] = df.groupby("market")["price_modal"].transform(
        lambda x: x.fillna(x.mean())
    )
    return df


def add_price_features(df):
    """Price-derived features: lags, moving averages, returns."""
    df = df.sort_values(["market", "date"])
    grp = df.groupby("market")["price_modal"]

    # lagged prices
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"price_lag{lag}"] = grp.shift(lag)

    # moving averages
    for window in [7, 14, 30]:
        df[f"price_ma{window}"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    # volatility (rolling std)
    df["price_std7"] = grp.transform(lambda x: x.rolling(7, min_periods=2).std())
    df["price_std30"] = grp.transform(lambda x: x.rolling(30, min_periods=5).std())

    # returns
    df["price_ret1"] = grp.pct_change(1)
    df["price_ret7"] = grp.pct_change(7)

    # momentum
    df["price_momentum"] = df["price_modal"] - df["price_ma30"]

    return df


def add_temporal_features(df):
    """Calendar and seasonal features."""
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year

    # cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # is_harvest dummies (crop-specific, added later)
    return df


def add_weather_features(df, weather_df):
    """Merge weather and derive secondary weather variables."""
    # merge on state + date
    weather_cols = ["date", "state", "max_temp", "min_temp", "rainfall_daily", "humidity"]
    available = [c for c in weather_cols if c in weather_df.columns]
    w = weather_df[available].copy()

    df = df.merge(w, on=["date", "state"], how="left")

    # derived weather vars
    if "max_temp" in df.columns:
        df["temp_range"] = df["max_temp"] - df["min_temp"]
        # temperature anomaly: deviation from 30-day rolling mean
        df["temp_anomaly"] = df.groupby("state")["max_temp"].transform(
            lambda x: x - x.rolling(30, min_periods=7).mean()
        )
        # extreme heat day (>40°C)
        df["extreme_heat_days"] = (df["max_temp"] > 40).astype(int)

    if "rainfall_daily" in df.columns:
        # 7-day cumulative rainfall
        df["rainfall_7d_cum"] = df.groupby("state")["rainfall_daily"].transform(
            lambda x: x.rolling(7, min_periods=1).sum()
        )
        # drought indicator: rainfall below 25th pct of monthly dist
        monthly_p25 = df.groupby(["state", "month"])["rainfall_daily"].transform("quantile", 0.25)
        df["drought_indicator"] = (df["rainfall_daily"] < monthly_p25).astype(int)

    return df


def add_policy_features(df, commodity, msp_data):
    """
    Add MSP-derived features (for Model A).
    These get EXCLUDED in Model B (policy-neutral).
    """
    # get MSP for this commodity by year
    msp_by_year = msp_data[msp_data["crop"] == commodity].set_index("year")["msp_rs_per_quintal"]

    df["msp_current"] = df["year"].map(msp_by_year)
    # forward fill for years not in the table
    df["msp_current"] = df["msp_current"].ffill().bfill()

    df["price_to_msp_ratio"] = df["price_modal"] / df["msp_current"]
    df["msp_change_pct"] = df.groupby("market")["msp_current"].pct_change().fillna(0)

    # lagged versions
    for col in ["msp_current", "price_to_msp_ratio", "msp_change_pct"]:
        df[f"{col}_lag7"] = df.groupby("market")[col].shift(7)

    return df


def add_market_cross_features(df):
    """Cross-market features: spread from market mean, etc."""
    daily_mean = df.groupby("date")["price_modal"].transform("mean")
    df["price_spread_from_mean"] = df["price_modal"] - daily_mean
    df["price_relative"] = df["price_modal"] / daily_mean
    return df


def build_target(df, horizon=1):
    """Target variable: future price."""
    df["price_target"] = df.groupby("market")["price_modal"].shift(-horizon)
    return df


def aggregate_to_daily(df):
    """
    Aggregate from market-day level to daily mean across markets.
    This is what the paper's evaluation metrics are computed on.
    """
    # keep date-level features (they're the same across markets)
    agg_cols = [c for c in df.columns
                if c not in ["market", "state", "commodity"]]

    numeric_cols = df[agg_cols].select_dtypes(include=[np.number]).columns.tolist()
    # also keep date
    result = df.groupby("date")[numeric_cols].mean().reset_index()
    return result


# ── main pipeline ──────────────────────────────────────

def process_commodity(commodity, raw_df, weather_df, msp_data, use_synthetic=False):
    """Full preprocessing pipeline for one commodity."""
    print(f"\n--- Processing {commodity.upper()} ---")

    df = raw_df.copy()

    # 1. impute missing
    n_missing = df["price_modal"].isna().sum()
    print(f"  Missing values: {n_missing} ({n_missing/len(df)*100:.1f}%)")
    df = impute_prices(df)

    # 2. price features
    df = add_price_features(df)
    print(f"  Price features added: {sum(1 for c in df.columns if 'price_' in c)} columns")

    # 3. temporal features
    df = add_temporal_features(df)

    # 4. weather features
    if weather_df is not None:
        df = add_weather_features(df, weather_df)
        print(f"  Weather features merged")

    # 5. policy features
    df = add_policy_features(df, commodity, msp_data)
    print(f"  Policy features added (MSP, price-to-MSP ratio)")

    # 6. cross-market features
    df = add_market_cross_features(df)

    # 7. target
    df = build_target(df, horizon=1)

    # 8. drop rows with NaN in critical columns (from lagging)
    initial = len(df)
    df = df.dropna(subset=["price_target", "price_lag7", "price_ma7"])
    print(f"  Dropped {initial - len(df)} rows with NaN from lagging")

    # count features
    feat_cols = [c for c in df.columns
                 if c not in ["date", "market", "state", "commodity",
                              "price_modal", "price_target"]]
    print(f"  Total features: {len(feat_cols)}")

    # save market-level panel
    outpath = config.DATA_PROC / f"{commodity}_daily_panel.csv"
    df.to_csv(outpath, index=False)
    print(f"  Saved: {outpath} ({len(df)} rows)")

    # also save aggregated daily series
    agg = aggregate_to_daily(df)
    agg_path = config.DATA_PROC / f"{commodity}_daily_agg.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  Saved aggregated: {agg_path} ({len(agg)} rows)")

    # descriptive stats
    print(f"\n  Descriptive stats ({commodity}):")
    print(f"    Mean:  {df['price_modal'].mean():.0f}")
    print(f"    SD:    {df['price_modal'].std():.0f}")
    print(f"    CV:    {df['price_modal'].std()/df['price_modal'].mean()*100:.1f}%")
    print(f"    Min:   {df['price_modal'].min():.0f}")
    print(f"    Max:   {df['price_modal'].max():.0f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate and use synthetic data for pipeline testing")
    args = parser.parse_args()

    print("=" * 60)
    print("  MSP Welfare Optimization — Preprocessing")
    print("=" * 60)

    # load MSP data
    msp_data = pd.read_csv(config.ROOT / "data" / "msp_procurement.csv")

    for commodity in config.COMMODITIES:
        if args.synthetic:
            print(f"\n[SYNTHETIC MODE] Generating synthetic data for {commodity}")
            raw_df = generate_synthetic_panel(commodity)
            weather_df = generate_synthetic_weather()
        else:
            raw_df = load_agmarknet_raw(commodity)
            weather_df = load_weather_raw()

            if raw_df is None:
                print(f"\n  No raw Agmarknet data found for {commodity}.")
                print(f"  Falling back to synthetic data for testing.")
                print(f"  (Run with --synthetic flag explicitly, or download real data first)")
                raw_df = generate_synthetic_panel(commodity)

            if weather_df is None:
                print(f"  No weather data found, generating synthetic")
                weather_df = generate_synthetic_weather()

        process_commodity(commodity, raw_df, weather_df, msp_data,
                         use_synthetic=args.synthetic)

    print("\n" + "=" * 60)
    print("  Preprocessing complete.")
    print(f"  Output in: {config.DATA_PROC}")
    print("=" * 60)
