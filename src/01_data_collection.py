"""
01_data_collection.py — Download raw data from public sources

Agmarknet data requires semi-manual download (CAPTCHA on the site).
This script provides helper functions and documents the process.
IMD data can be partially automated via the POWER API as backup.

Usage:
    python src/01_data_collection.py
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ── Agmarknet helper ───────────────────────────────────
# NOTE: Agmarknet (agmarknet.gov.in) uses session-based forms and CAPTCHAs.
# Fully automated scraping is unreliable. The recommended approach is:
#   1. Use the "Daily Report" interface manually for each market/commodity/year
#   2. Export as Excel/CSV
#   3. Save to data/raw/agmarknet/
#
# The function below attempts the direct URL approach which works intermittently.

AGMARKNET_BASE = "https://agmarknet.gov.in/SearchCmmMkt.aspx"

def build_agmarknet_params(commodity, state, market, date_from, date_to):
    """Build query params for Agmarknet daily price report."""
    # commodity codes — these are Agmarknet's internal IDs
    # You can find them by inspecting the dropdown on the site
    commodity_codes = {
        "wheat": "1",
        "rice": "13",
    }
    return {
        "Ession_Id": "",
        "Ession_val": "",
        "Sess_Id": "",
        "Sess_Val": "",
        "Sess_ValAcc": "",
        "Sess_Valacc": "",
        "Sess_Valacc1": "",
        "Sess_Valacc2": "",
        "Commodity": commodity_codes.get(commodity, "1"),
        "State": state,
        "Market": market,
        "DateFrom": date_from,
        "DateTo": date_to,
    }


def download_agmarknet_manual_instructions():
    """Print step-by-step instructions for manual download."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  AGMARKNET MANUAL DOWNLOAD INSTRUCTIONS                  ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  1. Open https://agmarknet.gov.in                       ║
    ║  2. Click "Commodity-wise Daily State level info"        ║
    ║  3. For EACH combination of:                            ║
    ║     - Commodity: Wheat, Rice                            ║
    ║     - State: Punjab, Haryana, UP, MP, West Bengal       ║
    ║     - Market: See list below                            ║
    ║     - Date range: 01-Jan-2020 to 31-Dec-2024           ║
    ║  4. Click "Submit" and then "Export to Excel"           ║
    ║  5. Save as: {State}_{Market}_{commodity}.csv           ║
    ║     in data/raw/agmarknet/                              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    print("Markets to download:")
    for state, markets in config.MARKETS.items():
        for m in markets:
            print(f"  {state}: {m}")
    print(f"\nTotal files needed: {sum(len(v) for v in config.MARKETS.values()) * 2} "
          f"(10 markets × 2 commodities)")


# ── IMD Weather data via NASA POWER API (backup) ──────
# Primary source is IMD directly, but NASA POWER provides good coverage
# of daily temperature and precipitation for Indian districts.

POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

# approximate centroids for each state (for POWER API queries)
STATE_COORDS = {
    "Punjab":         {"lat": 31.15, "lon": 75.34},
    "Haryana":        {"lat": 29.06, "lon": 76.09},
    "Uttar Pradesh":  {"lat": 26.85, "lon": 80.91},
    "Madhya Pradesh": {"lat": 23.47, "lon": 77.95},
    "West Bengal":    {"lat": 22.99, "lon": 87.75},
}

def download_nasa_power(state, start="20200101", end="20241231"):
    """
    Download daily weather from NASA POWER API for a state centroid.
    Returns pandas DataFrame.

    This is a backup source — IMD data is preferred for the paper.
    """
    coords = STATE_COORDS[state]
    params = {
        "start": start,
        "end": end,
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "community": "ag",
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M",
        "format": "json",
    }
    print(f"  Downloading POWER data for {state}...", end=" ", flush=True)

    try:
        resp = requests.get(POWER_API, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        records = data["properties"]["parameter"]
        df = pd.DataFrame(records)
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.rename(columns={
            "T2M_MAX": "max_temp",
            "T2M_MIN": "min_temp",
            "PRECTOTCORR": "rainfall_daily",
            "RH2M": "humidity",
        })
        df["state"] = state
        print("OK")
        return df

    except Exception as e:
        print(f"FAILED: {e}")
        return pd.DataFrame()


def download_all_weather():
    """Download weather for all states. Save to data/raw/imd/ (as backup source)."""
    outdir = config.DATA_RAW / "imd"
    outdir.mkdir(parents=True, exist_ok=True)

    all_weather = []
    for state in config.MARKETS.keys():
        df = download_nasa_power(state)
        if not df.empty:
            fpath = outdir / f"{state.replace(' ', '')}_weather_power.csv"
            df.to_csv(fpath, index=False)
            all_weather.append(df)
        time.sleep(2)  # be polite to the API

    if all_weather:
        combined = pd.concat(all_weather, ignore_index=True)
        combined.to_csv(outdir / "all_states_weather.csv", index=False)
        print(f"\nWeather data saved: {len(combined)} records across {len(all_weather)} states")
    else:
        print("\nWARNING: No weather data downloaded successfully")


# ── MSP and procurement data (already in repo) ────────
def load_msp_procurement():
    """Load the hand-compiled MSP/procurement CSV."""
    fpath = config.ROOT / "data" / "msp_procurement.csv"
    df = pd.read_csv(fpath)
    print(f"Loaded MSP/procurement data: {len(df)} records")
    return df


# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MSP Welfare Optimization — Data Collection")
    print("=" * 60)

    # MSP data is already in the repo
    msp = load_msp_procurement()
    print(msp.to_string(index=False))

    # weather — try the NASA POWER backup
    print("\n--- Downloading weather data (NASA POWER backup) ---")
    download_all_weather()

    # Agmarknet — needs manual download
    print("\n--- Agmarknet price data ---")
    download_agmarknet_manual_instructions()

    print("\nDone. After manually downloading Agmarknet data,")
    print("run: python src/02_preprocessing.py")
