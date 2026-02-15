"""
07_monte_carlo.py — Dual Monte Carlo uncertainty decomposition

MC-1: Elasticity uncertainty — hold forecast fixed, draw elasticities
MC-2: Forecast uncertainty — hold elasticities fixed, perturb forecast

N=1,000 draws each. Reports median optimal MSP, 90% CI, SD.

Usage:
    python src/07_monte_carlo.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import print_table
import importlib
_welfare = importlib.import_module("src.05_welfare_optimization")
find_optimal_msp = _welfare.find_optimal_msp

warnings.filterwarnings("ignore")


def load_forecast_prices():
    """Load Model B forecast prices. Fall back to paper values."""
    try:
        fc = pd.read_csv(config.RESULTS / "model_b_forecasts.csv")
        return dict(zip(fc["crop"], fc["forecast_price"]))
    except FileNotFoundError:
        print("  Using paper reference values (run 04 first for actual)")
        return {"wheat": 2350, "rice": 2967}


def load_model_b_rmse():
    """Load Model B RF RMSE for forecast perturbation scaling."""
    rmse = {}
    for crop in config.COMMODITIES:
        try:
            res = pd.read_csv(config.RESULTS / f"model_b_results_{crop}.csv")
            rf_row = res[res["model"] == "Random Forest"]
            if not rf_row.empty:
                rmse[crop] = rf_row["rmse"].values[0]
            else:
                # fallback to paper values
                rmse[crop] = {"wheat": 241.62, "rice": 189.43}[crop]
        except FileNotFoundError:
            rmse[crop] = {"wheat": 241.62, "rice": 189.43}[crop]
    return rmse


# ── MC-1: Elasticity uncertainty ──────────────────────

def mc_elasticity(crop, p_hat, n_draws, weights, seed):
    """
    Hold forecast fixed at p_hat. Draw elasticities from their
    calibrated distributions (normal, truncated to positive supply
    and negative demand).
    """
    rng = np.random.RandomState(seed)
    elast = config.ELASTICITIES[crop]

    optimal_msps = []

    for i in range(n_draws):
        # draw supply elasticity (truncated > 0)
        eps_s = -1
        while eps_s <= 0:
            eps_s = rng.normal(elast["supply"], elast["supply_se"])

        # draw demand elasticity (truncated < 0)
        eps_d = 1
        while eps_d >= 0:
            eps_d = rng.normal(elast["demand"], elast["demand_se"])

        # temporarily override config for this draw
        orig_s = config.ELASTICITIES[crop]["supply"]
        orig_d = config.ELASTICITIES[crop]["demand"]
        config.ELASTICITIES[crop]["supply"] = eps_s
        config.ELASTICITIES[crop]["demand"] = eps_d

        m_star, _ = find_optimal_msp(p_hat, crop, weights)
        optimal_msps.append(m_star)

        # restore
        config.ELASTICITIES[crop]["supply"] = orig_s
        config.ELASTICITIES[crop]["demand"] = orig_d

    return np.array(optimal_msps)


# ── MC-2: Forecast uncertainty ────────────────────────

def mc_forecast(crop, p_hat, rmse, n_draws, weights, seed):
    """
    Hold elasticities at central values. Perturb the forecast price
    with SD = 30% of the Model B RMSE.
    """
    rng = np.random.RandomState(seed + 500)
    perturbation_sd = config.FORECAST_RMSE_FRACTION * rmse

    optimal_msps = []

    for i in range(n_draws):
        p_perturbed = rng.normal(p_hat, perturbation_sd)
        # keep it positive and reasonable
        p_perturbed = max(p_perturbed, p_hat * 0.5)

        m_star, _ = find_optimal_msp(p_perturbed, crop, weights)
        optimal_msps.append(m_star)

    return np.array(optimal_msps)


# ── main ──────────────────────────────────────────────

def run_dual_mc():
    forecast_prices = load_forecast_prices()
    rmse_vals = load_model_b_rmse()
    weights = config.SCENARIOS["equal_weights"]
    n = config.MC_N_DRAWS
    seed = config.MC_SEED

    print(f"\n  Monte Carlo parameters:")
    print(f"    N draws: {n}")
    print(f"    Seed: {seed}")
    print(f"    Forecast perturbation: {config.FORECAST_RMSE_FRACTION*100:.0f}% of RMSE")

    rows = []
    all_draws = {}

    for crop in config.COMMODITIES:
        p_hat = forecast_prices[crop]
        rmse = rmse_vals[crop]

        print(f"\n  {crop.upper()} (forecast={p_hat:.0f}, RMSE={rmse:.1f}):")

        # MC-1: elasticity
        print(f"    Running MC-1 (elasticity, N={n})...", end=" ", flush=True)
        mc1 = mc_elasticity(crop, p_hat, n, weights, seed)
        print("done")

        mc1_median = np.median(mc1)
        mc1_ci = np.percentile(mc1, [5, 95])
        mc1_sd = np.std(mc1)

        rows.append({
            "source": "Elasticity (MC-1)",
            "crop": crop,
            "median_msp": round(mc1_median),
            "ci_lower": round(mc1_ci[0]),
            "ci_upper": round(mc1_ci[1]),
            "ci_width": round(mc1_ci[1] - mc1_ci[0]),
            "sd": round(mc1_sd, 1),
        })
        all_draws[f"{crop}_mc1"] = mc1

        print(f"    MC-1: median={mc1_median:.0f}, 90% CI=[{mc1_ci[0]:.0f}, {mc1_ci[1]:.0f}], "
              f"SD={mc1_sd:.1f}")

        # MC-2: forecast
        print(f"    Running MC-2 (forecast, N={n})...", end=" ", flush=True)
        mc2 = mc_forecast(crop, p_hat, rmse, n, weights, seed)
        print("done")

        mc2_median = np.median(mc2)
        mc2_ci = np.percentile(mc2, [5, 95])
        mc2_sd = np.std(mc2)

        rows.append({
            "source": "Forecast (MC-2)",
            "crop": crop,
            "median_msp": round(mc2_median),
            "ci_lower": round(mc2_ci[0]),
            "ci_upper": round(mc2_ci[1]),
            "ci_width": round(mc2_ci[1] - mc2_ci[0]),
            "sd": round(mc2_sd, 1),
        })
        all_draws[f"{crop}_mc2"] = mc2

        print(f"    MC-2: median={mc2_median:.0f}, 90% CI=[{mc2_ci[0]:.0f}, {mc2_ci[1]:.0f}], "
              f"SD={mc2_sd:.1f}")

        # ratio
        ratio = (mc2_ci[1] - mc2_ci[0]) / max(mc1_ci[1] - mc1_ci[0], 1)
        print(f"    Forecast/Elasticity CI ratio: {ratio:.1f}:1")

    results_df = pd.DataFrame(rows)
    print_table(results_df, "Dual Monte Carlo Results")
    results_df.to_csv(config.RESULTS / "monte_carlo_results.csv", index=False)

    # save raw draws for figure generation
    draws_df = pd.DataFrame(all_draws)
    draws_df.to_csv(config.RESULTS / "mc_draws.csv", index=False)

    # combined confidence bands
    print("\n  Combined 90% confidence bands (MC-1 + MC-2):")
    for crop in config.COMMODITIES:
        mc1 = all_draws[f"{crop}_mc1"]
        mc2 = all_draws[f"{crop}_mc2"]
        # combine by adding the two sources of variation
        # (conservative: take wider band)
        combined_lower = min(np.percentile(mc1, 5), np.percentile(mc2, 5))
        combined_upper = max(np.percentile(mc1, 95), np.percentile(mc2, 95))
        current = config.CURRENT_MSP[crop]
        print(f"    {crop}: [{combined_lower:.0f}, {combined_upper:.0f}] "
              f"(current MSP = {current})")
        if current < combined_lower:
            print(f"      → Current MSP is BELOW the entire confidence band")
        elif current > combined_upper:
            print(f"      → Current MSP is ABOVE the entire confidence band")
        else:
            print(f"      → Current MSP falls WITHIN the confidence band")

    return results_df, all_draws


if __name__ == "__main__":
    print("=" * 60)
    print("  MSP Welfare Optimization — Monte Carlo Decomposition")
    print("=" * 60)

    results, draws = run_dual_mc()

    print("\n" + "=" * 60)
    print("  Monte Carlo analysis complete.")
    print("=" * 60)
