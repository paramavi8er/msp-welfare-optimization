"""
08_sensitivity.py — Robustness checks

1. Procurement endogeneity: relax exogenous procurement assumption.
   Model Q_proc(m) = Q0 + eta*(m - P_hat)/100, with eta in {0,0.5,1,2,3}.

2. Elasticity stress test: ±50% perturbation across 5 supply×demand combos.

Usage:
    python src/08_sensitivity.py
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
    try:
        fc = pd.read_csv(config.RESULTS / "model_b_forecasts.csv")
        return dict(zip(fc["crop"], fc["forecast_price"]))
    except FileNotFoundError:
        return {"wheat": 2350, "rice": 2967}


# ── 1. Procurement endogeneity ────────────────────────

def procurement_sensitivity(forecast_prices):
    """
    Test how optimal MSP changes when procurement responds to MSP-price gap.
    eta = MMT of additional procurement per Rs 100/qtl MSP increase.
    eta = 3.0 is deliberately extreme (implies Rs 100 increase draws 3 MMT more,
    roughly 10% of annual wheat procurement).
    """
    print("\n  Procurement Endogeneity Sensitivity")
    print("  " + "-" * 50)

    weights = config.SCENARIOS["equal_weights"]
    rows = []

    for crop in config.COMMODITIES:
        p_hat = forecast_prices[crop]
        baseline_msp = None

        for eta in config.PROCUREMENT_ETAS:
            m_star, w_star = find_optimal_msp(p_hat, crop, weights, eta=eta)

            if eta == 0:
                baseline_msp = m_star

            delta = round(m_star - baseline_msp) if baseline_msp else 0

            rows.append({
                "crop": crop,
                "eta": eta,
                "optimal_msp": round(m_star),
                "delta_from_baseline": delta,
                "welfare": round(w_star),
            })

            print(f"    {crop:6s} eta={eta:.1f}: m*={m_star:.0f} "
                  f"(Δ={delta:+d} from baseline)")

    results = pd.DataFrame(rows)
    results.to_csv(config.RESULTS / "sensitivity_procurement.csv", index=False)
    return results


# ── 2. Elasticity stress test ─────────────────────────

def elasticity_stress_test(forecast_prices):
    """
    Perturb supply and demand elasticities by ±50% across a 3×3 grid
    (actually 5 named scenarios from the paper).
    """
    print("\n  Elasticity Stress Test (±50%)")
    print("  " + "-" * 50)

    weights = config.SCENARIOS["equal_weights"]
    multipliers = config.ELASTICITY_MULTIPLIERS  # [0.5, 1.0, 1.5]

    scenarios = [
        ("Baseline",                1.0, 1.0),
        ("Low supply, low demand",  0.5, 0.5),
        ("High supply, high demand", 1.5, 1.5),
        ("Low supply, high demand", 0.5, 1.5),
        ("High supply, low demand", 1.5, 0.5),
    ]

    rows = []

    for crop in config.COMMODITIES:
        p_hat = forecast_prices[crop]
        current = config.CURRENT_MSP[crop]

        orig_s = config.ELASTICITIES[crop]["supply"]
        orig_d = config.ELASTICITIES[crop]["demand"]

        for name, s_mult, d_mult in scenarios:
            # temporarily modify elasticities
            config.ELASTICITIES[crop]["supply"] = orig_s * s_mult
            config.ELASTICITIES[crop]["demand"] = orig_d * d_mult

            m_star, w_star = find_optimal_msp(p_hat, crop, weights)

            above_current = m_star > current
            gap = round(m_star - current)

            rows.append({
                "scenario": name,
                "crop": crop,
                "supply_mult": s_mult,
                "demand_mult": d_mult,
                "eps_s": round(orig_s * s_mult, 3),
                "eps_d": round(orig_d * d_mult, 3),
                "optimal_msp": round(m_star),
                "gap_from_current": gap,
                "above_current": above_current,
            })

            print(f"    {crop:6s} {name:30s}: m*={m_star:.0f} "
                  f"({'above' if above_current else 'below'} current by {abs(gap)})")

            # restore
            config.ELASTICITIES[crop]["supply"] = orig_s
            config.ELASTICITIES[crop]["demand"] = orig_d

    results = pd.DataFrame(rows)
    results.to_csv(config.RESULTS / "sensitivity_elasticity.csv", index=False)
    return results


# ── main ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MSP Welfare Optimization — Sensitivity Analysis")
    print("=" * 60)

    fp = load_forecast_prices()
    print(f"\n  Forecast prices: wheat={fp['wheat']:.0f}, rice={fp['rice']:.0f}")

    proc_results = procurement_sensitivity(fp)
    elast_results = elasticity_stress_test(fp)

    # summary
    print("\n  ══ ROBUSTNESS SUMMARY ══")

    # check if rice always above current
    rice_all_above = elast_results[
        (elast_results["crop"] == "rice") & (elast_results["above_current"])
    ]
    print(f"  Rice above current MSP in {len(rice_all_above)}/{len(elast_results[elast_results['crop']=='rice'])} "
          f"elasticity scenarios")

    # check procurement: even at eta=3, rice still above current?
    rice_eta3 = proc_results[(proc_results["crop"] == "rice") &
                              (proc_results["eta"] == 3.0)]
    if not rice_eta3.empty:
        m = rice_eta3["optimal_msp"].values[0]
        gap = m - config.CURRENT_MSP["rice"]
        print(f"  Rice at eta=3.0: m*={m}, still Rs {gap} above current")

    print("\n" + "=" * 60)
    print("  Sensitivity analysis complete.")
    print("=" * 60)
