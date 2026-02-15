"""
05_welfare_optimization.py — Welfare optimization across policy scenarios

Implements the partial equilibrium welfare function:
    W(m) = alpha*DeltaPS - beta*DeltaCS - gamma*GC - delta*DWL

Proves strict concavity numerically and solves for m* under 4 weighting
scenarios using Model B forecast prices.

Usage:
    python src/05_welfare_optimization.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.utils import print_table

warnings.filterwarnings("ignore")


# ── welfare function components ────────────────────────

def delta_producer_surplus(m, p_hat, q_base, p_base, eps_s):
    """
    Change in producer surplus from MSP.
    Linear part: (m - p_hat) * Q_proc  [handled separately since Q_proc is exogenous]
    Quadratic part: from supply response
    """
    gap = m - p_hat
    # direct transfer to farmers via procurement
    linear = gap * config.PROCUREMENT_2024.get("wheat", 26.64)  # placeholder, overridden
    # supply-response surplus gain
    quadratic = 0.5 * eps_s * (gap**2) * q_base / p_base
    return linear + quadratic


def delta_consumer_surplus(m, p_hat, q_base, p_base, eps_d):
    """Change in consumer surplus (loss when m > p_hat)."""
    gap = m - p_hat
    # price pass-through to consumers — partial, mediated by market structure
    # for simplicity (and per the paper), assume full pass-through
    linear = gap * q_base  # consumer faces higher price
    quadratic = 0.5 * abs(eps_d) * (gap**2) * q_base / p_base
    return linear + quadratic


def govt_cost(m, p_hat, q_proc, kappa):
    """Government procurement cost including overhead."""
    gap = m - p_hat
    return gap * q_proc * (1 + kappa)


def deadweight_loss(m, p_hat, q_base, p_base, eps_s, eps_d):
    """DWL from price distortion."""
    gap = m - p_hat
    return 0.5 * (eps_s + abs(eps_d)) * (gap**2) * q_base / p_base


def welfare(m, p_hat, crop, weights, q_proc=None, eta=0):
    """
    Full welfare function W(m).

    Args:
        m: MSP level (Rs/quintal)
        p_hat: forecast market price (Model B)
        crop: "wheat" or "rice"
        weights: dict with alpha, beta, gamma, delta
        q_proc: procurement volume in MMT (if None, use config)
        eta: procurement endogeneity param (MMT per Rs 100 gap)
    """
    alpha = weights["alpha"]
    beta = weights["beta"]
    gamma = weights["gamma"]
    delta = weights.get("delta", 0)

    elast = config.ELASTICITIES[crop]
    eps_s = elast["supply"]
    eps_d = elast["demand"]

    if q_proc is None:
        q_proc = config.PROCUREMENT_2024[crop]

    # endogenous procurement adjustment
    if eta > 0:
        q_proc = q_proc + eta * (m - p_hat) / 100  # eta is per Rs 100

    q_base = config.TOTAL_PRODUCTION[crop]
    p_base = p_hat  # use forecast as base price

    kappa = config.KAPPA

    gap = m - p_hat

    # ── component calculations (Rs crore) ──
    # conversion: gap (Rs/qtl) × quantity (MMT) × 10 = Rs crore
    # (1 MMT = 10^7 quintals, 1 crore = 10^7, so MMT * Rs/qtl = crore)

    # producer surplus change
    ps_linear = gap * q_proc * 10  # direct transfer
    ps_quad = 0.5 * eps_s * (gap**2 / p_base) * q_base * 10  # supply response
    d_ps = ps_linear + ps_quad

    # consumer surplus change (loss)
    q_consumer = q_base - q_proc
    cs_loss = abs(gap) * q_consumer * 10 if gap > 0 else 0
    cs_quad = 0.5 * abs(eps_d) * (gap**2 / p_base) * q_consumer * 10
    d_cs = cs_loss + cs_quad

    # government cost
    gc = gap * q_proc * (1 + kappa) * 10

    # deadweight loss
    dwl = 0.5 * (eps_s + abs(eps_d)) * (gap**2 / p_base) * q_base * 10

    # weighted welfare
    W = alpha * d_ps - beta * d_cs - gamma * gc - delta * dwl
    return W


def welfare_neg(m, p_hat, crop, weights, q_proc=None, eta=0):
    """Negative welfare for minimization."""
    return -welfare(m, p_hat, crop, weights, q_proc, eta)


# ── strict concavity check ────────────────────────────

def check_concavity(p_hat, crop, weights, n_points=500):
    """
    Numerically verify d²W/dm² < 0 over the relevant range.
    Returns True if strictly concave everywhere checked.
    """
    m_range = np.linspace(p_hat - 500, p_hat + 1500, n_points)
    W_vals = [welfare(m, p_hat, crop, weights) for m in m_range]

    # numerical second derivative
    d2W = np.gradient(np.gradient(W_vals, m_range), m_range)

    is_concave = np.all(d2W < 0)
    max_d2W = np.max(d2W)
    return is_concave, max_d2W


# ── find optimal MSP ──────────────────────────────────

def find_optimal_msp(p_hat, crop, weights, q_proc=None, eta=0):
    """Find m* that maximizes W(m). Uses bounded scalar optimization."""
    # search over a wide range around the forecast price
    result = minimize_scalar(welfare_neg, bounds=(p_hat - 200, p_hat + 1500),
                              method="bounded",
                              args=(p_hat, crop, weights, q_proc, eta))

    m_star = result.x
    w_star = -result.fun
    return m_star, w_star


# ── welfare decomposition at a given m ────────────────

def welfare_components(m, p_hat, crop, weights, q_proc=None):
    """Return individual welfare components at a given MSP level."""
    elast = config.ELASTICITIES[crop]
    eps_s = elast["supply"]
    eps_d = elast["demand"]

    if q_proc is None:
        q_proc = config.PROCUREMENT_2024[crop]
    q_base = config.TOTAL_PRODUCTION[crop]
    q_consumer = q_base - q_proc
    p_base = p_hat
    kappa = config.KAPPA
    gap = m - p_hat

    # all in Rs crore (gap * quantity_MMT * 10)
    d_ps = gap * q_proc * 10 + 0.5 * eps_s * (gap**2 / p_base) * q_base * 10
    d_cs = abs(gap) * q_consumer * 10 + 0.5 * abs(eps_d) * (gap**2 / p_base) * q_consumer * 10 if gap > 0 else 0
    gc = gap * q_proc * (1 + kappa) * 10
    dwl = 0.5 * (eps_s + abs(eps_d)) * (gap**2 / p_base) * q_base * 10

    return {"delta_ps": d_ps, "delta_cs": d_cs, "govt_cost": gc, "dwl": dwl}


# ── welfare regret (policy loss function) ─────────────

def welfare_regret(y_true, y_pred, crop, weights):
    """
    Welfare regret from using y_pred instead of y_true for optimization.
    L(P_hat) = W(m*(P)) - W(m*(P_hat))
    """
    # optimal MSP under true prices
    m_true, w_true = find_optimal_msp(np.mean(y_true), crop, weights)
    # optimal MSP under predicted prices
    m_pred, _ = find_optimal_msp(np.mean(y_pred), crop, weights)
    # welfare at m_pred evaluated at true prices
    w_at_pred = welfare(m_pred, np.mean(y_true), crop, weights)
    regret = w_true - w_at_pred
    return max(regret, 0)  # regret is non-negative


# ── main scenario analysis ────────────────────────────

def run_scenarios(forecast_prices):
    """Run welfare optimization for all crop × scenario combinations."""
    print("\n" + "=" * 60)
    print("  Welfare Optimization — Scenario Analysis")
    print("=" * 60)

    rows = []

    for scenario_name, weights in config.SCENARIOS.items():
        for crop in config.COMMODITIES:
            p_hat = forecast_prices[crop]
            current_msp = config.CURRENT_MSP[crop]

            # check concavity
            is_concave, max_d2 = check_concavity(p_hat, crop, weights)
            if not is_concave:
                print(f"  WARNING: not strictly concave for {crop}/{scenario_name}")

            # find optimal
            m_star, w_star = find_optimal_msp(p_hat, crop, weights)

            # components at optimal
            comps = welfare_components(m_star, p_hat, crop, weights)

            rows.append({
                "scenario": scenario_name,
                "crop": crop,
                "current_msp": current_msp,
                "forecast_price": round(p_hat),
                "optimal_msp": round(m_star),
                "delta_ps": round(comps["delta_ps"]),
                "delta_cs": round(-comps["delta_cs"]),
                "govt_cost": round(comps["govt_cost"]),
                "net_welfare": round(w_star),
                "gap_from_current": round(m_star - current_msp),
                "concave": is_concave,
            })

    results_df = pd.DataFrame(rows)
    print_table(results_df[["scenario", "crop", "current_msp", "forecast_price",
                             "optimal_msp", "delta_ps", "delta_cs", "net_welfare"]],
                "Welfare Optimization Results")

    results_df.to_csv(config.RESULTS / "welfare_scenarios.csv", index=False)
    return results_df


# ── welfare regret comparison ─────────────────────────

def compute_welfare_regret():
    """
    Compare welfare regret across models.
    Loads Model A and B predictions and computes regret for each.
    """
    print("\n  Computing welfare regret across models...")
    weights = config.SCENARIOS["equal_weights"]

    for crop in config.COMMODITIES:
        try:
            preds_a = pd.read_csv(config.RESULTS / f"predictions_a_{crop}.csv")
            preds_b = pd.read_csv(config.RESULTS / f"predictions_b_{crop}.csv")
        except FileNotFoundError:
            print(f"    Predictions not found for {crop}. Run 03/04 first.")
            continue

        y_true = preds_a["actual"].values

        regrets = {}
        for col in ["naive", "rf", "gb"]:
            if col in preds_a.columns:
                regrets[f"{col}_A"] = welfare_regret(y_true, preds_a[col].values,
                                                      crop, weights)
            if col in preds_b.columns:
                regrets[f"{col}_B"] = welfare_regret(y_true, preds_b[col].values,
                                                      crop, weights)

        print(f"\n  Welfare regret ({crop}, Rs crore/year):")
        for model, reg in sorted(regrets.items(), key=lambda x: x[1]):
            print(f"    {model:20s}: {reg:.1f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  MSP Welfare Optimization")
    print("=" * 60)

    # load Model B forecast prices
    try:
        fc_df = pd.read_csv(config.RESULTS / "model_b_forecasts.csv")
        forecast_prices = dict(zip(fc_df["crop"], fc_df["forecast_price"]))
        print(f"\n  Model B forecast prices loaded:")
        for c, p in forecast_prices.items():
            print(f"    {c}: Rs {p:.0f}/quintal")
    except FileNotFoundError:
        print("\n  Model B forecasts not found. Run 04_policy_neutral.py first.")
        print("  Using paper's reference values for demonstration.")
        # these are the Model B RF mean forecasts from the paper (Table 4)
        forecast_prices = {"wheat": 2350, "rice": 2967}

    # concavity verification
    print("\n  Concavity verification:")
    for crop in config.COMMODITIES:
        for name, w in config.SCENARIOS.items():
            ok, d2 = check_concavity(forecast_prices[crop], crop, w)
            status = "PASS" if ok else "FAIL"
            print(f"    {crop:6s} / {name:20s}: {status} (max d²W/dm² = {d2:.4f})")

    # scenario analysis
    results = run_scenarios(forecast_prices)

    # welfare regret
    compute_welfare_regret()

    print("\n" + "=" * 60)
    print("  Welfare optimization complete.")
    print("=" * 60)
