"""
efficient_frontier.py
=====================
Traces the efficient frontier for MIQP and MVO portfolios.
Analyzes constraint sensitivity: how cardinality and sector caps
affect risk-return outcomes.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from miqp_optimizer import simulate_returns, miqp_optimize, mvo_continuous, equal_weight, portfolio_metrics, ASSETS, SECTOR_MAP, N


def trace_frontier(mu, Sigma, n_points=30, method="mvo"):
    """
    Trace efficient frontier by varying risk aversion parameter.
    Returns DataFrame of (Ann_Vol, Ann_Return, Sharpe) pairs.
    """
    lambdas = np.logspace(-1, 2, n_points)
    points  = []

    for lam in lambdas:
        if method == "mvo":
            sol = mvo_continuous(mu, Sigma, risk_aversion=lam)
        else:
            sol = miqp_optimize(mu, Sigma, risk_aversion=lam,
                                max_holdings=8, min_weight=0.02,
                                max_sector_wt=0.45, sector_map=SECTOR_MAP)

        p_ret  = sol["weights"] @ mu
        p_vol  = np.sqrt(sol["weights"] @ Sigma @ sol["weights"])
        points.append({"Ann_Return": p_ret, "Ann_Vol": p_vol,
                       "Sharpe": p_ret / p_vol if p_vol > 0 else 0,
                       "N_Holdings": sol["n_holdings"]})

    return pd.DataFrame(points)


def cardinality_sensitivity(mu, Sigma, k_range=None):
    """
    How does Sharpe change as max_holdings increases from 2 to N?
    """
    if k_range is None:
        k_range = range(2, N + 1)

    rows = []
    for k in k_range:
        sol = miqp_optimize(mu, Sigma, risk_aversion=2.0,
                            max_holdings=k, min_weight=0.02,
                            max_sector_wt=0.99)  # relax sector limit
        p_ret = sol["weights"] @ mu
        p_vol = np.sqrt(sol["weights"] @ Sigma @ sol["weights"])
        rows.append({
            "Max_Holdings": k,
            "N_Selected":   sol["n_holdings"],
            "Ann_Return_%": round(p_ret * 100, 2),
            "Ann_Vol_%":    round(p_vol * 100, 2),
            "Sharpe":       round(p_ret / p_vol, 3) if p_vol > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def sector_cap_sensitivity(mu, Sigma, cap_range=None):
    """
    How does portfolio quality change as sector concentration limit tightens?
    """
    if cap_range is None:
        cap_range = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.99]

    rows = []
    for cap in cap_range:
        sol = miqp_optimize(mu, Sigma, risk_aversion=2.0,
                            max_holdings=8, min_weight=0.02,
                            max_sector_wt=cap, sector_map=SECTOR_MAP)
        p_ret = sol["weights"] @ mu
        p_vol = np.sqrt(sol["weights"] @ Sigma @ sol["weights"])
        rows.append({
            "Sector_Cap":   cap,
            "N_Holdings":   sol["n_holdings"],
            "Ann_Return_%": round(p_ret * 100, 2),
            "Ann_Vol_%":    round(p_vol * 100, 2),
            "Sharpe":       round(p_ret / p_vol, 3) if p_vol > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_frontiers(mvo_pts, miqp_pts, ew_vol, ew_ret, save_path="outputs/efficient_frontier.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mvo_pts["Ann_Vol"]*100,  mvo_pts["Ann_Return"]*100,
            label="MVO (Continuous)", color="#1f77b4", lw=2)
    ax.plot(miqp_pts["Ann_Vol"]*100, miqp_pts["Ann_Return"]*100,
            label="MIQP (k=8)",      color="#d62728", lw=2, linestyle="--")
    ax.scatter([ew_vol*100], [ew_ret*100], marker="*", s=200,
               color="#2ca02c", label="1/N Equal Weight", zorder=5)
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Efficient Frontier: MIQP vs MVO vs 1/N")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Frontier plot saved to {save_path}")


if __name__ == "__main__":
    returns = simulate_returns()
    mu      = returns.mean().values * 252
    Sigma   = returns.cov().values * 252

    print("Tracing MVO frontier...")
    mvo_pts  = trace_frontier(mu, Sigma, method="mvo")

    print("Tracing MIQP frontier...")
    miqp_pts = trace_frontier(mu, Sigma, method="miqp")

    ew = equal_weight(N)
    ew_ret = ew["weights"] @ mu
    ew_vol = np.sqrt(ew["weights"] @ Sigma @ ew["weights"])

    plot_frontiers(mvo_pts, miqp_pts, ew_vol, ew_ret)

    print("\nCardinality Sensitivity:")
    card_df = cardinality_sensitivity(mu, Sigma, k_range=range(2, N+1))
    print(card_df.to_string(index=False))
    card_df.to_csv("outputs/cardinality_sensitivity.csv", index=False)

    print("\nSector Cap Sensitivity:")
    sec_df = sector_cap_sensitivity(mu, Sigma)
    print(sec_df.to_string(index=False))
    sec_df.to_csv("outputs/sector_cap_sensitivity.csv", index=False)
