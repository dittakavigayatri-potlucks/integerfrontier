"""
Multi-Asset Portfolio Optimization via Mixed-Integer Quadratic Programming (MIQP)
==================================================================================
Formulates institutionally realistic portfolios subject to:
  - Cardinality constraints (max number of holdings)
  - Minimum lot size thresholds
  - Sector concentration limits
  - Mean-variance objective

Solver: Gurobi (requires license; falls back to a continuous MVO relaxation via scipy)
Benchmarks: standard MVO, 1/N equal-weight

Author: Naga Siva Gayatri Dittakavi
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Gurobi not found. Using scipy fallback for continuous MVO.")

from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. UNIVERSE & DATA
# ---------------------------------------------------------------------------

ASSET_UNIVERSE = {
    # Equities
    "SPY":  {"sector": "equity",       "asset_class": "equity"},
    "QQQ":  {"sector": "equity",       "asset_class": "equity"},
    "IWM":  {"sector": "equity",       "asset_class": "equity"},
    "EFA":  {"sector": "equity",       "asset_class": "equity"},
    "EEM":  {"sector": "equity",       "asset_class": "equity"},
    # Fixed Income
    "AGG":  {"sector": "fixed_income", "asset_class": "fixed_income"},
    "TLT":  {"sector": "fixed_income", "asset_class": "fixed_income"},
    "HYG":  {"sector": "fixed_income", "asset_class": "fixed_income"},
    "EMB":  {"sector": "fixed_income", "asset_class": "fixed_income"},
    # Commodities
    "GLD":  {"sector": "commodity",    "asset_class": "commodity"},
    "SLV":  {"sector": "commodity",    "asset_class": "commodity"},
    "USO":  {"sector": "commodity",    "asset_class": "commodity"},
    "DBA":  {"sector": "commodity",    "asset_class": "commodity"},
}

SECTOR_MAP = {asset: meta["sector"] for asset, meta in ASSET_UNIVERSE.items()}
ASSETS     = list(ASSET_UNIVERSE.keys())
N          = len(ASSETS)


def simulate_returns(n_assets: int = N, n_periods: int = 1260, seed: int = 42) -> pd.DataFrame:
    """
    Simulate daily returns with realistic cross-asset correlation structure.
    In production: replace with yfinance/Bloomberg/FactSet historical returns.
    """
    np.random.seed(seed)

    # Approximate annualized return vector (daily)
    ann_ret = np.array([0.10, 0.12, 0.09, 0.08, 0.07,     # equities
                        0.04, 0.03, 0.05, 0.05,             # fixed income
                        0.05, 0.04, 0.03, 0.02])            # commodities
    daily_ret = ann_ret / 252

    # Block correlation matrix
    corr = np.eye(n_assets)
    equity_idx = list(range(5))
    fi_idx     = list(range(5, 9))
    com_idx    = list(range(9, 13))

    for i in equity_idx:
        for j in equity_idx:
            if i != j: corr[i, j] = 0.70 + np.random.randn() * 0.05
    for i in fi_idx:
        for j in fi_idx:
            if i != j: corr[i, j] = 0.60 + np.random.randn() * 0.05
    for i in com_idx:
        for j in com_idx:
            if i != j: corr[i, j] = 0.40 + np.random.randn() * 0.05
    for i in equity_idx:
        for j in fi_idx:
            corr[i, j] = corr[j, i] = -0.20 + np.random.randn() * 0.05

    # Force PSD
    corr = (corr + corr.T) / 2
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 0:
        corr += (-eigvals.min() + 1e-6) * np.eye(n_assets)

    vols = np.array([0.18, 0.22, 0.20, 0.16, 0.22,
                     0.05, 0.12, 0.10, 0.12,
                     0.14, 0.22, 0.28, 0.16]) / np.sqrt(252)

    cov_daily = np.outer(vols, vols) * corr
    L         = np.linalg.cholesky(cov_daily)
    z         = np.random.randn(n_periods, n_assets)
    returns   = z @ L.T + daily_ret

    return pd.DataFrame(returns, columns=ASSETS)


# ---------------------------------------------------------------------------
# 2. MIQP FORMULATION (Gurobi)
# ---------------------------------------------------------------------------

def miqp_optimize(
    mu:              np.ndarray,
    Sigma:           np.ndarray,
    risk_aversion:   float = 2.0,
    max_holdings:    int   = 8,
    min_weight:      float = 0.02,
    max_sector_wt:   float = 0.45,
    sector_map:      dict  = None,
) -> dict:
    """
    Solve the MIQP:

        min  w' Sigma w  -  (1/risk_aversion) * mu' w
        s.t. sum(w) = 1
             w_i >= min_weight * z_i      (minimum lot if selected)
             w_i <= z_i                   (binary selection)
             sum(z_i) <= max_holdings     (cardinality)
             sum_{i in sector s} w_i <= max_sector_wt  (concentration)
             z_i in {0,1}, w_i >= 0

    Returns optimal weights dict.
    """
    if not GUROBI_AVAILABLE:
        print("Gurobi unavailable — running MVO relaxation instead.")
        return mvo_continuous(mu, Sigma, risk_aversion)

    n = len(mu)
    try:
        m = gp.Model("MIQP_Portfolio")
        m.setParam("OutputFlag", 0)

        w = m.addVars(n, lb=0.0, ub=1.0, name="w")
        z = m.addVars(n, vtype=GRB.BINARY, name="z")

        # Objective: min risk - return/lambda
        quad_expr = gp.QuadExpr()
        for i in range(n):
            for j in range(n):
                quad_expr += Sigma[i, j] * w[i] * w[j]
        lin_expr = gp.LinExpr()
        for i in range(n):
            lin_expr += (1.0 / risk_aversion) * mu[i] * w[i]

        m.setObjective(quad_expr - lin_expr, GRB.MINIMIZE)

        # Budget constraint
        m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1.0)

        # Cardinality
        m.addConstr(gp.quicksum(z[i] for i in range(n)) <= max_holdings)

        # Min lot and linking
        for i in range(n):
            m.addConstr(w[i] >= min_weight * z[i])
            m.addConstr(w[i] <= z[i])

        # Sector concentration
        if sector_map:
            sectors = set(sector_map.values())
            for sector in sectors:
                idx = [j for j, a in enumerate(ASSETS) if sector_map.get(a) == sector]
                m.addConstr(gp.quicksum(w[j] for j in idx) <= max_sector_wt)

        m.optimize()

        if m.status == GRB.OPTIMAL:
            weights = np.array([w[i].X for i in range(n)])
            selected = np.array([z[i].X for i in range(n)])
            return {
                "weights":      weights,
                "selected":     selected.astype(int),
                "n_holdings":   int(selected.sum()),
                "status":       "OPTIMAL",
                "obj_value":    m.ObjVal,
            }
        else:
            print(f"Gurobi status: {m.status}")
            return mvo_continuous(mu, Sigma, risk_aversion)

    except Exception as e:
        print(f"Gurobi error: {e}. Falling back to MVO.")
        return mvo_continuous(mu, Sigma, risk_aversion)


# ---------------------------------------------------------------------------
# 3. BENCHMARK PORTFOLIOS
# ---------------------------------------------------------------------------

def mvo_continuous(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float = 2.0) -> dict:
    """Standard mean-variance optimization (continuous, long-only)."""
    n = len(mu)

    def objective(w):
        return w @ Sigma @ w - (1.0 / risk_aversion) * mu @ w

    def grad(w):
        return 2 * Sigma @ w - (1.0 / risk_aversion) * mu

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(0.0, 1.0)] * n
    w0          = np.ones(n) / n

    res = minimize(objective, w0, jac=grad, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})

    return {
        "weights":    res.x,
        "selected":   (res.x > 1e-3).astype(int),
        "n_holdings": int((res.x > 1e-3).sum()),
        "status":     "MVO_CONTINUOUS",
        "obj_value":  res.fun,
    }


def equal_weight(n: int) -> dict:
    return {
        "weights":    np.ones(n) / n,
        "selected":   np.ones(n, dtype=int),
        "n_holdings": n,
        "status":     "1_OVER_N",
        "obj_value":  None,
    }


# ---------------------------------------------------------------------------
# 4. PERFORMANCE METRICS
# ---------------------------------------------------------------------------

def portfolio_metrics(weights: np.ndarray, returns: pd.DataFrame) -> dict:
    port_ret  = returns.values @ weights
    ann_ret   = port_ret.mean() * 252
    ann_vol   = port_ret.std() * np.sqrt(252)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cumret    = (1 + port_ret).cumprod()
    drawdowns = cumret / cumret.cummax() - 1
    max_dd    = drawdowns.min()

    return {
        "Ann_Return_%":   round(ann_ret  * 100, 2),
        "Ann_Vol_%":      round(ann_vol  * 100, 2),
        "Sharpe":         round(sharpe, 3),
        "Max_DD_%":       round(max_dd  * 100, 2),
        "Calmar":         round(ann_ret / abs(max_dd), 3) if max_dd < 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# 5. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Simulating returns...")
    returns = simulate_returns()
    mu      = returns.mean().values * 252         # annualized
    Sigma   = returns.cov().values * 252          # annualized

    print(f"Universe: {N} assets across equity, fixed income, commodity\n")

    # --- MIQP ---
    print("Running MIQP optimization (cardinality=8, min_lot=2%, max_sector=45%)...")
    miqp_sol = miqp_optimize(mu, Sigma, risk_aversion=2.0,
                              max_holdings=8, min_weight=0.02,
                              max_sector_wt=0.45, sector_map=SECTOR_MAP)

    # --- MVO ---
    mvo_sol  = mvo_continuous(mu, Sigma, risk_aversion=2.0)

    # --- 1/N ---
    ew_sol   = equal_weight(N)

    results = {}
    for label, sol in [("MIQP", miqp_sol), ("MVO", mvo_sol), ("1/N", ew_sol)]:
        m = portfolio_metrics(sol["weights"], returns)
        m["N_Holdings"] = sol["n_holdings"]
        results[label]  = m

    df = pd.DataFrame(results).T
    print("\n--- Portfolio Comparison ---")
    print(df.to_string())

    # --- Weight breakdown ---
    w_df = pd.DataFrame({
        "Asset":   ASSETS,
        "Sector":  [SECTOR_MAP[a] for a in ASSETS],
        "MIQP_w":  miqp_sol["weights"].round(4),
        "MVO_w":   mvo_sol["weights"].round(4),
        "EW_w":    ew_sol["weights"].round(4),
    }).set_index("Asset")
    print("\n--- Weight Breakdown ---")
    print(w_df[w_df["MIQP_w"] > 0.001].to_string())

    df.to_csv("outputs/portfolio_comparison.csv")
    w_df.to_csv("outputs/portfolio_weights.csv")
    print("\nSaved to outputs/")
