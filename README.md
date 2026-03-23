# Multi-Asset Portfolio Optimization via MIQP

Constructs institutionally realistic portfolios by formulating a Mixed-Integer Quadratic Program (MIQP) in Gurobi. Encodes cardinality constraints, minimum-lot thresholds, and sector concentration limits within a mean-variance objective. Benchmarked against standard MVO and 1/N baselines across equity, fixed income, and commodity universes, quantifying efficiency gains and constraint sensitivity on the risk-return frontier.

## Structure

```
miqp_portfolio/
├── miqp_optimizer.py      # Core: MIQP formulation, MVO/1N benchmarks, metrics
├── efficient_frontier.py  # Frontier tracing, cardinality & sector cap sensitivity
├── outputs/               # CSVs, plots
├── data/                  # Historical returns
└── requirements.txt
```

## Optimization Problem

### Objective
Minimize risk-adjusted portfolio variance:

```
min  w' Σ w  -  (1/λ) μ' w
```

Where:
- `w` = portfolio weights (continuous)
- `z` = binary selection indicators  
- `λ` = risk aversion parameter
- `μ` = expected return vector (annualized)
- `Σ` = covariance matrix (annualized)

### Constraints

| Constraint | Description |
|---|---|
| `Σ w_i = 1` | Budget (fully invested) |
| `w_i ≥ min_weight × z_i` | Minimum lot if asset selected |
| `w_i ≤ z_i` | No weight without selection |
| `Σ z_i ≤ K` | Cardinality (max K holdings) |
| `Σ_{i∈s} w_i ≤ cap_s` | Sector concentration cap |
| `z_i ∈ {0,1}, w_i ≥ 0` | Integrality and non-negativity |

### Default Parameters
- `max_holdings = 8` (cardinality)
- `min_weight = 2%` (minimum lot)
- `max_sector_wt = 45%` (concentration)
- `risk_aversion λ = 2.0`

## Asset Universe

| Asset Class | Tickers |
|---|---|
| Equity | SPY, QQQ, IWM, EFA, EEM |
| Fixed Income | AGG, TLT, HYG, EMB |
| Commodity | GLD, SLV, USO, DBA |

## Usage

```python
# Run full MIQP optimization and compare vs MVO/1N
python miqp_optimizer.py

# Trace efficient frontier and run sensitivity analysis
python efficient_frontier.py
```

## Requirements

Gurobi license required for MIQP. The code automatically falls back to a continuous MVO relaxation (via `scipy.optimize`) if Gurobi is unavailable.

```
gurobipy      # requires academic or commercial license
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
```

## Key Outputs

- `portfolio_comparison.csv` — Sharpe, Calmar, drawdown across MIQP/MVO/1N
- `portfolio_weights.csv` — Weight breakdown by asset and method
- `efficient_frontier.png` — Frontier plot comparing all three approaches
- `cardinality_sensitivity.csv` — Sharpe vs. max holdings K
- `sector_cap_sensitivity.csv` — Sharpe vs. sector concentration cap

## Notes

Simulated returns with block correlation structure are used for demonstration. In production, replace `simulate_returns()` with historical daily returns from Bloomberg, FactSet, or CRSP. Covariance shrinkage (Ledoit-Wolf) is recommended for live use.
