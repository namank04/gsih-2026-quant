# GSIH 2026 — Quantitative Hackathon

My solutions to the three problems from the Goldman Sachs GSIH 2026 quantitative
hackathon, spanning equities, fixed income and rates. Each problem is a
self-contained study with the working code, the key figures, and a writeup of the
approach and the reasoning behind it.

The set placed **4th** in the finals.

## The three problems

| # | Problem | Area | One line |
|---|---------|------|----------|
| 01 | [Regime Navigator](01-regime-navigator) | Equities · allocation | A weekly-rebalanced portfolio of 100 stocks that reads the market regime and switches playbook to match. |
| 02 | [Proxy Models for Bond ETFs](02-etf-proxy-models) | Fixed income · risk | Replicate five undisclosed bond funds from liquid instruments, then measure their rate/credit risk and hedge a book at lowest cost. |
| 03 | [Bermudan Swaption](03-bermudan-swaption) | Rates · derivatives | Price and risk-manage a co-terminal Bermudan payer swaption from first principles with a calibrated Hull–White model. |

Each folder has its own `README` with the full approach, a `solution.py`, and a
`figures/` directory.

## A theme across all three

The thread that runs through the three is the gap between *fitting data* and
*understanding the problem underneath it*: ranking equity signals by what protects
the book rather than what predicts returns (Q1); recognising that a fund's price is
recoverable from returns but its risk breakdown is not (Q2); and calibrating a small,
economically meaningful set of parameters rather than over-flexible ones (Q3).

## Tech

Python, with NumPy, pandas, scikit-learn (ridge regression) and SciPy. The figures
are produced with Matplotlib. Each `solution.py` runs standalone.

## Layout

```
01-regime-navigator/     # equities — regime-switching portfolio
02-etf-proxy-models/     # fixed income — replication, risk, hedging
03-bermudan-swaption/    # rates — Hull–White calibration + lattice pricing
slides/slides.pdf        # the presentation deck
```
