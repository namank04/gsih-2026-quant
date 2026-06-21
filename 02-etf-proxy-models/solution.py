import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

##### DO NOT MODIFY/DELETE THE BELOW CODE #############################################################################

#######################################################################################################################
# CONSTANTS STARTS
# BELOW ARE "NECESSARY" CONSTANTS
######################################################################################################################=

"""End-Of-Day (EOD ) prices of target ETFs and proxy instruments, you have to use this to build whatever model you wan to."""
EOD_PRICES     = pd.read_csv( "./eod_prices.csv", index_col = 0, parse_dates=True ).sort_index()

# Defn. of Instruments - use these while constructing your output dataframes
TARGET_ETFS  = [f"Target_ETF_{i}" for i in range(1, 6)]

# PROXY INSTRUMENTS AVAILABLE

# 1). ETFs
PROXY_ETFS   = [f"Proxy_ETF_{i}" for i in range(1, 11)]

# 2). TREASURY FUTURES
TSY_FUTURES  = ["TU", "FV", "TY", "US"]

# 3). TREASURY OTR ("one-the-run") BONDS
TSY_BONDS    = ["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"]

# 4). CREDIT DEFAULT SWAP INDICES
CDX          = ["CDX_IG_5Y", "CDX_IG_10Y", "CDX_HY_5Y", "CDX_HY_10Y"]

"""You can use all proxy instruments for 'NAV' but ONLY 'RISK_PROXY' instruments for building proxy model for 'Risk'."""

# AVAILABE PROXY INSTRUMENTS FOR NAV MODEL ( = ALL PROXY INSTRUMENTS )
ALL_PROXY    = PROXY_ETFS + TSY_FUTURES + TSY_BONDS + CDX

# AVAILABE PROXY INSTRUMENTS FOR RISK MODEL ( = ALL - PROXT ETFs INSTRUMENTS )
RISK_PROXY   = TSY_FUTURES + TSY_BONDS + CDX

# Par-quoted instruments (price per 100); ETFs are NAV per share
# EOD_PRICES has price per 100 par notional for PAR_QUOTED instruments, whereas, for ETFs (target & proxy), it's per share == NAV
PAR_QUOTED   = set(TSY_FUTURES + TSY_BONDS + CDX)

# Transaction cost for instruemnts, this will be applied on your hedging basket
# One-way transaction cost (bps on |market value|)
COST_BPS = {**{e: 2.0 for e in PROXY_ETFS},
            **{f: 0.5 for f in TSY_FUTURES},
            **{b: 0.5 for b in TSY_BONDS},
            **{c: 1.0 for c in CDX}}

#######################################################################################################################
# CONSTANTS END
#######################################################################################################################

##### DO NOT MODIFY/DELETE THE BELOW CODE #############################################################################

#######################################################################################################################
# HELPER FUNCTIONS

# BELOW FUNCTIONS ARE PROVIDED AS BOILER-PLATE CODE
# THESE FUNCTIONS ARE ALSO USED IN EVALUATION, RECOMMENDED TO USE THEM, BUT NOT MANDATORY
#######################################################################################################################

#######################################################################################################################
# HELPER FUNCTIONS START
#######################################################################################################################

def compute_returns(prices):
    """Compute simple daily returns from prices."""
    return prices.pct_change().dropna()

def predict_returns(weights, proxy_returns):
    """Predict target ETF returns from proxy returns and model weights."""
    return proxy_returns @ weights

def reconstruct_nav(weights, proxy_returns, base_nav):
    """Reconstruct NAV level from predicted returns chained off a base NAV."""
    pred_returns = predict_returns(weights, proxy_returns)
    return (1 + pred_returns).cumprod() * base_nav

def compute_mape(predicted_nav, actual_nav):
    """Compute mean absolute percentage error between predicted and actual NAV."""
    aligned = pd.concat([predicted_nav, actual_nav], axis=1, join="inner").dropna()
    aligned.columns = ["predicted", "actual"]
    return (abs(aligned["predicted"] - aligned["actual"]) / aligned["actual"]).mean() * 100
  
def compute_pnl_series(notionals: dict, prices: pd.DataFrame) -> pd.Series:
    """mm USD PnL_t = sum_i scaling_i * (P_i,t - P_i,t-1)."""
    if not notionals:
        return pd.Series(0.0, index=prices.index[1:])
    p0 = prices.iloc[0]
    scaled = pd.Series({c: n * (1.0 / p0[c] if c not in PAR_QUOTED else 1.0 / 100.0)
                        for c, n in notionals.items()})
    return prices[scaled.index].diff().iloc[1:].mul(scaled, axis=1).sum(axis=1)

def compute_cost(notionals: dict, prices: pd.DataFrame) -> float:
    """
    compute transaction cost of setting up the hedging basket
    One-way cost on |market value| at t=0:
      ETF : |MV| * bps / 1e4                       (Notional IS MV)
      Par : |N|/100 * P_0 * bps / 1e4              (par * dirty price / 100)
    """
    p0 = prices.iloc[0]
    return float(sum(
        abs(n) * COST_BPS[i] / 1e4 if i not in PAR_QUOTED
        else (abs(n) / 100.0) * p0[i] * COST_BPS[i] / 1e4
        for i, n in notionals.items()
    ))

def compute_her(pnl_port: pd.Series, pnl_hedge: pd.Series) -> float:
    """compute hedge effectiveness ratio.""" 
    var_p = pnl_port.var(ddof=0)
    return 0.0 if var_p <= 0 else float(
        1.0 - pnl_port.add(pnl_hedge, fill_value=0.0).var(ddof=0) / var_p)

#######################################################################################################################
# HELPER FUNCTIONS END
#######################################################################################################################


#######################################################################################################################
# YOUR CODE STARTS HERE
#######################################################################################################################

import sys
from sklearn.linear_model import RidgeCV

# ---------------------------------------------------------------------------------------------------------------------
# Training universe: simple daily returns over the full 2025 training window.
# compute_returns() = prices.pct_change().dropna() -> r_t = (P_t - P_{t-1}) / P_{t-1}, rows with any NaN dropped.
# ---------------------------------------------------------------------------------------------------------------------
_RET = compute_returns(EOD_PRICES)

# Ridge cross-validation grid (RidgeCV uses efficient leave-one-out / GCV to pick alpha).
_RIDGE_ALPHAS = np.logspace(-6, 6, 100)
# Coefficients with |w| below this are treated as exactly zero (per problem spec).
_COEF_TOL     = 1e-6
# Hedge legs smaller than this (mm USD) are dropped to avoid paying transaction cost for negligible positions.
_MIN_NOTIONAL = 0.5
# Max in-sample HER we are willing to give up (vs the full OLS hedge) in exchange for a sparser, cheaper basket.
_HER_BUDGET   = 0.01


def _fit_ridge_block(targets, fit_proxies, out_proxies=None):
    """Ridge regression (CV alpha, no intercept) of each target ETF return on `fit_proxies`.
    The result is laid out across `out_proxies` columns (instruments not fitted are left at 0)."""
    fit_proxies = list(fit_proxies)
    out_proxies = list(out_proxies) if out_proxies is not None else fit_proxies
    X = _RET[fit_proxies].values
    rows = {}
    for etf in targets:
        y = _RET[etf].values
        model = RidgeCV(alphas=_RIDGE_ALPHAS, fit_intercept=False)
        model.fit(X, y)
        coef = np.asarray(model.coef_, dtype=float)
        coef[np.abs(coef) < _COEF_TOL] = 0.0
        rows[etf] = coef
    out = pd.DataFrame.from_dict(rows, orient="index", columns=fit_proxies)
    out = out.reindex(index=list(targets), columns=out_proxies).fillna(0.0).astype(float)
    out.index.name = "ETF"
    return out


def train_nav_model():
    """Part 1 - NAV: regress each target ETF return on ALL proxy instruments (replication / pricing)."""
    return _fit_ridge_block(TARGET_ETFS, ALL_PROXY)


def train_risk_model():
    """Part 2 - Risk: regress each target ETF return on Treasury futures + CDX only.
    On-the-run bonds are excluded (near-collinear with the futures, which blurs per-tenor DV01)
    and reported as 0 to keep the required RISK_PROXY column schema."""
    return _fit_ridge_block(TARGET_ETFS, TSY_FUTURES + CDX, out_proxies=RISK_PROXY)


def _unit_pnl_returns():
    """Per-unit-notional daily PnL series, matching the evaluation's compute_pnl_series convention:
    par-quoted instruments use dP/100, ETFs use the simple return dP/P_{t-1}."""
    diff = EOD_PRICES.diff()
    pct  = EOD_PRICES.pct_change()
    unit = pd.DataFrame(index=EOD_PRICES.index)
    for c in EOD_PRICES.columns:
        unit[c] = (diff[c] / 100.0) if c in PAR_QUOTED else pct[c]
    return unit.dropna()


def _ols_rss(Xs, y):
    """OLS (no intercept) residual sum of squares and coefficients for a column subset."""
    if Xs.shape[1] == 0:
        return float(y @ y), np.zeros(0)
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    r = y - Xs @ beta
    return float(r @ r), beta


def _cost_per_mm(instruments):
    """Approximate one-way transaction cost (USD) per 1 mm of |notional|, per compute_cost():
    ETFs cost bps on notional (= market value); par instruments cost bps on (P0/100)*notional."""
    p0 = EOD_PRICES.iloc[0]
    out = []
    for inst in instruments:
        if inst in PAR_QUOTED:
            out.append((float(p0[inst]) / 100.0) * COST_BPS[inst] / 1e4)
        else:
            out.append(COST_BPS[inst] / 1e4)
    return np.asarray(out, dtype=float)


def compute_hedge(portfolio):
    """Part 3 - Hedging: variance-minimising static hedge for the given target-ETF portfolio (mm USD)."""
    instruments = list(ALL_PROXY)  # all proxies allowed; target ETFs may NOT be used as hedges

    def _basket(values):
        """Return a basket of only the non-zero legs (submission format = instruments actually used)."""
        s = pd.Series(values, index=instruments, dtype=float)
        s = s[s != 0.0]
        out = s.rename("Notional").to_frame()
        out.index.name = "Instrument"
        return out

    def _default_basket():
        """Non-empty, schema-valid placeholder for the no-portfolio (validation) case."""
        out = pd.Series(1.0, index=instruments, dtype=float).rename("Notional").to_frame()
        out.index.name = "Instrument"
        return out

    port = {}
    if portfolio:
        port = {k: float(v) for k, v in portfolio.items()
                if k in TARGET_ETFS and v is not None and float(v) != 0.0}

    # No portfolio to hedge (e.g. schema-validation case): return a non-empty, schema-valid placeholder basket.
    if not port:
        return _default_basket()

    unit = _unit_pnl_returns()
    cols = list(dict.fromkeys(list(port.keys()) + instruments))
    data = unit[cols].dropna()

    # Portfolio per-day PnL (mm USD) and hedge-instrument design matrix in the same PnL units.
    y = np.zeros(len(data))
    for k, n in port.items():
        y = y + n * data[k].values
    X = data[instruments].values
    n_obs = len(y)
    var_y = float(np.var(y))

    rss_full, beta_full = _ols_rss(X, y)
    # Degenerate portfolio (no PnL variance) -> nothing meaningful to hedge.
    if var_y <= 0 or not np.all(np.isfinite(beta_full)):
        return _default_basket()

    # Cost-aware backward elimination: drop the instrument that saves the most transaction cost per
    # unit of variance re-introduced, as long as cumulative HER loss stays within _HER_BUDGET of full OLS.
    cost_mm    = _cost_per_mm(instruments)
    budget_rss = rss_full + _HER_BUDGET * var_y * n_obs
    active     = list(range(len(instruments)))
    cur_rss, cur_beta = rss_full, beta_full
    while len(active) > 1:
        cur_not = np.zeros(len(instruments))
        cur_not[active] = -cur_beta
        best = None
        for pos, j in enumerate(active):
            trial = active[:pos] + active[pos + 1:]
            rss_t, beta_t = _ols_rss(X[:, trial], y)
            if rss_t > budget_rss + 1e-9:
                continue
            delta = max(rss_t - cur_rss, 1e-18)
            score = (abs(cur_not[j]) * cost_mm[j]) / delta   # cost saved per unit variance lost
            if best is None or score > best[0]:
                best = (score, trial, rss_t, beta_t)
        if best is None:
            break
        _, active, cur_rss, cur_beta = best

    notional = np.zeros(len(instruments))
    notional[active] = -cur_beta

    # Drop negligible legs to trim transaction cost without materially hurting variance reduction.
    pruned = np.where(np.abs(notional) < _MIN_NOTIONAL, 0.0, notional)
    if not np.any(pruned):           # never return an all-zero basket for a real portfolio
        keep = int(np.argmax(np.abs(notional)))
        pruned = np.zeros(len(instruments))
        pruned[keep] = notional[keep]
    result = _basket(pruned)
    return result if not result.empty else _default_basket()


# ---------------------------------------------------------------------------------------------------------------------
# Read the test-case JSON (mode / key / portfolio) from stdin. The hedge basket is portfolio-dependent;
# NAV and Risk models are static. If no input is piped, the portfolio is simply empty.
# ---------------------------------------------------------------------------------------------------------------------
_test_input = {}
try:
    if not sys.stdin.isatty():
        _raw = sys.stdin.read()
        if _raw and _raw.strip():
            _test_input = json.loads(_raw)
except Exception:
    _test_input = {}
_portfolio = _test_input.get("portfolio", {}) or {}

#######################################################################################################################
# YOUR CODE ENDS HERE
#######################################################################################################################

#######################################################################################################################
# SUBMISSION FORMAT - ASSIGN YOUR OUTPUT PANDAS DATAFRAMES BELOW
# YOU DO NOT NEED TO MODIFY THE DATAFRAMES FOR TASKS WHICH YOU HAVE NOT SOLVED, WE'LL DO PARTIAL GRADING 
# e.g., IF YOU'VE ONLY SOLVED THE 'NAV' TASK, JUST ASSIGN YOUR SOLUTION FOR IT TO 'nav_model' VARIABLE, AND LEAVE THE  # OTHER TWO UNTOUCHED
#######################################################################################################################

# TASK 1 - NAV ESTIMATION
nav_model = train_nav_model()

# TASK 2 - RISK ESTIMATION
risk_model = train_risk_model()

# TASK 3 - HEDGE CREATION
hedging_basket = compute_hedge(_portfolio)

#######################################################################################################################
# SUBMISSION FORMAT - DO NOT MODIFY/DELETE THE BELOW CODE
#######################################################################################################################

submission = {
  "nav": nav_model.to_csv(),
  "risk": risk_model.to_csv(),
  "hedge": hedging_basket.to_csv() }

final_output = json.dumps( submission )
print( final_output )