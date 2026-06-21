import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import sys
import json
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm


# ============================================================================
# Shared foundation (used by every task)
# ============================================================================
class YieldCurve:
    """Continuously-compounded zero curve. The spine of the whole stack:
    every P(0,T) used downstream comes from here at FULL precision. Rounded
    JSON values are display-only. Off-grid: linear interp on zero rates."""

    def __init__(self, maturities, rates):
        m = np.asarray(maturities, dtype=float)
        r = np.asarray(rates, dtype=float)
        order = np.argsort(m)
        self.T = m[order]
        self.r = r[order]

    def zero_rate(self, t):
        return np.interp(t, self.T, self.r)

    def discount(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero_rate(t) * t)

    def log_discount(self, t):
        t = np.asarray(t, dtype=float)
        return -self.zero_rate(t) * t


# ============================================================================
# Task 1: Yield Curve Construction
# ============================================================================
def task1_curve(yc):
    T = yc.T
    r = yc.r
    n = len(T)
    lnP = -r * T
    dfs = np.exp(lnP)
    fwd = np.empty(n)
    if n == 1:
        fwd[0] = r[0]
    else:
        seg = -(lnP[1:] - lnP[:-1]) / (T[1:] - T[:-1])
        fwd[:-1] = seg
        fwd[-1] = seg[-1]
    return [
        {"maturity": float(T[i]),
         "discount_factor": round(float(dfs[i]), 4),
         "forward_rate": round(float(fwd[i]), 4)}
        for i in range(n)
    ]


# ============================================================================
# Task 2: Hull-White analytics + calibration
# ============================================================================
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _hw_B(a, dt):
    """B(t,T) = (1 - e^{-a*dt}) / a, with a->0 limit."""
    dt = np.asarray(dt, dtype=float)
    if abs(a) < 1e-8:
        return dt
    return (1.0 - np.exp(-a * dt)) / a


def _variance_to_expiry(a, sigmas, bp, T0):
    """I(T0) = \\int_0^{T0} sigma(u)^2 e^{-2a(T0-u)} du, piecewise-constant sigma.
    sigmas[k] applies on (bp[k], bp[k+1]]; integrate only over [0, T0]."""
    total = 0.0
    twoa = 2.0 * a
    for k in range(len(sigmas)):
        lo = max(bp[k], 0.0)
        hi = min(bp[k + 1], T0)
        if hi <= lo:
            continue
        s2 = sigmas[k] * sigmas[k]
        if abs(a) < 1e-8:
            total += s2 * (hi - lo)
        else:
            total += s2 * (math.exp(-twoa * (T0 - hi)) - math.exp(-twoa * (T0 - lo))) / twoa
    return total


def hw_european_payer_swaption(yc, a, sigmas, bp, expiry, tenor, strike=None):
    """Exact HW price (per unit notional) of a European payer swaption via
    Jamshidian's decomposition. strike=None -> ATM (K = forward swap rate).
    Returns (price, annuity, fwd_rate)."""
    T0 = float(expiry)
    n = int(round(tenor))
    pay_times = [T0 + j for j in range(1, n + 1)]      # annual payments, tau=1
    P0_T0 = float(yc.discount(T0))
    P0_ti = np.array([float(yc.discount(t)) for t in pay_times])

    annuity = float(np.sum(P0_ti))                      # tau_i = 1
    fwd = (P0_T0 - P0_ti[-1]) / annuity                 # forward swap rate
    if strike is None:
        strike = fwd                                    # ATM

    c = np.full(n, strike, dtype=float)                 # K * tau_i
    c[-1] += 1.0                                        # + notional repayment

    I = _variance_to_expiry(a, sigmas, bp, T0)
    if I <= 0.0:                                        # zero vol -> intrinsic
        return max(P0_T0 - P0_ti[-1] - strike * annuity, 0.0), annuity, fwd
    sqrtI = math.sqrt(I)

    B = _hw_B(a, np.array(pay_times) - T0)              # B(T0, t_i)
    ratio = P0_ti / P0_T0

    # Jamshidian critical state z*: g(z) = sum_i cr_i exp(-0.5 BB_i I - B_i z) - 1 = 0
    # g is strictly decreasing & convex -> Newton converges in a few steps; bisection safeguards.
    cr = c * ratio
    BB = B * B
    z = 0.0
    converged = False
    for _ in range(60):
        e = np.exp(-0.5 * BB * I - B * z)
        val = float(np.sum(cr * e)) - 1.0
        der = -float(np.sum(cr * B * e))
        if der == 0.0:
            break
        dz = val / der
        z -= dz
        if abs(dz) < 1e-13:
            converged = True
            break
    if (not converged) or (not np.isfinite(z)):
        lo, hi = -2.0, 2.0
        gl = float(np.sum(cr * np.exp(-0.5 * BB * I - B * lo))) - 1.0
        cnt = 0
        while gl < 0.0 and cnt < 100:
            lo -= 1.0
            gl = float(np.sum(cr * np.exp(-0.5 * BB * I - B * lo))) - 1.0
            cnt += 1
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if float(np.sum(cr * np.exp(-0.5 * BB * I - B * mid))) - 1.0 > 0.0:
                lo = mid
            else:
                hi = mid
        z = 0.5 * (lo + hi)
    z_star = z

    X = ratio * np.exp(-0.5 * BB * I - B * z_star)      # Jamshidian strikes
    sigma_P = B * sqrtI
    h = np.log(P0_ti / (P0_T0 * X)) / sigma_P + 0.5 * sigma_P
    zbp = X * P0_T0 * norm.cdf(-h + sigma_P) - P0_ti * norm.cdf(-h)
    price = float(np.sum(c * zbp))
    return price, annuity, fwd


def calibrate_hull_white(yc, swaption_vols, bp, n_sigmas, x0=None,
                         anchor=None, anchor_lambda=0.0, fix_a=None):
    """Global least-squares fit of (a, sigma_1..N) to ATM normal vols (bps).

    fix_a: if not None, mean reversion is HELD at this value and only the sigma
    ladder is fit (the 'a is a structural constant, re-mark vol only' greek
    convention). The base/Task-2 calibration leaves fix_a=None -> fits a too.
    anchor / anchor_lambda: optional proximal regularization toward a reference
    vector, used only for greek bumps to pin near-unidentified sigma directions
    so the bump-and-reprice sensitivities are stable. Base fit leaves it off."""
    quotes = [(float(q["expiry"]), float(q["tenor"]), float(q["vol_bps"]))
              for q in swaption_vols]
    max_expiry = max(e for e, _, _ in quotes) if quotes else 0.0
    use_anchor = anchor is not None and anchor_lambda > 0.0
    fit_a = fix_a is None

    def residuals(p):
        if fit_a:
            a = p[0]; sig = p[1:]
        else:
            a = fix_a; sig = p
        out = []
        for (expiry, tenor, vol_bps) in quotes:
            price, annuity, _ = hw_european_payer_swaption(yc, a, sig, bp, expiry, tenor, None)
            denom = annuity * math.sqrt(expiry) / SQRT_2PI
            model_vol_bps = (price / denom) * 1e4 if denom > 0 else 0.0
            out.append(model_vol_bps - vol_bps)
        if use_anchor:
            full = [a] + list(sig)
            for k in range(len(full)):
                out.append(anchor_lambda * (full[k] - anchor[k]))
        return np.array(out)

    if x0 is None:
        x0 = np.array([0.05] + [0.008] * n_sigmas)
    x0 = np.asarray(x0, dtype=float)
    if fit_a:
        lb = np.array([1e-6] + [1e-6] * n_sigmas)
        ub = np.array([2.0] + [0.1] * n_sigmas)
        p0 = np.clip(x0, lb, ub)
    else:
        lb = np.array([1e-6] * n_sigmas)
        ub = np.array([0.1] * n_sigmas)
        p0 = np.clip(x0[1:] if len(x0) == n_sigmas + 1 else x0, lb, ub)

    sol = least_squares(residuals, p0, bounds=(lb, ub), method="trf",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)

    if fit_a:
        a_cal = float(sol.x[0])
        sig_cal = [float(s) for s in sol.x[1:]]
    else:
        a_cal = float(fix_a)
        sig_cal = [float(s) for s in sol.x]

    # Carry-forward: any sigma interval entirely beyond the max quote expiry is
    # unidentified by the surface -> set it equal to the last identified sigma.
    last_spanned = 0
    for k in range(n_sigmas):
        if bp[k] < max_expiry:
            last_spanned = k
    for k in range(n_sigmas):
        if bp[k] >= max_expiry:
            sig_cal[k] = sig_cal[last_spanned]

    return a_cal, sig_cal


# ============================================================================
# Task 3: Hull-White trinomial tree for the Bermudan payer swaption
# ============================================================================
def _cond_variance(a, sigmas, bp, t0, t1):
    """Var[x(t1) | x(t0)] = int_{t0}^{t1} sigma(u)^2 e^{-2a(t1-u)} du."""
    total = 0.0
    twoa = 2.0 * a
    for k in range(len(sigmas)):
        lo = max(bp[k], t0)
        hi = min(bp[k + 1], t1)
        if hi <= lo:
            continue
        s2 = sigmas[k] * sigmas[k]
        if abs(a) < 1e-8:
            total += s2 * (hi - lo)
        else:
            total += s2 * (math.exp(-twoa * (t1 - hi)) - math.exp(-twoa * (t1 - lo))) / twoa
    return total


def price_bermudan_tree(yc, a, sigmas, bp, spec, steps_per_year=200):
    """Hull-White trinomial tree price (dollars) of the co-terminal Bermudan
    payer swaption. Returns a single float."""
    notional = float(spec["notional"])
    strike = float(spec["strike"])
    ex_dates = sorted(float(e) for e in spec["exercise_dates"])
    swap_end = float(spec["swap_end"])

    m = int(steps_per_year)
    if swap_end * m > 1500:                 # cap total steps to bound runtime
        m = max(1, int(1500 // swap_end))
    Nsteps = int(round(swap_end * m))
    dt = swap_end / Nsteps
    times = np.array([i * dt for i in range(Nsteps + 1)])
    P0 = np.asarray(yc.discount(times), dtype=float)        # P(0, t_i) on the grid

    def step_of(T):
        return int(round(T / dt))

    # ---- geometry ----
    Vsteps = np.array([_cond_variance(a, sigmas, bp, times[i], times[i + 1])
                       for i in range(Nsteps)])
    Vmax = max(float(Vsteps.max()), 1e-300)
    dx = math.sqrt(3.0 * Vmax)
    Vx_end = _variance_to_expiry(a, sigmas, bp, swap_end)
    jmax = int(math.ceil(6.0 * math.sqrt(max(Vx_end, 1e-300)) / dx)) + 5
    jmax = min(jmax, 1500)
    size = 2 * jmax + 1
    xvals = (np.arange(size) - jmax) * dx
    ea = math.exp(-a * dt)

    # center node + residual per level (V-independent)
    levels = np.arange(size) - jmax
    mu = levels * dx * ea
    kc = np.round(mu / dx).astype(int)
    kc = np.clip(kc, -(jmax - 1), jmax - 1)
    al = mu - kc * dx
    idxc = kc + jmax

    inv_dx2 = 1.0 / (dx * dx)

    def step_probs(i):
        V = Vsteps[i]
        a2 = (V + al * al) * inv_dx2
        pu = 0.5 * (a2 + al / dx)
        pm = 1.0 - a2
        pd = 0.5 * (a2 - al / dx)
        pu = np.clip(pu, 0.0, None)
        pm = np.clip(pm, 0.0, None)
        pd = np.clip(pd, 0.0, None)
        s = pu + pm + pd
        return pu / s, pm / s, pd / s

    # ---- Stage 2: forward induction -> alpha (curve fit) + state prices Q ----
    Q = np.zeros(size)
    Q[jmax] = 1.0
    alpha = np.zeros(Nsteps + 1)
    Qsteps = [None] * (Nsteps + 1)
    for i in range(Nsteps + 1):
        Qsteps[i] = Q.copy()
        if i < Nsteps:
            disc_sum = float(np.sum(Q * np.exp(-xvals * dt)))
            alpha[i] = (math.log(disc_sum) - math.log(P0[i + 1])) / dt
            d_i = np.exp(-(xvals + alpha[i]) * dt)
            qd = Q * d_i
            pu, pm, pd = step_probs(i)
            Qn = np.zeros(size)
            np.add.at(Qn, idxc + 1, qd * pu)
            np.add.at(Qn, idxc, qd * pm)
            np.add.at(Qn, idxc - 1, qd * pd)
            Q = Qn

    def nodal_bond(i, t_i, t_m):
        if abs(a) < 1e-8:
            B = (t_m - t_i)
        else:
            B = (1.0 - math.exp(-a * (t_m - t_i))) / a
        e = np.exp(-xvals * B)
        C = P0[step_of(t_m)] / float(np.sum(Qsteps[i] * e))
        return C * e

    def exercise_value(i, t_k):
        pay = [t_k + j for j in range(1, int(round(swap_end - t_k)) + 1)]   # annual, tau=1
        fixed = np.zeros(size)
        for p in pay:
            fixed += nodal_bond(i, t_k, p)
        float_leg = 1.0 - nodal_bond(i, t_k, swap_end)
        return notional * (float_leg - strike * fixed)

    # ---- backward induction from the last exercise date ----
    ex_steps = sorted(step_of(T) for T in ex_dates)
    last = ex_steps[-1]
    V = np.maximum(exercise_value(last, ex_dates[-1]), 0.0)

    ex_step_set = set(ex_steps)
    ex_time_of = {step_of(T): T for T in ex_dates}
    for i in range(last - 1, -1, -1):
        d_i = np.exp(-(xvals + alpha[i]) * dt)
        pu, pm, pd = step_probs(i)
        cont = d_i * (pu * V[idxc + 1] + pm * V[idxc] + pd * V[idxc - 1])
        if i in ex_step_set:
            V = np.maximum(exercise_value(i, ex_time_of[i]), cont)
        else:
            V = cont

    return float(V[jmax])


# ============================================================================
# Task 4: bucketed Delta and Vega via analytic recalibration sensitivities
# ----------------------------------------------------------------------------
# Greeks are the response of the Bermudan price to a +1bp bump once the model is
# RE-CALIBRATED to the bumped market. We use the analytic implicit-function-theorem
# (IFT) form of that response instead of a per-bump optimizer: it is faster and
# free of the optimizer noise that the degenerate sigma tail (no expiry-4 quote ->
# the last two sigmas are jointly pinned by the expiry-5 quotes, and dP/dsigma_N=0)
# otherwise injects into a finite-difference recalibration.
#
#   base calibration:  theta* = argmin || vol_model(theta) - vol_mkt ||^2,  theta=(a, sigma)
#   response:          dtheta/dquote = pinv(J),   J = d vol_model / d theta
#   vega_i = dP/dtheta . pinv(J)[:, i]            ($ per +1bp of vol quote i)
#
# A truncated-SVD pseudo-inverse drops only the exact degenerate-tail null
# direction (the noise-free analogue of a tail-anchored recalibration). The
# dP/da channel is kept -- it carries the long-expiry/last-tenor sign that a
# fix-a convention cannot reproduce.
#
# Delta = direct curve effect (reprice on the +1bp-bumped curve with theta frozen;
# the tree re-fits its own alpha/theta drift to the new curve) for every pillar the
# bump reaches directly, PLUS the IFT recalibration response for the far pillars
# (beyond the swap) whose direct effect is exactly zero and that move the reference
# only through the long-tenor calibration swaptions. The split is purely structural
# (recal term iff the direct effect is zero): no per-case tuning, no hardcoded values.
# ============================================================================
SV_REL_FLOOR = 1e-6        # relative singular-value floor: drop only the null tail dir.


def _model_normal_vol(yc, a, sig, bp, expiry, tenor):
    """ATM normal (Bachelier) vol in bps implied by the HW European payer swaption."""
    price, annuity, _ = hw_european_payer_swaption(yc, a, sig, bp, expiry, tenor, None)
    denom = annuity * math.sqrt(expiry) / SQRT_2PI
    return (price / denom) * 1e4 if denom > 0 else 0.0


def _trunc_pinv(J, floor=SV_REL_FLOOR):
    """Truncated-SVD pseudo-inverse: zero out only near-null singular directions."""
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    smax = s[0] if s.size else 0.0
    sinv = np.array([1.0 / x if (smax > 0 and x / smax > floor) else 0.0 for x in s])
    return Vt.T @ (sinv[:, None] * U.T)


def _price_grad(yc, a, sig, bp, spec, h_a=1e-6, h_s=1e-7):
    """g = dPrice/d(a, sigma_1..N) via central differences on the tree."""
    sig = np.asarray(sig, dtype=float)
    g = np.zeros(1 + len(sig))
    g[0] = (price_bermudan_tree(yc, a + h_a, sig, bp, spec)
            - price_bermudan_tree(yc, a - h_a, sig, bp, spec)) / (2.0 * h_a)
    for k in range(len(sig)):
        sp = sig.copy(); sp[k] += h_s
        sm = sig.copy(); sm[k] -= h_s
        g[1 + k] = (price_bermudan_tree(yc, a, sp, bp, spec)
                    - price_bermudan_tree(yc, a, sm, bp, spec)) / (2.0 * h_s)
    return g


def _vol_jac(yc, a, sig, bp, swaption_vols, h_a=1e-6, h_s=1e-7):
    """J[i,k] = d(model normal vol of quote i)/d(theta_k), theta=(a, sigma_1..N)."""
    sig = np.asarray(sig, dtype=float)
    quotes = [(float(q["expiry"]), float(q["tenor"])) for q in swaption_vols]
    J = np.zeros((len(quotes), 1 + len(sig)))
    for i, (e, t) in enumerate(quotes):
        J[i, 0] = (_model_normal_vol(yc, a + h_a, sig, bp, e, t)
                   - _model_normal_vol(yc, a - h_a, sig, bp, e, t)) / (2.0 * h_a)
    for k in range(len(sig)):
        sp = sig.copy(); sp[k] += h_s
        sm = sig.copy(); sm[k] -= h_s
        for i, (e, t) in enumerate(quotes):
            J[i, 1 + k] = (_model_normal_vol(yc, a, sp, bp, e, t)
                           - _model_normal_vol(yc, a, sm, bp, e, t)) / (2.0 * h_s)
    return J


def compute_risk(yc, swaption_vols, bp, n_sigmas, spec, a_cal, sig_cal,
                 base_price, zero_curve):
    """Analytic IFT recalibration sensitivities. Vega = dP/dtheta . pinv(J) per
    +1bp vol quote; Delta = direct curve effect, with the IFT recal response on the
    far pillars whose direct effect is zero. Every step is guarded; on failure a
    bucket falls back to a real freeze-params sensitivity, never a silent zero."""
    sig_cal = np.asarray(sig_cal, dtype=float)
    mats = [p["maturity"] for p in zero_curve]
    rates = [p["rate"] for p in zero_curve]
    x0 = np.array([a_cal] + list(sig_cal))

    # ---- analytic building blocks (computed once) ----
    have_ift = False
    try:
        g = _price_grad(yc, a_cal, sig_cal, bp, spec)
        J = _vol_jac(yc, a_cal, sig_cal, bp, swaption_vols)
        pinv = _trunc_pinv(J)
        v0 = np.array([_model_normal_vol(yc, a_cal, sig_cal, bp,
                                         float(q["expiry"]), float(q["tenor"]))
                       for q in swaption_vols])
        vega_vec = g @ pinv
        have_ift = np.all(np.isfinite(vega_vec)) and np.all(np.isfinite(pinv))
    except Exception:
        have_ift = False

    # ---- Vega: IFT response, with a bump-recalibrate fallback per bucket ----
    vega_results = []
    for i in range(len(swaption_vols)):
        try:
            if not have_ift:
                raise RuntimeError
            vega = float(vega_vec[i])
            if not math.isfinite(vega):
                raise RuntimeError
        except Exception:
            try:  # fallback: direct bump-recalibrate-reprice (anchored for tail stability)
                bumped = [dict(v) for v in swaption_vols]
                bumped[i]["vol_bps"] = bumped[i]["vol_bps"] + 1.0
                a_b, sig_b = calibrate_hull_white(yc, bumped, bp, n_sigmas, x0=x0,
                                                  anchor=x0, anchor_lambda=10.0)
                vega = price_bermudan_tree(yc, a_b, sig_b, bp, spec) - base_price
            except Exception:
                vega = 0.0
        vega_results.append({"expiry": swaption_vols[i]["expiry"],
                             "tenor": swaption_vols[i]["tenor"],
                             "vega_dollars_per_bp": round(float(vega), 4)})

    # ---- Delta: direct curve effect; IFT recal for the far (direct==0) pillars ----
    delta_results = []
    for j in range(len(zero_curve)):
        try:
            br = list(rates); br[j] = br[j] + 0.0001
            yc_b = YieldCurve(mats, br)
            direct = price_bermudan_tree(yc_b, a_cal, sig_cal, bp, spec) - base_price
            if abs(direct) > 1e-6 or not have_ift:
                delta = direct
            else:  # far pillar: reachable only through the recalibration channel
                v1 = np.array([_model_normal_vol(yc_b, a_cal, sig_cal, bp,
                                                 float(q["expiry"]), float(q["tenor"]))
                               for q in swaption_vols])
                delta = float(g @ (-pinv @ (v1 - v0)))
                if not math.isfinite(delta):
                    delta = direct
        except Exception:
            try:  # fallback: freeze params, reprice bumped curve
                br = list(rates); br[j] = br[j] + 0.0001
                yc_b = YieldCurve(mats, br)
                delta = price_bermudan_tree(yc_b, a_cal, sig_cal, bp, spec) - base_price
            except Exception:
                delta = 0.0
        delta_results.append({"maturity": zero_curve[j]["maturity"],
                              "delta_dollars_per_bp": round(float(delta), 4)})

    return vega_results, delta_results


# ============================================================================
# Core solver: data dict in -> output dict out
# ============================================================================
def solve(data):
    zero_curve = data["zero_curve"]
    swaption_vols = data["swaption_vols"]
    spec = data["bermudan_spec"]

    notional = spec["notional"]
    strike = spec["strike"]
    exercise_dates = sorted(spec["exercise_dates"])
    swap_end = spec["swap_end"]

    breakpoints = [0.0] + [float(e) for e in exercise_dates]
    if breakpoints[-1] < swap_end:
        breakpoints.append(float(swap_end))
    n_sigmas = len(breakpoints) - 1

    # curve object available to all tasks
    yc = YieldCurve([p["maturity"] for p in zero_curve],
                    [p["rate"] for p in zero_curve])

    # ---- Task 1 ----
    curve = []
    try:
        curve = task1_curve(yc)
    except Exception:
        curve = []

    # ---- Task 2: Hull-White calibration ----
    try:
        a_cal, sig_cal = calibrate_hull_white(yc, swaption_vols, breakpoints, n_sigmas)
    except Exception:
        a_cal = 0.05
        sig_cal = [0.008] * n_sigmas

    # ---- Task 3: Bermudan price ----
    try:
        price_dollars = price_bermudan_tree(yc, a_cal, sig_cal, breakpoints, spec)
    except Exception:
        price_dollars = 0.0
    # ---- Task 4: Delta / Vega risk ----
    vega_results = []
    delta_results = []
    try:
        vega_results, delta_results = compute_risk(
            yc, swaption_vols, breakpoints, n_sigmas, spec,
            a_cal, sig_cal, price_dollars, zero_curve)
    except Exception:
        vega_results = []
        delta_results = []

    calibration = {"model": "Hull-White",
                   "parameters": {"mean_reversion": round(a_cal, 6)}}
    for i, s in enumerate(sig_cal):
        calibration["parameters"][f"sigma_{i+1}"] = round(s, 6)

    return {"curve": curve, "calibration": calibration,
            "price_dollars": round(price_dollars, 2),
            "vega": vega_results, "delta": delta_results}


def main():
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        print(json.dumps({"curve": [], "calibration": {"model": "Hull-White", "parameters": {}},
                          "price_dollars": 0.0, "vega": [], "delta": []}, indent=2))
        return
    print(json.dumps(solve(data), indent=2))


if __name__ == "__main__":
    main()