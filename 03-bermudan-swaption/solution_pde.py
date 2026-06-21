import sys, json, copy
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm


###############################################################
#  MARKET CURVE
###############################################################

class MarketCurve:

    def __init__(self, raw_pillars):
        ordered     = sorted(raw_pillars, key=lambda r: r['maturity'])
        self._tenors = np.array([r['maturity'] for r in ordered])
        self._zeros  = np.array([r['rate']     for r in ordered])
        self._raw    = ordered

    def zero(self, t):
        return float(np.interp(t, self._tenors, self._zeros))

    def df(self, t):
        if t < 0:
            return 1.0
        r = self.zero(t)
        # clamp so exp(-r*t) never underflows to 0
        return float(np.exp(-min(r * t, 700.0)))

    def fwd(self, t):
        if t < 1e-12:
            return self.zero(0.0)
        r   = self.zero(t)
        idx = int(np.searchsorted(self._tenors, t))
        if idx == 0 or idx >= len(self._tenors):
            t0, t1 = self._tenors[-2], self._tenors[-1]
            r0, r1 = self._zeros[-2],  self._zeros[-1]
        else:
            t0, t1 = self._tenors[idx-1], self._tenors[idx]
            r0, r1 = self._zeros[idx-1],  self._zeros[idx]
        # FIX: guard identical maturities (duplicate pillars)
        dt = t1 - t0
        grad = (r1 - r0) / dt if abs(dt) > 1e-14 else 0.0
        return r + t * grad

    def to_records(self):
        rows = []
        n    = len(self._raw)
        for k, node in enumerate(self._raw):
            tau = node['maturity']
            p   = self.df(tau)
            if k < n - 1:
                tau2 = self._raw[k+1]['maturity']
                p2   = self.df(tau2)
                dtau = tau2 - tau
                # FIX: both discount factors are clamped >0; still guard dtau
                inst_fwd = (-(np.log(p2) - np.log(p)) / dtau
                            if abs(dtau) > 1e-14 else 0.0)
            else:
                tau0 = self._raw[k-1]['maturity']
                p0   = self.df(tau0)
                dtau = tau - tau0
                inst_fwd = (-(np.log(p) - np.log(p0)) / dtau
                            if abs(dtau) > 1e-14 else 0.0)
            rows.append({
                'maturity':       tau,
                'discount_factor': round(p, 4),
                'forward_rate':    round(inst_fwd, 4)
            })
        return rows


###############################################################
#  ONE-FACTOR SHORT-RATE MODEL  (Hull-White)
###############################################################

class OneFactorModel:

    def __init__(self, speed, vols, pillars):
        self.speed   = float(speed)
        self.vols    = list(vols)
        self.pillars = list(pillars)

    def _risk_factor(self, from_t, to_t):
        k   = self.speed
        tau = to_t - from_t
        if abs(k) < 1e-10:
            return tau
        return (1.0 - np.exp(-k * tau)) / k

    def _bucket_vol(self, t):
        for j in range(len(self.pillars) - 1):
            if self.pillars[j] <= t < self.pillars[j+1]:
                return self.vols[j]
        return self.vols[-1]

    def short_rate_variance(self, horizon):
        acc = 0.0
        k   = self.speed
        for j, sig in enumerate(self.vols):
            lo = self.pillars[j]
            hi = (self.pillars[j+1]
                  if j + 1 < len(self.pillars) else horizon)
            hi = min(hi, horizon)
            if lo >= horizon:
                break
            if abs(k) < 1e-10:
                acc += sig**2 * (hi - lo)
            else:
                acc += (sig**2 / (2*k) *
                        (np.exp(-2*k*(horizon-hi)) -
                         np.exp(-2*k*(horizon-lo))))
        return max(acc, 0.0)

    def convexity_adjustment(self, horizon):
        # HW deterministic shift phi(t) = alpha(t) - f(0,t), i.e. the gap
        # between the curve-implied forward and the model short rate.
        #   phi(t) = (1/a) * sum_k sigma_k^2 * INT (e^{-a(t-s)} - e^{-2a(t-s)}) ds
        # which collapses to (sigma^2 / 2a^2)(1 - e^{-a t})^2 for constant sigma.
        acc = 0.0
        k   = self.speed
        for j, sig in enumerate(self.vols):
            lo = self.pillars[j]
            hi = (self.pillars[j+1]
                  if j + 1 < len(self.pillars) else horizon)
            hi = min(hi, horizon)
            if lo >= horizon:
                break
            if abs(k) < 1e-10:
                # a -> 0 limit: phi(t) = sigma^2 * INT (t - s) ds
                acc += sig**2 * ((horizon-lo)**2 - (horizon-hi)**2) / 2.0
            else:
                e1_hi = np.exp(-k*(horizon-hi)); e1_lo = np.exp(-k*(horizon-lo))
                e2_hi = np.exp(-2*k*(horizon-hi)); e2_lo = np.exp(-2*k*(horizon-lo))
                acc += (sig**2 / k) * ((e1_hi - e1_lo)/k
                                       - (e2_hi - e2_lo)/(2*k))
        return max(acc, 0.0)

    def jamshidian_price(self, expiry, final_pay, atm, mkt):
        schedule  = np.arange(expiry + 1, final_pay + 1)
        cashflows = np.full(len(schedule), atm)
        cashflows[-1] += 1.0
        var_e     = self.short_rate_variance(expiry)
        df_expiry = mkt.df(expiry)

        # FIX: if df_expiry == 0 we cannot price  return large penalty
        if df_expiry < 1e-14:
            return 1e6

        def _swap_at_x(x):
            total = 0.0
            for i, tk in enumerate(schedule):
                b       = self._risk_factor(expiry, tk)
                df_tk    = mkt.df(tk)
                fwd_bond = (df_tk / df_expiry) * np.exp(-0.5*b**2*var_e - b*x)
                total   += cashflows[i] * fwd_bond
            return total - 1.0

        x_star = None
        for lo, hi in [(-2,2), (-10,10), (-30,30), (-100,100)]:
            if _swap_at_x(lo) * _swap_at_x(hi) < 0:
                x_star = brentq(_swap_at_x, lo, hi)
                break
        if x_star is None:
            return 1e6

        value = 0.0
        for i, tk in enumerate(schedule):
            b    = self._risk_factor(expiry, tk)
            df_tk = mkt.df(tk)
            fwd  = (df_tk / df_expiry) * np.exp(-0.5*b**2*var_e - b*x_star)
            vol2 = b**2 * var_e
            if vol2 < 1e-14:         # already guarded, made explicit
                continue
            v    = np.sqrt(vol2)
            # FIX: guard log argument  fwd * df_expiry must be > 0
            denom = fwd * df_expiry
            if denom < 1e-14 or df_tk < 1e-14:
                continue
            d1 = np.log(df_tk / denom) / v + 0.5*v
            d2 = d1 - v
            bond_put = (fwd * df_expiry * norm.cdf(-d2) -
                        df_tk * norm.cdf(-d1))
            value += cashflows[i] * bond_put
        return value

    def to_normal_vol(self, px, expiry, final_pay, mkt):
        annuity = sum(mkt.df(tk)
                      for tk in np.arange(expiry+1, final_pay+1))
        # FIX: guard annuity=0 and expiry<=0
        if annuity < 1e-14 or expiry < 1e-14:
            return 0.0
        return max(px * np.sqrt(2*np.pi) / (annuity * np.sqrt(expiry)),
                   0.0)


###############################################################
#  CALIBRATION
###############################################################

def strip_to_swaption_grid(quotes, time_buckets, mkt,
                           starting_guess=None):
    n_buckets = len(time_buckets) - 1
    x0  = (starting_guess if starting_guess is not None
           else [0.05] + [0.008]*n_buckets)
    # a-lower-bound 1e-6 (not 1e-4): a flat vol surface leaves mean reversion
    # unidentified, so it pins to the bound; the reference uses 1e-6.
    box = [(1e-6, 1.0)] + [(1e-4, 0.2)]*n_buckets

    def _rms(params):
        mdl  = OneFactorModel(params[0], params[1:], time_buckets)
        errs = []
        for q in quotes:
            exp, tenor = q['expiry'], q['tenor']
            end   = exp + tenor
            dates = np.arange(exp+1, end+1)
            ann   = sum(mkt.df(tk) for tk in dates)
            # FIX: skip quote if annuity is degenerate
            if ann < 1e-14:
                errs.append(1e6)
                continue
            atm  = (mkt.df(exp) - mkt.df(end)) / ann
            px   = mdl.jamshidian_price(exp, end, atm, mkt)
            nvol = mdl.to_normal_vol(px, exp, end, mkt) * 10_000
            errs.append((nvol - q['vol_bps'])**2)
        return np.sqrt(np.mean(errs)) if errs else 1e6

    sol = minimize(_rms, x0, method='L-BFGS-B', bounds=box,
                   options={'ftol':1e-12, 'gtol':1e-12})
    return sol.x


###############################################################
#  PDE PRICER  (Crank-Nicolson on x-grid)
###############################################################

def _dense_time_axis(call_dates, maturity, step=0.02):
    # FIX: guarantee step > 0 so the while loop always terminates
    step = max(step, 1e-6)
    mandatory = set(np.round(call_dates, 10)) | {0.0, maturity}
    t = maturity
    while t > step:
        t -= step
        mandatory.add(round(t, 10))
    mandatory.add(0.0)
    return np.sort(np.array(list(mandatory)))[::-1]


def _solve_tridiagonal(sub, diag, sup, rhs):
    n  = len(diag)
    cc = np.zeros(n)
    dd = np.zeros(n)
    # FIX: guard first pivot  fall back to numpy if zero
    if abs(diag[0]) < 1e-15:
        M = (np.diag(diag) +
             np.diag(sup[1:], 1) +
             np.diag(sub[1:], -1))
        return np.linalg.solve(M, rhs)
    cc[0] = sup[0] / diag[0]
    dd[0] = rhs[0] / diag[0]
    for i in range(1, n):
        pivot = diag[i] - sub[i]*cc[i-1]
        if abs(pivot) < 1e-15:
            M = (np.diag(diag) +
                 np.diag(sup[1:], 1) +
                 np.diag(sub[1:], -1))
            return np.linalg.solve(M, rhs)
        cc[i] = sup[i]/pivot if i < n-1 else 0.0
        dd[i] = (rhs[i] - sub[i]*dd[i-1]) / pivot
    out = np.zeros(n)
    out[-1] = dd[-1]
    for i in range(n-2, -1, -1):
        out[i] = dd[i] - cc[i]*out[i+1]
    return out


def value_bermudan_receiver(mdl, contract, mkt):
    maturity   = float(contract['swap_end'])
    fixed_cpn  = float(contract['strike'])
    notional   = float(contract['notional'])
    call_dates = np.sort(contract['exercise_dates'])
    pay_freq   = {'annual':1.0,'semi-annual':0.5,
                  'quarterly':0.25}.get(
                      contract.get('payment_frequency','annual'), 1.0)

    #  spatial grid 
    peak_var  = mdl.short_rate_variance(maturity)
    half_span = 6.0*np.sqrt(max(peak_var, 1e-14)) + 1e-8
    N         = 200
    # FIX: guard h=0  half_span is always > 1e-8 above
    h         = 2.0*half_span / N        # safe: N=200, half_span>0
    nodes     = np.linspace(-half_span, half_span, N+1)

    t_axis  = _dense_time_axis(call_dates, maturity)
    ex_set  = set(np.round(call_dates, 8))

    fwd_vals = np.array([mkt.fwd(t)                for t in t_axis])
    adj_vals = np.array([mdl.convexity_adjustment(t) for t in t_axis])
    drift    = fwd_vals + adj_vals
    loc_vol  = np.array([mdl._bucket_vol(t)          for t in t_axis])

    def _intrinsic(step_idx):
        t     = t_axis[step_idx]
        var   = mdl.short_rate_variance(t)
        df_t   = mkt.df(t)
        # FIX: if df_t is zero we cannot compute forward bonds
        if df_t < 1e-14:
            return np.zeros(N+1)
        pay_leg = np.arange(t + pay_freq, maturity + 1e-10, pay_freq)
        if len(pay_leg) == 0:
            return np.zeros(N+1)
        floater = np.zeros(N+1)
        for tk in pay_leg:
            b     = mdl._risk_factor(t, tk)
            fwd_p  = (mkt.df(tk)/df_t) * np.exp(-0.5*b**2*var - b*nodes)
            floater += fixed_cpn * pay_freq * fwd_p
        b_mat  = mdl._risk_factor(t, maturity)
        df_mat  = mkt.df(maturity)
        redemp  = (df_mat/df_t) * np.exp(-0.5*b_mat**2*var - b_mat*nodes)
        floater += redemp
        return np.maximum(1.0 - floater, 0.0)

    grid = np.zeros(N+1)

    for step in range(len(t_axis) - 1):
        t_late  = t_axis[step]
        t_early = t_axis[step+1]
        tau     = t_late - t_early

        s_late,  s_early  = loc_vol[step],  loc_vol[step+1]
        mu_late, mu_early = drift[step],    drift[step+1]

        sub   = np.zeros(N+1)
        diag = np.ones(N+1)
        sup   = np.zeros(N+1)
        rhs   = np.zeros(N+1)

        h2 = h * h        # pre-computed; h > 0 guaranteed above

        for i in range(1, N):
            xi = nodes[i]
            k  = mdl.speed

            D2_l = 0.5 * s_late**2
            D1_l = -k * xi
            R_l  = -(mu_late + xi)
            # CN explicit side (I + 0.5*tau*L) applied to the later slice
            ql   =  0.5*tau*(D2_l/h2 - D1_l/(2*h))
            pl   =  1.0 + 0.5*tau*(-2*D2_l/h2 + R_l)
            rl   =  0.5*tau*(D2_l/h2 + D1_l/(2*h))
            rhs[i] = ql*grid[i-1] + pl*grid[i] + rl*grid[i+1]

            # CN implicit side (I - 0.5*tau*L) solved for the earlier slice
            D2_e = 0.5 * s_early**2
            D1_e = -k * xi
            R_e  = -(mu_early + xi)
            sub[i]  = -0.5*tau*(D2_e/h2 - D1_e/(2*h))
            diag[i] =  1.0 - 0.5*tau*(-2*D2_e/h2 + R_e)
            sup[i]  = -0.5*tau*(D2_e/h2 + D1_e/(2*h))

        iv_early   = _intrinsic(step+1)
        diag[0]    = 1.0; sup[0]  = 0.0; rhs[0]  = 0.0
        diag[N]    = 1.0; sub[N]  = 0.0; rhs[N]  = iv_early[N]

        grid = _solve_tridiagonal(sub, diag, sup, rhs)

        if np.round(t_early, 8) in ex_set:
            grid = np.maximum(grid, iv_early)

    return grid[N//2] * notional


###############################################################
#  SENSITIVITY ENGINE
###############################################################

def _bump_vol(quotes, target_q, shift=1.0):
    out = copy.deepcopy(quotes)
    for item in out:
        if (item['expiry'] == target_q['expiry'] and
                item['tenor'] == target_q['tenor']):
            item['vol_bps'] += shift
    return out


def _bump_curve(raw_nodes, idx, shift=0.0001):
    out = copy.deepcopy(raw_nodes)
    out[idx]['rate'] += shift
    return out


def compute_sensitivities(base_px, fitted, buckets,
                          contract, base_mkt,
                          quotes, raw_nodes):
    vega_ladder  = []
    delta_ladder = []

    for q in quotes:
        shifted_q = _bump_vol(quotes, q)
        p2        = strip_to_swaption_grid(shifted_q, buckets,
                                            base_mkt,
                                            starting_guess=fitted)
        m2        = OneFactorModel(p2[0], p2[1:], buckets)
        vega_ladder.append({
            'expiry': q['expiry'],
            'tenor':  q['tenor'],
            'vega_dollars_per_bp': round(
                value_bermudan_receiver(m2, contract, base_mkt)
                - base_px, 2)
        })

    for i, node in enumerate(raw_nodes):
        shifted_nodes = _bump_curve(raw_nodes, i)
        shifted_mkt   = MarketCurve(shifted_nodes)
        p2            = strip_to_swaption_grid(quotes, buckets,
                                               shifted_mkt,
                                               starting_guess=fitted)
        m2            = OneFactorModel(p2[0], p2[1:], buckets)
        delta_ladder.append({
            'maturity': node['maturity'],
            'delta_dollars_per_bp': round(
                value_bermudan_receiver(m2, contract, shifted_mkt)
                - base_px, 2)
        })

    return vega_ladder, delta_ladder


###############################################################
#  ENTRY POINT
###############################################################

def main():
    payload   = json.load(sys.stdin)
    raw_nodes = sorted(payload['zero_curve'],
                       key=lambda z: z['maturity'])
    quotes    = payload['swaption_vols']
    contract  = payload['bermudan_spec']

    mkt    = MarketCurve(raw_nodes)
    ex_dates = sorted(contract['exercise_dates'])
    buckets  = [0.0] + ex_dates + [float(contract['swap_end'])]

    fitted  = strip_to_swaption_grid(quotes, buckets, mkt)
    mdl     = OneFactorModel(fitted[0], fitted[1:], buckets)
    base_px = value_bermudan_receiver(mdl, contract, mkt)

    vega_ladder, delta_ladder = compute_sensitivities(
        base_px, fitted, buckets, contract,
        mkt, quotes, raw_nodes
    )

    sig_map = {f'sigma_{i+1}': round(s, 6)
               for i, s in enumerate(fitted[1:])}

    print(json.dumps({
        'curve': mkt.to_records(),
        'calibration': {
            'model': 'Hull-White',
            'parameters': {
                'mean_reversion': round(fitted[0], 6),
                **sig_map
            }
        },
        'price_dollars': round(base_px, 2),
        'vega':  vega_ladder,
        'delta': delta_ladder
    }, indent=2))


if __name__ == '__main__':
    main()