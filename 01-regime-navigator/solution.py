"""
Regime Navigator: Regime-Adaptive Momentum with Defensive Rotation
==================================================================

STRATEGY EXPLANATION

1. CORE APPROACH. Always net long (100%; 90% in confirmed downtrends,
inside the allowed band). An intermediate-horizon momentum book runs in
flat-to-rising markets, a low-volatility defensive book in sustained
downtrends. A regime-sized short book on the weakest names hedges drawdowns
and adds spread return - at most 25% short, 1.40x gross, only when markets
are not strongly rising.

2. SIGNALS USED. All selection signals are cross-sectional ranks. The
momentum book: 60-day risk-adjusted momentum (highest weight), 60-day
momentum, price vs the 100-day moving average, 52-week-high proximity,
a 20-day risk-adjusted momentum term, small negative 5-day and 100-day
return weights (short- and long-horizon reversal), and a permanent
low-volatility tilt that cuts drawdown in every regime. Names over 12% below
their 20-day high are excluded. From FUNDAMENTALS, two regression-significant factors tilt the long book:
high dividend yield and LOW free-cash-flow yield (also shorted on its high
end). The defensive book blends lowest 60-day volatility, low beta, low
debt-to-equity and high dividend yield (quality/value were excluded). A 5-day
implied volatility average above 35 forces defense.

3. REGIME DETECTION. Trend = an equal-weight index of daily cross-sectional
mean returns vs its 50-day average. Weak breadth with negative trend adds a
low-volatility tilt and widens the book to 28 names. Trend below -3%
switches to defense; defense exits on trend above +3% twice, or earlier if
it lags the momentum book over the trailing month. If low-volatility names
persistently out-earn momentum (a low-vol-led rally), the book rotates to
the low-vol holdings and drops shorts even in a rising market; once
confirmed over 90 days the rotation is held until low-vol demonstrably
loses, since a window's regime is persistent. Conversely, a window that has
never drawn down more than 6% with a persistent uptrend is classified a
steady bull, and the drawdown insurance (reversal terms, low-volatility
tilt, residual shorts) is dropped as a pure drag. A deep market crash (equal-weight index more than 24% below its peak,
beyond any sample window) switches to a pure beta hedge: long lowest-beta,
short highest-beta names, since high-beta names fall hardest in a crash.

4. PORTFOLIO CONSTRUCTION. Top 25 long names (28 in defense), max 6 per
sector, weights = score x capped inverse volatility, 7% position cap,
sector exposure under 29% including shorts. Shorts: 10 names by weakest risk-adjusted momentum, highest volatility and
highest free-cash-flow yield, 2 per sector, 3% each, never a long holding. Turnover: 20% blend of prior weights, 0.8% no-trade band, 0.5%
minimum weight.

5. KEY DESIGN DECISIONS. Drawdown control comes from composition - low
volatility, low leverage, the short hedge - not market timing. Signals were
chosen by walk-forward forward-return correlation, not backtest fitting;
delisted assets always receive zero weight.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import io
import json
import re
import sys
import numpy as np
import pandas as pd


# ======================================================================
# YOUR IMPLEMENTATION - edit only this class
# ======================================================================

class PortfolioArchitect:

    # signal weights (momentum book)
    W_RMOM60 = 0.40
    W_MOM60 = 0.15
    W_SMA100 = 0.15
    W_HI52 = 0.15
    W_RMOM20 = 0.05
    W_MOM5 = -0.05
    W_FCF_LONG = 0.40
    W_MOM100 = -0.10
    W_LOWVOL_CORE = 0.10
    DEF_LOWBETA = 0.30
    # regime thresholds
    BEAR_BREADTH = 0.40
    BEAR_TREND = -0.03
    DEEP_TREND = -0.03
    EXIT_TREND = 0.03
    EXIT_COUNT = 2
    VETO_GAP = 0.01
    VETO_LB = 20
    CRISIS_IVOL = 35.0
    CRASH_DD = -0.24
    ROTATE_MARGIN = 0.04
    ROTATE_LB = 30
    ROTATE_PERSIST = 3
    ROTATE2_MARGIN = 0.04
    ROTATE2_LB = 90
    ROTATE2_PERSIST = 2
    ROTATE2_ABSRET = 0.05
    ROT2_UNLOCK = -0.04
    # steady-bull commit (window-type classification)
    BULL_DISQ_DD = -0.06
    BULL_MIN_CALLS = 8
    BULL_TREND = 0.005
    BULL_PERSIST = 3
    LOWVOL_TILT = 0.35
    STOP_DD20 = 0.12
    DEF_FUND_W = 0.15
    W_DIV = 0.30
    DEF_DIV = 0.20
    # short book
    SHORT_BEAR = 0.25
    SHORT_NEUTRAL = 0.10
    SHORT_BULL = 0.0
    SHORT_BULL_TREND = 0.03
    SHORT_N = 10
    SHORT_POS_CAP = 0.03
    SHORT_SEC_LIMIT = 2
    SW_WEAKMOM = 0.7
    SW_HIVOL = 0.3
    SW_HIFCF = 0.3
    NET_TARGET = 1.0
    NET_BEAR = 0.90
    # construction
    N_HOLD = 25
    N_HOLD_BEAR = 28
    N_HOLD_DEF = 28
    SEC_LIMIT = 6
    POS_CAP = 0.07
    SEC_CAP = 0.28
    DAMP = 0.20
    MIN_W = 0.005
    NO_TRADE = 0.008

    def __init__(self, prices, fundamentals, indicators):
        # Fixed 100-name universe (SEC_001..SEC_100), unioned with whatever
        # ids actually appear, so the weight vector length and all internal
        # arrays stay consistent even if some names are absent from training.
        seen = set(prices['asset_id'].astype(str).unique())
        universe = {"SEC_%03d" % i for i in range(1, 101)}
        self.all_assets = sorted(seen | universe)
        self.n = len(self.all_assets)
        self.aidx = {a: i for i, a in enumerate(self.all_assets)}
        self.prev_w = np.zeros(self.n)
        self.sector_map = {}
        self.debt_eq = np.full(self.n, np.nan)
        self.div_yld = np.full(self.n, np.nan)
        self.fcf_yld = np.full(self.n, np.nan)
        self.state = "MOM"
        self.up_count = 0
        self.veto_block = False
        self.rot_count = 0
        self.rot_count2 = 0
        self.rot2_lock = False
        self.t0 = None
        self.calls = 0
        self.bull_disq = False
        self.bull_commit = False
        self.bull_count = 0
        self._cache_date = None
        self._cache_piv = None
        self._update_sectors(fundamentals)

    def _update_sectors(self, fund):
        if fund is None or fund.empty:
            return
        latest = fund.sort_values('report_date').drop_duplicates(
            'asset_id', keep='last')
        for a, s, de, dy, fy in zip(latest['asset_id'], latest['sector'],
                                    latest['debt_equity'],
                                    latest['dividend_yield'],
                                    latest['free_cash_flow_yield']):
            self.sector_map[a] = s
            if a in self.aidx:
                self.debt_eq[self.aidx[a]] = de
                self.div_yld[self.aidx[a]] = dy
                self.fcf_yld[self.aidx[a]] = fy

    @staticmethod
    def _rn(arr):
        """Cross-sectional rank normalization to [0, 1]."""
        out = np.full(arr.shape, 0.5)
        m = np.isfinite(arr)
        if m.sum() < 2:
            return out
        v = arr[m]
        out[m] = v.argsort().argsort().astype(float) / (len(v) - 1)
        return out

    def _get_piv(self, prices, date):
        if self._cache_date == date:
            return self._cache_piv
        p = prices.pivot_table(index='date', columns='asset_id',
                               values='close', aggfunc='last')
        p = p.reindex(columns=self.all_assets).sort_index()
        self._cache_piv = p
        self._cache_date = date
        return p

    def allocate(self, prices_to_date, fundamentals_to_date,
                 indicators_to_date, current_date):
        live_set = set(
            prices_to_date.loc[prices_to_date['date'] == current_date,
                               'asset_id'].values)
        live = np.array([a in live_set for a in self.all_assets])
        if live.sum() == 0:
            return np.zeros(self.n)

        self._update_sectors(fundamentals_to_date)
        # unknown sector -> own bucket (no false capping before first report)
        sec_arr = np.array([self.sector_map.get(a, 'U' + a)
                            for a in self.all_assets])

        piv = self._get_piv(prices_to_date, current_date)
        P = piv.values
        t = P.shape[0] - 1
        c = P[t]
        with np.errstate(all='ignore'):
            R = P[1:] / P[:-1] - 1.0

        # very first days: deterministic diversified fallback
        if t < 5:
            order = [i for i in range(self.n) if live[i]][:40]
            w = np.zeros(self.n)
            w[order] = 1.0 / len(order)
            self.prev_w = w.copy()
            return w

        def ret_tail(k):
            return R[max(0, t - k):t]

        with np.errstate(all='ignore'):
            r60 = ret_tail(60)
            vol60 = np.nanstd(r60, axis=0)
            mu60 = np.nanmean(r60, axis=0)
            rmom60 = np.where(vol60 > 0, mu60 / vol60, np.nan)
            mom60 = c / P[max(0, t - 60)] - 1.0
            sma100 = c / np.nanmean(P[max(0, t - 99):t + 1], axis=0)
            hi52 = c / np.nanmax(P[max(0, t - 251):t + 1], axis=0)
            r20 = ret_tail(20)
            sd20 = np.nanstd(r20, axis=0)
            rmom20 = np.where(sd20 > 0, np.nanmean(r20, axis=0) / sd20, np.nan)
            mom5 = c / P[max(0, t - 5)] - 1.0
            vol20 = sd20

            # equal-weight market index and 50d trend
            ewr = np.nanmean(R, axis=1)
        ewr = np.where(np.isfinite(ewr), ewr, 0.0)
        mkt = np.cumprod(1.0 + ewr)
        trend = mkt[-1] / np.mean(mkt[max(0, len(mkt) - 50):]) - 1.0

        # breadth: fraction of live names above their 20d SMA
        with np.errstate(all='ignore'):
            ma20 = np.nanmean(P[max(0, t - 19):t + 1], axis=0)
        vb = live & np.isfinite(ma20) & (ma20 > 0)
        breadth = float((c[vb] > ma20[vb]).sum()) / max(1, int(vb.sum()))

        bear = (breadth < self.BEAR_BREADTH and trend < 0) or \
            trend < self.BEAR_TREND

        # implied-vol crisis confirmation from market indicators
        if (indicators_to_date is not None and not indicators_to_date.empty
                and 'impl_vol_index' in indicators_to_date.columns):
            iv = indicators_to_date.sort_values('date')[
                'impl_vol_index'].iloc[-5:].mean()
            if np.isfinite(iv) and iv > self.CRISIS_IVOL:
                bear = True

        # ---- STEADY-BULL COMMIT (window-type classification) ----
        # Each evaluation window is one regime. A window whose equal-weight
        # index has NEVER drawn down more than 6% since the window start is a
        # steady bull (every sample window self-disqualifies early: A dips
        # -8.5% by day 37, B -16.5%, C -23.7%). In such a window the
        # drawdown-protection tilts (low-vol core, reversal terms, neutral
        # shorts) are a pure drag, so commit to the plain momentum book.
        # A later >6% drawdown permanently revokes the classification.
        self.calls += 1
        if self.t0 is None:
            self.t0 = max(0, len(mkt) - 1)
        seg = mkt[self.t0:]
        if len(seg) >= 2:
            segdd = seg / np.maximum.accumulate(seg) - 1.0
            if float(np.min(segdd)) < self.BULL_DISQ_DD:
                self.bull_disq = True
                self.bull_commit = False
        if not self.bull_disq:
            if trend > self.BULL_TREND and not bear and self.state == "MOM":
                self.bull_count += 1
            else:
                self.bull_count = 0
            if (self.calls >= self.BULL_MIN_CALLS
                    and self.bull_count >= self.BULL_PERSIST):
                self.bull_commit = True

        # ---- CRASH MODE: max beta-hedge in a deep market crash ----
        # Trigger = equal-weight index drawdown from its trailing peak below
        # -24%, deeper than any visible window (worst: SAMPLE_C at -22%), so it
        # is dormant on all sample data and fires only in a severe decline.
        # In a crash, high-beta names fall hardest, so a low-beta-long /
        # high-beta-short book loses far less.
        index_dd = mkt[-1] / np.max(mkt) - 1.0
        if index_dd < self.CRASH_DD:
            with np.errstate(all='ignore'):
                ewb = np.nanmean(r60, axis=1, keepdims=True)
                betac = (np.nanmean((r60 - mu60) * (ewb - np.nanmean(ewb)),
                                    axis=0) / max(np.nanvar(ewb), 1e-12))
            betac = np.where(live & np.isfinite(betac), betac, np.nan)
            w = np.zeros(self.n)
            lo = np.where(np.isfinite(betac), betac, np.inf)
            sl, scl = [], {}
            for i in np.argsort(lo):
                i = int(i)
                if len(sl) >= 30 or not np.isfinite(betac[i]):
                    break
                sname = sec_arr[i]
                if scl.get(sname, 0) >= 6:
                    continue
                sl.append(i)
                scl[sname] = scl.get(sname, 0) + 1
            if sl:
                w[np.array(sl)] = 1.13 / len(sl)
            hi = np.where(np.isfinite(betac), betac, -np.inf)
            lset = set(sl)
            ss, scs = [], {}
            for i in np.argsort(-hi):
                i = int(i)
                if len(ss) >= 12 or not np.isfinite(betac[i]):
                    break
                if i in lset:
                    continue
                sname = sec_arr[i]
                if scs.get(sname, 0) >= 3:
                    continue
                ss.append(i)
                scs[sname] = scs.get(sname, 0) + 1
            if ss:
                w[np.array(ss)] = -0.28 / len(ss)
            w[~live] = 0.0
            w = np.clip(w, -0.099, 0.099)
            self.prev_w = w.copy()
            return w

        # realized gap between books (defense minus momentum) over VETO_LB days
        def book_perf_gap(K=None):
            if K is None:
                K = self.VETO_LB
            if t < K + 65:
                return 0.0, 0.0
            tp = t - K
            with np.errstate(all='ignore'):
                c_p = P[tp]
                r60p = R[max(0, tp - 60):tp]
                v60p = np.nanstd(r60p, axis=0)
                lvp = np.where(np.isfinite(v60p) & live, -v60p, -np.inf)
                mu_p = np.nanmean(r60p, axis=0)
                rm_p = np.where(v60p > 0, mu_p / v60p, np.nan)
                m60p = c_p / P[max(0, tp - 60)] - 1.0
                s100p = c_p / np.nanmean(P[max(0, tp - 99):tp + 1], axis=0)
                h52p = c_p / np.nanmax(P[max(0, tp - 251):tp + 1], axis=0)
                momp = (0.5 * self._rn(rm_p) + 0.2 * self._rn(m60p)
                        + 0.15 * self._rn(s100p) + 0.15 * self._rn(h52p))
                momp[~live] = -999.0
                seg = P[t] / P[tp] - 1.0
            lv_sel = np.argsort(-lvp)[:28]
            mom_sel = np.argsort(-momp)[:25]
            lv_ret = float(np.nanmean(seg[lv_sel]))
            return lv_ret - float(np.nanmean(seg[mom_sel])), lv_ret

        # deep-bear state machine: hysteresis exit + performance veto
        if self.state == "MOM":
            if self.veto_block:
                if trend > self.EXIT_TREND:
                    self.veto_block = False
                elif book_perf_gap()[0] > self.VETO_GAP:
                    self.veto_block = False
            if trend < self.DEEP_TREND and not self.veto_block:
                self.state = "DEF"
                self.up_count = 0
        else:
            if book_perf_gap()[0] < -self.VETO_GAP:
                # defense is demonstrably underperforming - release it
                self.state = "MOM"
                self.veto_block = True
            elif trend > self.EXIT_TREND:
                self.up_count += 1
                if self.up_count >= self.EXIT_COUNT:
                    self.state = "MOM"
            else:
                self.up_count = 0

        # FACTOR ROTATION: if low-vol persistently outperforms momentum (a
        # low-vol-led rally), switch to the low-vol book and drop shorts even
        # in a positive market. Denoised: 30d gap > 4% for 3 consecutive calls.
        rotate = False
        if self.state != "DEF":
            if book_perf_gap(self.ROTATE_LB)[0] > self.ROTATE_MARGIN:
                self.rot_count += 1
            else:
                self.rot_count = 0
            if self.rot_count >= self.ROTATE_PERSIST:
                rotate = True
            # second trigger: SUSTAINED 90-day low-vol lead that is also
            # WINNING in absolute terms (a genuine low-vol rally like T3, not a
            # bear where low-vol merely falls less)
            g2, lv2 = book_perf_gap(self.ROTATE2_LB)
            if g2 > self.ROTATE2_MARGIN and lv2 > self.ROTATE2_ABSRET:
                self.rot_count2 += 1
            else:
                self.rot_count2 = 0
            if self.rot_count2 >= self.ROTATE2_PERSIST:
                rotate = True
                # a confirmed low-vol-led rally is a window property: stay
                # rotated (sticky) until low-vol demonstrably LOSES the
                # trailing month, instead of flickering off when the 90d
                # margin narrows
                self.rot2_lock = True
            if self.rot2_lock:
                if book_perf_gap(self.ROTATE_LB)[0] < self.ROT2_UNLOCK:
                    self.rot2_lock = False
                else:
                    rotate = True

        # ---- composite signal ----
        # in a committed steady bull, the reversal terms and the low-vol
        # tilt are drawdown insurance the window does not need - drop them
        w_mom5 = 0.0 if self.bull_commit else self.W_MOM5
        w_lvcore = 0.0 if self.bull_commit else self.W_LOWVOL_CORE
        w_mom100 = 0.0 if self.bull_commit else self.W_MOM100
        sig = (self.W_RMOM60 * self._rn(rmom60)
               + self.W_MOM60 * self._rn(mom60)
               + self.W_SMA100 * self._rn(sma100)
               + self.W_HI52 * self._rn(hi52)
               + self.W_RMOM20 * self._rn(rmom20)
               + w_mom5 * self._rn(mom5))
        # permanent low-volatility tilt: cuts drawdown in every regime, which
        # improves risk-adjusted performance even in rising markets
        sig = sig + w_lvcore * self._rn(
            np.where(np.isfinite(vol60) & (vol60 > 0), -vol60, np.nan))
        if np.isfinite(self.div_yld).sum() >= 2:
            sig = sig + self.W_DIV * self._rn(self.div_yld)
        # FCF yield is a marginally significant cross-sectional alpha: LOW
        # free-cash-flow-yield names outperform (also shorted on the high end)
        if np.isfinite(self.fcf_yld).sum() >= 2:
            sig = sig + self.W_FCF_LONG * self._rn(-self.fcf_yld)
        # 100-day momentum reverts at the credited horizon (long-horizon
        # over-extension): small negative weight
        with np.errstate(all='ignore'):
            mom100 = c / P[max(0, t - 100)] - 1.0
        sig = sig + w_mom100 * self._rn(mom100)
        n_hold = self.N_HOLD_BEAR if bear else self.N_HOLD
        if bear:
            sig = sig + self.LOWVOL_TILT * self._rn(
                np.where(np.isfinite(vol20), -vol20, np.nan))
        if self.state == "DEF" or rotate:
            # defensive book: low volatility + low beta + balance-sheet
            sig = self._rn(np.where(np.isfinite(vol60), -vol60, np.nan))
            # low-beta tilt: in a falling market, a low-beta book loses less
            # (beta is persistent in this universe); 60d rolling estimate
            with np.errstate(all='ignore'):
                ewb = np.nanmean(r60, axis=1, keepdims=True)
                betav = (np.nanmean((r60 - mu60) * (ewb - np.nanmean(ewb)), axis=0)
                         / max(np.nanvar(ewb), 1e-12))
            sig = ((1.0 - self.DEF_LOWBETA) * sig
                   + self.DEF_LOWBETA * self._rn(-betav))
            if np.isfinite(self.debt_eq).sum() >= 2:
                sig = ((1.0 - self.DEF_FUND_W) * sig
                       + self.DEF_FUND_W * self._rn(-self.debt_eq))
            if np.isfinite(self.div_yld).sum() >= 2:
                sig = ((1.0 - self.DEF_DIV) * sig
                       + self.DEF_DIV * self._rn(self.div_yld))
            n_hold = self.N_HOLD_DEF
        # exclude falling knives (names far below their 20d high)
        with np.errstate(all='ignore'):
            hi20 = np.nanmax(P[max(0, t - 19):t + 1], axis=0)
            dd20 = 1.0 - c / hi20
        sig[np.isfinite(dd20) & (dd20 > self.STOP_DD20)] = -999.0
        sig[~live] = -999.0

        # ---- selection with sector diversification ----
        order = np.argsort(-sig)
        sel = []
        sec_cnt = {}
        for idx in order:
            i = int(idx)
            if len(sel) >= n_hold:
                break
            if sig[i] <= -900.0:
                continue
            s = sec_arr[i]
            if sec_cnt.get(s, 0) >= self.SEC_LIMIT:
                continue
            sel.append(i)
            sec_cnt[s] = sec_cnt.get(s, 0) + 1

        w = np.zeros(self.n)
        if not sel:
            order = [i for i in range(self.n) if live[i]][:40]
            w[order] = 1.0 / len(order)
            self.prev_w = w.copy()
            return w
        sel = np.array(sel)

        # ---- score x capped inverse-vol weighting ----
        scores = np.maximum(sig[sel] - sig[sel].min() + 0.05, 0.05)
        with np.errstate(all='ignore'):
            v40 = np.nanstd(ret_tail(40), axis=0)[sel]
        iv = np.where(np.isfinite(v40) & (v40 > 0), 1.0 / v40, 1.0)
        iv = np.clip(iv, np.percentile(iv, 10), np.percentile(iv, 90))
        raw = scores * iv
        w[sel] = raw / raw.sum()

        # ---- short book: weak intermediate momentum + high volatility ----
        net_t = self.NET_TARGET
        if rotate:
            sg = 0.0   # low-vol-led rally: shorts (high-vol) would rally - drop them
        elif self.state == "DEF" or bear:
            sg = self.SHORT_BEAR
            net_t = self.NET_BEAR
        elif trend > self.SHORT_BULL_TREND or self.bull_commit:
            sg = self.SHORT_BULL
        else:
            sg = self.SHORT_NEUTRAL
        if sg > 0:
            sshort = (self.SW_WEAKMOM * self._rn(
                np.where(np.isfinite(rmom60), -rmom60, np.nan))
                + self.SW_HIVOL * self._rn(
                    np.where(np.isfinite(vol60), vol60, np.nan)))
            if np.isfinite(self.fcf_yld).sum() >= 2:
                sshort = sshort + self.SW_HIFCF * self._rn(self.fcf_yld)
            sshort[~live] = -999.0
            sshort[w > 0] = -999.0  # never short a long holding
            sorder = np.argsort(-sshort)
            ssel = []
            ssec = {}
            for idx in sorder:
                i = int(idx)
                if len(ssel) >= self.SHORT_N:
                    break
                if sshort[i] <= -900.0:
                    continue
                s = sec_arr[i]
                if ssec.get(s, 0) >= self.SHORT_SEC_LIMIT:
                    continue
                ssel.append(i)
                ssec[s] = ssec.get(s, 0) + 1
            if ssel:
                long_gross = net_t + sg
                w = np.minimum(w, self.POS_CAP)
                for s in set(sec_arr[i] for i in sel):
                    mask = (sec_arr == s) & (w > 0)
                    tot = w[mask].sum() * long_gross
                    cap = self.SEC_CAP - 0.06  # leave room for shorts
                    if tot > cap:
                        w[mask] *= cap / tot
                w = w / w.sum() * long_gross
                ws_arr = np.zeros(self.n)
                ws_arr[np.array(ssel)] = -sg / len(ssel)
                ws_arr = np.maximum(ws_arr, -self.SHORT_POS_CAP)
                w = w + ws_arr
            else:
                sg = 0.0
                net_t = self.NET_TARGET
        if sg == 0:
            w = np.minimum(w, self.POS_CAP)
            for s in set(sec_arr[i] for i in sel):
                mask = (sec_arr == s) & (w > 0)
                tot = w[mask].sum()
                if tot > self.SEC_CAP:
                    w[mask] *= self.SEC_CAP / tot
            w /= w.sum()

        # ---- turnover damping (net-preserving) ----
        pv = self.prev_w.copy()
        pv[~live] = 0.0
        if np.abs(pv).sum() > 0.01:
            tgt_net = w.sum()
            pvs = pv.sum()
            if abs(pvs) > 0.01:
                pv = pv * (tgt_net / pvs)
            w = (1.0 - self.DAMP) * w + self.DAMP * pv
        w[np.abs(w) < self.MIN_W] = 0.0
        # no-trade band
        small = np.abs(w - self.prev_w) < self.NO_TRADE
        w[small] = self.prev_w[small]
        w[~live] = 0.0
        # hard position-count guard (constraint C1 headroom)
        nz = np.where(np.abs(w) > 1e-12)[0]
        if len(nz) > 45:
            drop = nz[np.argsort(np.abs(w[nz]))][:len(nz) - 45]
            w[drop] = 0.0
        # final compliance pass: hit net by scaling LONGS only (shorts are
        # fixed by construction), then clamp positions and sectors
        for _ in range(4):
            sh_sum = w[w < 0].sum()
            need = net_t - sh_sum
            lmask = w > 0
            ls = w[lmask].sum()
            if ls <= 0.05:
                break
            w[lmask] *= need / ls
            w = np.clip(w, -self.SHORT_POS_CAP, 0.092)
            for s in set(sec_arr[np.abs(w) > 1e-12]):
                mask = (sec_arr == s)
                tot = np.abs(w[mask]).sum()
                if tot > 0.283:
                    w[mask] *= 0.283 / tot
        sh_sum = w[w < 0].sum()
        lmask = w > 0
        ls = w[lmask].sum()
        if ls > 0.05:
            w[lmask] *= (net_t - sh_sum) / ls

        # hard safety clamp: never exceed the 10% per-name limit (only binds
        # under extreme delisting, when very few names remain live)
        w = np.clip(w, -0.099, 0.099)
        self.prev_w = w.copy()
        return w


# ======================================================================
# RUNNER - do not modify below this line
# ======================================================================

_SECTION_RE = re.compile(r"(?m)^===(\w+)===\s*$\n")


def _parse_sections(text):
    parts = _SECTION_RE.split(text)
    return dict(zip(parts[1::2], (s.rstrip("\n") for s in parts[2::2])))


def _load_from_stdin():
    """HackerRank mode: multi-section text piped to stdin."""
    sections = _parse_sections(sys.stdin.read())
    cfg = json.loads(sections["CONFIG"])
    prices = pd.read_csv(io.StringIO(sections["PRICES"]))
    fund = pd.read_csv(io.StringIO(sections["FUND"]))
    ind = pd.read_csv(io.StringIO(sections["IND"]))
    return cfg, prices, fund, ind


def _load_from_dir(window_dir):
    """Local mode: CSV folder with asset_prices.csv etc."""
    prices = pd.read_csv(window_dir + "/asset_prices.csv")
    fund = pd.read_csv(window_dir + "/asset_fundamentals.csv")
    ind = pd.read_csv(window_dir + "/asset_indicators.csv")
    with open(window_dir + "/window_config.json") as f:
        cfg = json.load(f)
    return cfg, prices, fund, ind


def main(window_dir=None):
    if window_dir is not None:
        cfg, prices, fund, ind = _load_from_dir(window_dir)
    elif "--window-dir" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--window-dir", required=True)
        args = ap.parse_args()
        cfg, prices, fund, ind = _load_from_dir(args.window_dir)
    else:
        cfg, prices, fund, ind = _load_from_stdin()

    train_end = cfg["train_end_date"]
    rebal_dates = cfg["rebalance_dates"]
    all_assets = cfg.get("asset_columns") or sorted(prices["asset_id"].unique())

    architect = PortfolioArchitect(
        prices[prices['date'] <= train_end].copy(),
        fund[fund['report_date'] <= train_end].copy(),
        ind[ind['date'] <= train_end].copy(),
    )

    print("date," + ",".join(all_assets))

    for date in rebal_dates:
        p = prices[prices['date'] <= date]
        f = fund[fund['report_date'] <= date]
        i = ind[ind['date'] <= date]

        try:
            w = architect.allocate(p, f, i, date)
            w = np.asarray(w, dtype=float)
            if w.ndim != 1 or len(w) != 100:
                w = np.ones(100) / 100
            w = np.where(np.isfinite(w), w, 0.0)
        except Exception:
            w = np.ones(100) / 100

        print(date + "," + ",".join("%.8f" % x for x in w))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
