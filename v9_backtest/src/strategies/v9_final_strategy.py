from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V9FinalConfig:
    ma_short: int = 105
    ma_long: int = 210
    ma_long_slope: int = 63
    mom3: int = 63
    mom6: int = 126
    rs6: int = 126
    vol12: int = 252
    kelly_mean: int = 756
    kelly_vol: int = 252
    min_kelly_obs: int = 252
    crisis_band: tuple[float, float] = (0.15, 0.35)
    riskoff_band: tuple[float, float] = (0.45, 0.65)
    neutral_band: tuple[float, float] = (0.70, 1.00)
    riskon_band: tuple[float, float] = (0.95, 1.15)
    strong_band: tuple[float, float] = (1.15, 1.40)
    frac_crisis: float = 0.25
    frac_riskoff: float = 1/3
    frac_neutral: float = 0.50
    frac_riskon: float = 0.60
    frac_strong: float = 0.75
    long_run_excess_prior: float = 0.06
    vol_floor: float = 0.08
    defense_strength: float = 0.85
    recovery_dd: float = -0.20
    vol_brake_1: float = 0.30
    vol_brake_2: float = 0.35
    one_way_ma: int = 175
    fast_ma: int = 20
    mid_ma: int = 50
    one_way_mom: int = 63
    one_way_rs: int = 63
    one_way_vol: int = 20
    one_way_vol_entry: float = 0.30
    one_way_dd_entry: float = -0.10
    tqqq_emergency_vol: float = 0.35
    tqqq_emergency_dd: float = -0.12
    strong_trend_boost: float = 0.11
    final_exposure_cap: float = 1.45
    tqqq_capital_cap: float = 0.10
    qqq_strong: float = 0.70
    qqq_strong_weak_rs: float = 0.55
    qqq_riskon: float = 0.65
    qqq_riskon_weak_rs: float = 0.50
    qqq_neutral: float = 0.55
    qqq_neutral_weak_rs: float = 0.45
    qqq_riskoff: float = 0.35
    qqq_crisis: float = 0.30
    weekly_bucket: str = "W-FRI"


class V9FinalStrategy:
    def __init__(self, config: Optional[V9FinalConfig] = None):
        self.c = config or V9FinalConfig()

    @staticmethod
    def _ann_mean(x: pd.Series, window: int, min_periods: int) -> pd.Series:
        return x.rolling(window, min_periods=min_periods).mean() * 252.0

    @staticmethod
    def _ann_vol(x: pd.Series, window: int, min_periods: int) -> pd.Series:
        return x.rolling(window, min_periods=min_periods).std(ddof=1) * np.sqrt(252.0)

    def features(self, prices: pd.DataFrame) -> pd.DataFrame:
        c = self.c
        px = prices.sort_index().astype(float)
        q, s = px["QQQ"], px["SPY"]
        ret_q = q.pct_change()
        ret_s = s.pct_change()
        ret_cash = px["CASH"].pct_change().fillna(0.0)
        f = pd.DataFrame(index=px.index)
        f["ma105"] = q.rolling(c.ma_short, min_periods=c.ma_short).mean()
        f["ma210"] = q.rolling(c.ma_long, min_periods=c.ma_long).mean()
        f["spy_ma210"] = s.rolling(c.ma_long, min_periods=c.ma_long).mean()
        f["ma210_slope"] = f["ma210"] / f["ma210"].shift(c.ma_long_slope) - 1.0
        f["mom63"] = q / q.shift(c.mom3) - 1.0
        f["mom126"] = q / q.shift(c.mom6) - 1.0
        f["rs126"] = (q / s) / (q / s).shift(c.rs6) - 1.0
        f["drawdown"] = q / q.cummax() - 1.0
        f["q_vol252"] = self._ann_vol(ret_q, c.vol12, max(63, c.vol12 // 2))
        basket_ret = 0.60 * ret_q + 0.40 * ret_s
        excess = basket_ret - ret_cash
        f["basket_mu"] = self._ann_mean(excess, c.kelly_mean, c.min_kelly_obs)
        f["basket_vol"] = self._ann_vol(basket_ret, c.kelly_vol, max(126, c.kelly_vol // 2))
        f["ma20"] = q.rolling(c.fast_ma, min_periods=c.fast_ma).mean()
        f["ma50"] = q.rolling(c.mid_ma, min_periods=c.mid_ma).mean()
        f["ma175"] = q.rolling(c.one_way_ma, min_periods=c.one_way_ma).mean()
        f["ma175_slope"] = f["ma175"] / f["ma175"].shift(20) - 1.0
        f["mom_oneway"] = q / q.shift(c.one_way_mom) - 1.0
        f["rs_oneway"] = (q / s) / (q / s).shift(c.one_way_rs) - 1.0
        f["vol20"] = self._ann_vol(ret_q, c.one_way_vol, c.one_way_vol)
        return f

    def _regime(self, prices: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
        q, s = prices["QQQ"], prices["SPY"]
        score = ((q > f["ma210"]).astype(int) + (s > f["spy_ma210"]).astype(int) +
                 (f["ma210_slope"] > 0).astype(int) + (f["mom126"] > 0).astype(int))
        crisis_cond = ((q < f["ma105"]) & (q < f["ma210"]) & (f["mom126"] < 0) & (f["drawdown"] <= -0.20))
        r = pd.Series("Neutral", index=prices.index, dtype="object")
        r[score <= 0] = "Crisis"
        r[score == 1] = "RiskOff"
        r[score == 2] = "Neutral"
        r[score == 3] = "RiskOn"
        r[score >= 4] = "StrongRiskOn"
        r[crisis_cond] = "Crisis"
        return r

    def _raw_exposure(self, regime: pd.Series, f: pd.DataFrame) -> pd.Series:
        c = self.c
        band = {"Crisis": c.crisis_band, "RiskOff": c.riskoff_band, "Neutral": c.neutral_band,
                "RiskOn": c.riskon_band, "StrongRiskOn": c.strong_band}
        frac = {"Crisis": c.frac_crisis, "RiskOff": c.frac_riskoff, "Neutral": c.frac_neutral,
                "RiskOn": c.frac_riskon, "StrongRiskOn": c.frac_strong}
        sigma = f["basket_vol"].clip(lower=c.vol_floor)
        mu = 0.5 * f["basket_mu"] + 0.5 * c.long_run_excess_prior
        kelly = (mu / sigma.pow(2)).clip(lower=0.0)
        prior_kelly = c.long_run_excess_prior / sigma.pow(2)
        kelly = kelly.where(f["basket_mu"].notna(), prior_kelly)
        target = pd.Series(index=f.index, dtype=float)
        for name in band:
            lo, hi = band[name]
            m = regime.eq(name)
            target.loc[m] = (kelly.loc[m] * frac[name]).clip(lo, hi)
        return target

    def _qqq_share(self, regime: pd.Series, rs126: pd.Series) -> pd.Series:
        c = self.c
        pos = rs126 > 0
        out = pd.Series(index=regime.index, dtype=float)
        out[regime.eq("Crisis")] = c.qqq_crisis
        out[regime.eq("RiskOff")] = c.qqq_riskoff
        out[regime.eq("Neutral") & pos] = c.qqq_neutral
        out[regime.eq("Neutral") & ~pos] = c.qqq_neutral_weak_rs
        out[regime.eq("RiskOn") & pos] = c.qqq_riskon
        out[regime.eq("RiskOn") & ~pos] = c.qqq_riskon_weak_rs
        out[regime.eq("StrongRiskOn") & pos] = c.qqq_strong
        out[regime.eq("StrongRiskOn") & ~pos] = c.qqq_strong_weak_rs
        return out.fillna(c.qqq_neutral_weak_rs)

    def _one_way_gate(self, prices: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
        c = self.c
        q = prices["QQQ"]
        return ((q > f["ma175"]) & (f["ma50"] > f["ma175"]) & (f["ma20"] > f["ma50"]) &
                (f["ma175_slope"] > 0) & (f["mom_oneway"] > 0) & (f["rs_oneway"] > 0) &
                (f["vol20"] < c.one_way_vol_entry) & (f["drawdown"] > c.one_way_dd_entry)).fillna(False)

    def daily_targets(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        c = self.c
        px = prices.sort_index().astype(float)
        required = ["QQQ", "SPY", "QLD", "SSO", "TQQQ", "CASH"]
        missing = [x for x in required if x not in px.columns]
        if missing:
            raise ValueError(f"prices missing columns: {missing}")
        f = self.features(px)
        regime = self._regime(px, f)
        e8 = self._raw_exposure(regime, f)
        recovery = (f["drawdown"] <= c.recovery_dd) & (f["mom63"] > 0) & (px["QQQ"] > f["ma105"])
        e8 = e8.where(~recovery, np.maximum(e8, 0.80))
        below_long = recovery & (px["QQQ"] < f["ma210"])
        e8 = e8.where(~below_long, np.minimum(e8, 1.00))
        e8 = e8.where(~(f["q_vol252"] > c.vol_brake_1), np.minimum(e8, 1.15))
        e8 = e8.where(~(f["q_vol252"] > c.vol_brake_2), np.minimum(e8, 1.00))
        exposure = 1.0 + c.defense_strength * (e8 - 1.0)
        one_way = self._one_way_gate(px, f) & regime.eq("StrongRiskOn")
        exposure = exposure + one_way.astype(float) * c.strong_trend_boost
        exposure = exposure.clip(lower=0.0, upper=c.final_exposure_cap)
        qshare = self._qqq_share(regime, f["rs126"])
        w = pd.DataFrame(0.0, index=px.index, columns=["QQQ", "SPY", "QLD", "SSO", "TQQQ", "CASH"])
        low = exposure <= 1.0
        w.loc[low, "QQQ"] = exposure[low] * qshare[low]
        w.loc[low, "SPY"] = exposure[low] * (1.0 - qshare[low])
        w.loc[low, "CASH"] = 1.0 - exposure[low]
        high = ~low
        t = pd.Series(0.0, index=px.index)
        t.loc[high & one_way] = np.minimum(c.tqqq_capital_cap, ((exposure - 1.0) / 2.0).clip(lower=0.0))[high & one_way]
        lev2 = (exposure - 1.0 - 2.0 * t).clip(lower=0.0)
        unlev = (1.0 - lev2 - t).clip(lower=0.0)
        qld = high & (f["rs126"] > 0)
        sso = high & ~qld
        w.loc[high, "TQQQ"] = t[high]
        w.loc[qld, "QLD"] = lev2[qld]
        w.loc[sso, "SSO"] = lev2[sso]
        w.loc[high, "QQQ"] = unlev[high] * qshare[high]
        w.loc[high, "SPY"] = unlev[high] * (1.0 - qshare[high])
        w = w.clip(lower=0.0)
        empty = w.sum(axis=1) <= 0
        if empty.any():
            w.loc[empty, "CASH"] = 1.0
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        meta = f.copy()
        meta["regime"] = regime
        meta["e8"] = e8
        meta["exposure"] = exposure
        meta["qqq_share"] = qshare
        meta["one_way"] = one_way
        meta["effective_from_weights"] = w["QQQ"] + w["SPY"] + 2*w["QLD"] + 2*w["SSO"] + 3*w["TQQQ"]
        return w, meta

    def generate_signals(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        c = self.c
        px = prices.sort_index().astype(float)
        targets, meta = self.daily_targets(px)
        signal_dates = targets.groupby(targets.index.to_period(c.weekly_bucket)).tail(1).index
        chosen = targets.loc[signal_dates].copy()
        reason = pd.Series("weekly", index=chosen.index, dtype=object)
        emergency = ((px["QQQ"] < meta["ma175"]) | (meta["vol20"] > c.tqqq_emergency_vol) |
                     (meta["drawdown"] < c.tqqq_emergency_dd)).fillna(False)
        prior_gate = meta["one_way"].shift(1).fillna(False).astype(bool)
        exit_dates = meta.index[prior_gate & emergency]
        if len(exit_dates):
            extras = []
            for d in exit_dates:
                row = targets.loc[d].copy()
                if row["TQQQ"] > 0:
                    target_e = min(float(meta.loc[d, "exposure"]), 1.34)
                    qs = float(meta.loc[d, "qqq_share"])
                    rs_pos = bool(meta.loc[d, "rs126"] > 0)
                    row[:] = 0.0
                    if target_e <= 1:
                        row["QQQ"] = target_e*qs; row["SPY"] = target_e*(1-qs); row["CASH"] = 1-target_e
                    else:
                        l2 = target_e - 1.0
                        u = 1.0 - l2
                        row["QQQ"] = u*qs; row["SPY"] = u*(1-qs)
                        row["QLD" if rs_pos else "SSO"] = l2
                extras.append((d, row))
            extra_df = pd.DataFrame({d:r for d,r in extras}).T
            extra_df.columns = targets.columns
            chosen = pd.concat([chosen, extra_df]).sort_index()
            chosen = chosen[~chosen.index.duplicated(keep="last")]
            reason = pd.Series("weekly", index=chosen.index, dtype=object)
            reason.loc[exit_dates.intersection(reason.index)] = "tqqq_emergency_exit"
        out_rows = []
        out_reason = []
        for d, row in chosen.iterrows():
            loc = px.index.get_indexer([d])[0]
            if loc + 1 >= len(px.index):
                continue
            exec_d = px.index[loc+1]
            out_rows.append((exec_d, row))
            out_reason.append((exec_d, reason.get(d, "weekly"), d))
        sig = pd.DataFrame({d:r for d,r in out_rows}).T if out_rows else pd.DataFrame(columns=targets.columns)
        sig.columns = targets.columns
        sig = sig[~sig.index.duplicated(keep="last")].sort_index()
        audit = pd.DataFrame(out_reason, columns=["execution_date", "reason", "decision_date"]).set_index("execution_date") if out_reason else pd.DataFrame()
        return sig, pd.concat([meta, audit], axis=1)
