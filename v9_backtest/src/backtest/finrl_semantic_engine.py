from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class BacktestConfig:
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000.0
    transaction_cost: float = 0.001


def align_weights(prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    px = prices.sort_index()
    sig = signals.sort_index().copy()
    common = [c for c in sig.columns if c in px.columns]
    # Preserve the most recent target issued before the backtest start date.
    # Reindexing signals directly to px.index would drop that pre-start target and
    # incorrectly leave buy-and-hold (and any warm-start strategy) at zero weight.
    idx = px.index.union(sig.index).sort_values()
    w = sig[common].reindex(idx).ffill().reindex(px.index).fillna(0.0)
    sums = w.sum(axis=1)
    nz = sums > 0
    w.loc[nz] = w.loc[nz].div(sums[nz], axis=0)
    return w


def run_weight_backtest(prices: pd.DataFrame, signals: pd.DataFrame, cfg: BacktestConfig):
    px = prices.sort_index().loc[cfg.start_date:cfg.end_date].ffill().dropna(how="all")
    w = align_weights(px, signals.loc[:cfg.end_date])
    ret = px[w.columns].pct_change().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * ret).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    costs = turnover * cfg.transaction_cost
    net = gross - costs
    nav = cfg.initial_capital * (1.0 + net).cumprod()
    return {"returns":net,"gross_returns":gross,"costs":costs,"turnover":turnover,"weights":w,"nav":nav}


def metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str,float]:
    r = returns.dropna()
    if len(r) < 2:
        return {}
    nav = (1+r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.2425
    total = nav.iloc[-1]/nav.iloc[0]-1
    cagr = nav.iloc[-1]**(1/years)-1 if years>0 else np.nan
    vol = r.std(ddof=1)*np.sqrt(252)
    sharpe = r.mean()/r.std(ddof=1)*np.sqrt(252) if r.std(ddof=1)>0 else np.nan
    dd = nav/nav.cummax()-1
    mdd = dd.min()
    calmar = cagr/abs(mdd) if mdd<0 else np.nan
    downside = r[r<0].std(ddof=1)*np.sqrt(252)
    sortino = (r.mean()*252)/downside if downside and np.isfinite(downside) else np.nan
    out = {"total_return":float(total),"cagr":float(cagr),"annual_vol":float(vol),"max_drawdown":float(mdd),"sharpe":float(sharpe),"sortino":float(sortino),"calmar":float(calmar)}
    if turnover is not None:
        out["annual_turnover"] = float(turnover.reindex(r.index).fillna(0).mean()*252)
    return out


def dca_account(strategy_returns: pd.Series, monthly_contribution: float = 20_000.0):
    r = strategy_returns.dropna().copy()
    month = r.index.to_period("M")
    first = ~month.duplicated()
    bal = 0.0
    flows = []
    vals = []
    for d, rr, is_first in zip(r.index, r.values, first):
        if is_first:
            bal += monthly_contribution
            flows.append((d, -monthly_contribution))
        bal *= 1.0 + rr
        vals.append(bal)
    if len(r):
        flows.append((r.index[-1], bal))
    return pd.Series(vals,index=r.index,name="dca_value"), flows


def xirr(flows):
    if not flows or len(flows)<2:
        return np.nan
    dates = [pd.Timestamp(d) for d,_ in flows]
    vals = np.array([v for _,v in flows],dtype=float)
    t = np.array([(d-dates[0]).days/365.2425 for d in dates])
    def f(rate):
        if rate <= -0.999999:
            return np.inf
        return np.sum(vals / np.power(1.0+rate,t))
    lo,hi=-0.999,10.0
    flo,fhi=f(lo),f(hi)
    for _ in range(200):
        if np.sign(flo)!=np.sign(fhi):
            break
        hi*=2; fhi=f(hi)
    if np.sign(flo)==np.sign(fhi):
        return np.nan
    for _ in range(200):
        mid=(lo+hi)/2; fm=f(mid)
        if abs(fm)<1e-8:
            return mid
        if np.sign(fm)==np.sign(flo):
            lo=mid; flo=fm
        else:
            hi=mid
    return (lo+hi)/2
