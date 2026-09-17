#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from src.strategies.v9_final_strategy import V9FinalStrategy, V9FinalConfig
from src.backtest.finrl_semantic_engine import BacktestConfig, run_weight_backtest, metrics, dca_account, xirr

ASSETS=['QQQ','SPY','QLD','SSO','TQQQ','CASH']


def load_two_col(path: Path, name: str) -> pd.Series:
    df=pd.read_csv(path,sep=None,engine='python')
    cols={c.lower():c for c in df.columns}
    dcol=cols.get('date',df.columns[0]); ccol=cols.get('close',df.columns[-1])
    idx=pd.to_datetime(df[dcol]).dt.tz_localize(None).dt.normalize()
    s=pd.Series(pd.to_numeric(df[ccol],errors='coerce').values,index=idx,name=name).dropna()
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_prices(data_dir: Path) -> pd.DataFrame:
    fmap={'QQQ':'synthetic-qqq.tsv','SPY':'spy.tsv','QLD':'synthetic-qld.tsv','SSO':'synthetic-sso.tsv','TQQQ':'synthetic-tqqq.tsv','CASH':'cash.tsv'}
    missing=[fn for fn in fmap.values() if not (data_dir/fn).exists()]
    if missing:
        raise FileNotFoundError('Missing data files: '+', '.join(missing))
    out=pd.concat([load_two_col(data_dir/fn,k) for k,fn in fmap.items()],axis=1).sort_index()
    out=out.ffill().dropna(subset=ASSETS)
    return out[ASSETS]


def buyhold_signal(px: pd.DataFrame, weights: dict[str,float]) -> pd.DataFrame:
    return pd.DataFrame([weights],index=[px.index[0]]).reindex(columns=ASSETS,fill_value=0.0)


def strategy_signals(px, variant):
    c=V9FinalConfig()
    if variant=='v9_nolev':
        c=replace(c,strong_band=(1.0,1.0),riskon_band=(0.95,1.0),tqqq_capital_cap=0.0,strong_trend_boost=0.0,final_exposure_cap=1.0)
    elif variant=='v9_134':
        c=replace(c,tqqq_capital_cap=0.0,strong_trend_boost=0.0,final_exposure_cap=1.34)
    elif variant=='v9_tqqq_same134':
        c=replace(c,strong_trend_boost=0.0,final_exposure_cap=1.34,tqqq_capital_cap=0.10)
    elif variant=='v9_final145':
        pass
    else:
        raise ValueError(variant)
    return V9FinalStrategy(c).generate_signals(px)


def underwater_months(r: pd.Series) -> int:
    nav=(1+r).cumprod(); dd=nav/nav.cummax()-1
    best=cur=0
    for x in (dd<0):
        cur=cur+1 if x else 0; best=max(best,cur)
    return int(round(best/21.0))


def crisis_stats(r):
    windows={'dotcom':['2000-03-24','2002-10-09'],'gfc':['2007-10-09','2009-03-09'],'covid':['2020-02-19','2020-03-23'],'bear2022':['2022-01-03','2022-10-14']}
    out={}
    for k,(a,b) in windows.items():
        x=r.loc[a:b]
        if len(x):
            n=(1+x).cumprod(); out[k]={'return':float(n.iloc[-1]-1),'max_drawdown':float((n/n.cummax()-1).min())}
    return out


def rolling_summary(r, bench, years):
    starts=pd.date_range(max(r.index.min(),bench.index.min()),min(r.index.max(),bench.index.max())-pd.DateOffset(years=years),freq='MS')
    diffs=[]; ddwins=[]
    for st in starts:
        en=st+pd.DateOffset(years=years)
        a=r.loc[st:en]; b=bench.loc[st:en]
        if len(a)<200*years or len(b)<200*years: continue
        wa=float((1+a).prod()); wb=float((1+b).prod()); diffs.append(wa/wb-1)
        dda=float(((1+a).cumprod()/((1+a).cumprod().cummax())-1).min())
        ddb=float(((1+b).cumprod()/((1+b).cumprod().cummax())-1).min())
        ddwins.append(dda>ddb)
    return {'n':len(diffs),'beat_wealth_pct':float(np.mean(np.array(diffs)>0)) if diffs else np.nan,'median_wealth_excess':float(np.median(diffs)) if diffs else np.nan,'lower_mdd_pct':float(np.mean(ddwins)) if ddwins else np.nan}


def run(data_dir,start,end,out_dir,monthly=20000.0):
    out_dir.mkdir(parents=True,exist_ok=True)
    px=load_prices(data_dir).loc[:end]
    warm=px.loc[:end]
    if pd.Timestamp(start)<warm.index.min():
        raise ValueError('start before available data')
    variants={'QQQ':buyhold_signal(px,{'QQQ':1}),'QQQ_SPY_50_50':buyhold_signal(px,{'QQQ':.5,'SPY':.5})}
    audits={}
    for v in ['v9_nolev','v9_134','v9_tqqq_same134','v9_final145']:
        variants[v],audits[v]=strategy_signals(warm,v)
    rows=[]; allr={}; navs={}
    for tc in [0.0005,0.0010,0.0020]:
        for name,sig in variants.items():
            bt=run_weight_backtest(px,sig,BacktestConfig(start,end,transaction_cost=tc))
            m=metrics(bt['returns'],bt['turnover']); dca,flows=dca_account(bt['returns'],monthly); irr=xirr(flows)
            row={'variant':name,'transaction_cost':tc,**m,'dca_final':float(dca.iloc[-1]),'dca_xirr':float(irr),'underwater_months':underwater_months(bt['returns'])}
            rows.append(row)
            if tc==.001:
                allr[name]=bt['returns']; navs[name]=bt['nav']
                bt['weights'].to_csv(out_dir/f'weights_{name}.csv')
                bt['turnover'].rename('turnover').to_csv(out_dir/f'turnover_{name}.csv')
    summary=pd.DataFrame(rows); summary.to_csv(out_dir/'summary.csv',index=False)
    pd.DataFrame(navs).to_csv(out_dir/'navs_10bp.csv')
    crises={k:crisis_stats(v) for k,v in allr.items()}
    (out_dir/'crisis_stats.json').write_text(json.dumps(crises,indent=2),encoding='utf-8')
    rolling={}
    for name,r in allr.items():
        if name=='QQQ': continue
        rolling[name]={'3y_vs_QQQ':rolling_summary(r,allr['QQQ'],3),'5y_vs_QQQ':rolling_summary(r,allr['QQQ'],5)}
    (out_dir/'rolling.json').write_text(json.dumps(rolling,indent=2),encoding='utf-8')
    for name,a in audits.items():
        a.to_csv(out_dir/f'audit_{name}.csv')
    print(summary.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,required=True); p.add_argument('--start',default='2000-01-03'); p.add_argument('--end',default='2025-12-31'); p.add_argument('--out-dir',type=Path,default=Path('results')); p.add_argument('--monthly',type=float,default=20000.0)
    a=p.parse_args(); run(a.data_dir,a.start,a.end,a.out_dir,a.monthly)
