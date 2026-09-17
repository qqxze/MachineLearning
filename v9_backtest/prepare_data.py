#!/usr/bin/env python3
from pathlib import Path
import urllib.request
import pandas as pd

BASE='https://raw.githubusercontent.com/bumbeishvili/tqqq.networthcast.com/main/data'
FILES=['synthetic-qqq.tsv','spy.tsv','synthetic-qld.tsv','synthetic-sso.tsv','synthetic-tqqq.tsv','short-rates.tsv']
OUT=Path('data')
OUT.mkdir(parents=True,exist_ok=True)

for fn in FILES:
    url=f'{BASE}/{fn}'
    print('download',url)
    urllib.request.urlretrieve(url, OUT/fn)

# Build a no-lookahead cash total-return proxy on the QQQ trading calendar.
# short-rates.tsv is annualized percent.  Return from t-1 close to t close uses the prior
# observed short rate, so the cash series never uses a future rate observation.
qqq=pd.read_csv(OUT/'synthetic-qqq.tsv',sep='\t')
rates=pd.read_csv(OUT/'short-rates.tsv',sep='\t')
qidx=pd.to_datetime(qqq['Date']).dt.tz_localize(None).dt.normalize()
ridx=pd.to_datetime(rates['Date']).dt.tz_localize(None).dt.normalize()
r=pd.Series(pd.to_numeric(rates['Rate'],errors='coerce').values,index=ridx).sort_index()
r=r[~r.index.duplicated(keep='last')]
calendar=pd.DatetimeIndex(qidx)
rate=r.reindex(calendar,method='ffill')
if rate.isna().any():
    rate=rate.bfill()
days=calendar.to_series().diff().dt.days.fillna(0).clip(lower=0)
prior_rate=rate.shift(1).fillna(rate.iloc[0]) / 100.0
growth=(1.0+prior_rate).pow(days.values/365.2425)
cash=100.0*growth.cumprod()
out=pd.DataFrame({'Date':pd.to_datetime(calendar).strftime('%m/%d/%Y 16:00:00'),'Close':cash.values})
out.to_csv(OUT/'cash.tsv',sep='\t',index=False)

for fn in FILES[:-1]+['cash.tsv']:
    df=pd.read_csv(OUT/fn,sep='\t')
    d=pd.to_datetime(df.iloc[:,0])
    print(fn, len(df), d.min(), d.max())
