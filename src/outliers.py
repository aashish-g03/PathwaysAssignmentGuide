import numpy as np
import pandas as pd

def area_diff(company_xy: pd.DataFrame, bench_xy: pd.DataFrame, start=2020, end=2035) -> float:
    c = company_xy[(company_xy.Year>=start)&(company_xy.Year<=end)].sort_values('Year')
    if c.empty: return float('nan')
    
    if c['Year'].duplicated().any():
        c = c.groupby('Year')['Intensity'].mean().reset_index()
    
    if bench_xy['Year'].duplicated().any():
        bench_xy = bench_xy.groupby('Year')['Benchmark'].mean().reset_index()
        
    b = bench_xy.set_index('Year').reindex(c['Year']).reset_index().rename(columns={'Benchmark':'b'})
    if b['b'].isna().all(): return float('nan')
    ydiff = c['Intensity'].values - b['b'].values
    return float(np.trapz(ydiff, c['Year'].values))

def sector_outliers(cp_long: pd.DataFrame, sb_long: pd.DataFrame, sector: str, scenario='1.5°C', region_pref=None, k=10, start=2020, end=2035):
    dfc = cp_long[cp_long['Sector']==sector]
    dfb = sb_long[(sb_long['Sector']==sector)]
    if 'Scenario' in dfb.columns:
        dfb = dfb[dfb['Scenario']==scenario]
    if region_pref and 'Region' in dfb.columns:
        cand = dfb[dfb['Region']==region_pref]
        if cand.empty:
            cand = dfb[dfb['Region']=='Global']
        dfb = cand
    scores = []
    for company, g in dfc.groupby('Company'):
        val = area_diff(g[['Year','Intensity']], dfb[['Year','Benchmark']], start=start, end=end)
        scores.append((company, val))
    s = pd.DataFrame(scores, columns=['Company','CBD']).dropna()
    if s.empty:
        s['z'] = []
        return s, s
    s['z'] = (s['CBD'] - s['CBD'].mean())/s['CBD'].std(ddof=1) if len(s) > 1 else 0.0
    good = s.nsmallest(k, 'CBD')
    bad  = s.nlargest(k, 'CBD')
    return good, bad
