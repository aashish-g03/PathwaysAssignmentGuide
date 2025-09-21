import numpy as np
import pandas as pd

def area_diff(company_df, benchmark_df, start=2020, end=2035):
    c_df = company_df[(company_df["year"] >= start) & (company_df["year"] <= end)].sort_values("year")
    b_df = benchmark_df[(benchmark_df["year"] >= start) & (benchmark_df["year"] <= end)].sort_values("year")
    
    if c_df.empty or b_df.empty:
        return float('nan')
    
    merged = pd.merge(c_df, b_df, on="year", how="inner")
    
    if merged.empty:
        return float('nan')
    
    diff = merged["intensity"] - merged["benchmark"]
    return float(np.trapz(diff, merged["year"]))

def sector_outliers(cp_long, sb_long, sector, scenario="1.5°C",
                  company_region=None, country=None,
                  start=2020, end=2035, k=10):
    sector_filter = cp_long.index.get_level_values('sector') == sector
    dfc = cp_long.loc[sector_filter].reset_index()
    
    if company_region and "companyregion" in dfc.columns:
        dfc = dfc[dfc["companyregion"] == company_region]
    
    if country and "geography" in dfc.columns:
        dfc = dfc[dfc["geography"] == country]
    
    if dfc.empty:
        empty_df = pd.DataFrame(columns=["company", "cbd", "z"])
        return empty_df, empty_df
        
    benchmark_filter = (
        (sb_long.index.get_level_values('sector') == sector) & 
        (sb_long.index.get_level_values('scenario') == scenario) & 
        (sb_long.index.get_level_values('region') == "Global")
    )
    dfb = sb_long.loc[benchmark_filter].reset_index()[["year", "benchmark"]]
    
    scores = []
    for company, g in dfc.groupby("company"):
        cbd = area_diff(g[["year", "intensity"]], dfb, start=start, end=end)
        scores.append((company, cbd))
        
    s = pd.DataFrame(scores, columns=["company", "cbd"]).dropna()
    
    if s.empty:
        return s, s
    
    s["z"] = (s["cbd"] - s["cbd"].mean()) / s["cbd"].std(ddof=1) if len(s) > 1 else 0.0
    
    return s.nsmallest(k, "cbd"), s.nlargest(k, "cbd")
