import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from .router import Reply

def _choose_from(q: str, options: List[str]) -> Optional[str]:
    if not options: return None
    ql = q.lower()
    for o in options:
        if o.lower() == ql: return o
    for o in options:
        if o.lower() in ql or ql in o.lower(): return o
    toks = set(re.findall(r"[A-Za-z0-9\-\.]+", ql))
    best, score = None, 0
    for o in options:
        ot = set(re.findall(r"[A-Za-z0-9\-\.]+", o.lower()))
        s = len(toks & ot)
        if s>score: best, score = o, s
    return best

def _parse_years(q: str, default: Tuple[int,int]) -> Tuple[int,int]:
    ys = [int(m.group(0)) for m in re.finditer(r"\b(19|20)\d{2}\b", q)]
    if len(ys)>=2: return (min(ys), max(ys))
    if len(ys)==1: return (ys[0], default[1])
    return default

def definition_handler(q: str, resources: Dict[str,str]) -> Reply:
    dd = resources.get("dictionary", {})
    meth = resources.get("methodology", "")
    for k, v in dd.items():
        if k.lower() in q.lower():
            return Reply("definition", f"{k}: {v}", ["resources/data_dictionary.json"])
    if "cbd" in q.lower():
        return Reply("definition",
                     "CBD (Cumulative Budget Deviation) is the trapezoid area of (company − benchmark) over the selected year window. Negative = better alignment.",
                     ["resources/methodology.md"])
    if "scenario" in q.lower() or "pledge" in q.lower():
        return Reply("definition",
                     "Scenarios are benchmark pathways normalized to: 1.5°C, Below 2°C, and National Pledges (aka International/Paris Pledges).",
                     ["resources/methodology.md"])
    return Reply("definition", meth.splitlines()[0] if meth else "Methodology available in resources.", ["resources/methodology.md"])

def slice_handler(q: str, state: Dict[str,Any], fact_company: pd.DataFrame, fact_bench: pd.DataFrame) -> Reply:
    if isinstance(fact_company.index, pd.MultiIndex):
        sectors = sorted(fact_company.index.get_level_values("sector").unique().tolist())
        companies = sorted(fact_company.index.get_level_values("company").unique().tolist())
    else:
        sectors = sorted(fact_company["sector"].unique().tolist())
        companies = sorted(fact_company["company"].unique().tolist())

    if isinstance(fact_bench.index, pd.MultiIndex):
        regions = sorted(fact_bench.index.get_level_values("region").unique().tolist())
        scenarios = sorted(fact_bench.index.get_level_values("scenario").unique().tolist())
    else:
        regions = sorted(fact_bench["region"].dropna().unique().tolist()) if "region" in fact_bench.columns else []
        scenarios = sorted(fact_bench["scenario"].dropna().unique().tolist()) if "scenario" in fact_bench.columns else []

    sector = _choose_from(q, sectors) or state.get("sector")
    company = _choose_from(q, companies) or state.get("company")
    region = _choose_from(q, regions) or state.get("region")
    wanted_scen = [s for s in scenarios if s.lower() in q.lower()] or state.get("scenarios") or []
    y0, y1 = _parse_years(q, state.get("year_window",(2015,2035)))

    try:
        from src.analytics import view_pathway
    except Exception:
        from src.views import view_pathway
    company_df, scen_map, bands = view_pathway(
        fact_company, fact_bench,
        sector=sector, company=company, region=region,
        scenarios=wanted_scen, year_range=(y0,y1),
        exact_region=state.get("exact_region", False)
    )

    text = f"Prepared slice: Sector={sector}, Company={company}, Region={region}, Scenarios={', '.join(wanted_scen) or 'default'}, Years={y0}-{y1}."
    attachments = {"company_slice.csv": company_df}
    for nm, df in scen_map.items():
        attachments[f"benchmark_{nm}.csv"] = df
    return Reply("slice_request", text, ["resources/data_dictionary.json"], set_filters={
        "sector": sector, "company": company, "region": region, "scenarios": wanted_scen, "year_window": (y0,y1)
    }, attachments=attachments)

def explain_handler(q: str, state: Dict[str,Any], fact_company: pd.DataFrame, fact_bench: pd.DataFrame) -> Reply:
    sector = state.get("sector"); company = state.get("company"); region = state.get("region")
    scenarios = state.get("scenarios") or []
    y0, y1 = state.get("year_window",(2015,2035))
    if not (sector and company):
        return Reply("explain_view", "Select sector and company first.", [])

    try:
        from src.analytics import view_pathway
    except Exception:
        from src.views import view_pathway
    cdf, scen_map, _ = view_pathway(fact_company, fact_bench, sector, company, region, scenarios or ["Below 2°C","1.5°C"], (y0,y1), state.get("exact_region", False))
    if cdf.empty:
        return Reply("explain_view", "No company data in the selected window.", [])

    # Check if we have valid intensity data
    intensity_data = cdf["intensity"].dropna()
    if intensity_data.empty:
        return Reply("explain_view", f"No intensity data available for {company} in {sector} sector.", [])
    
    y_start, y_end = int(cdf["year"].min()), int(cdf["year"].max())
    
    # Get intensity values, handling NaN cases
    start_data = cdf.loc[cdf["year"]==y_start, "intensity"]
    end_data = cdf.loc[cdf["year"]==y_end, "intensity"]
    
    if start_data.empty or pd.isna(start_data.values[0]):
        return Reply("explain_view", f"No intensity data available for {company} at start year {y_start}.", [])
    if end_data.empty or pd.isna(end_data.values[0]):
        return Reply("explain_view", f"No intensity data available for {company} at end year {y_end}.", [])
        
    c_start = float(start_data.values[0])
    c_end = float(end_data.values[0])

    ref_name = "Below 2°C" if "Below 2°C" in scen_map else ("1.5°C" if "1.5°C" in scen_map else None)
    detail = ""
    if ref_name:
        ref = scen_map[ref_name]
        m = cdf.merge(ref, on="year", how="inner")
        if not m.empty:
            delta_end = c_end - float(m.loc[m["year"]==y_end, "benchmark"].values[0])
            cbd = float(np.trapz(m["intensity"].values - m["benchmark"].values, m["year"].values))
            detail = f" Against {ref_name}, Δ{y_end}={delta_end:+.1f}, CBD({y0}-{y1})={cbd:+.0f}."
    text = f"{company} moved from {c_start:.1f} to {c_end:.1f} between {y_start}-{y_end}.{detail}"
    return Reply("explain_view", text, ["resources/methodology.md"])

def diagnostics_handler(q: str, state: Dict[str,Any], fact_company: pd.DataFrame, fact_bench: pd.DataFrame) -> Reply:
    sector = state.get("sector"); company = state.get("company"); region = state.get("region")
    y0, y1 = state.get("year_window",(2015,2035))
    msgs = []
    if sector is not None:
        fb = fact_bench
        if "sector" in fb.columns:
            fb = fb[fb["sector"]==sector]
        elif isinstance(fb.index, pd.MultiIndex):
            try:
                fb = fb.xs(sector, level="sector")
            except Exception:
                pass
        scens = sorted(fb["scenario"].dropna().unique().tolist()) if "scenario" in fb.columns else []
        regs = sorted(fb["region"].dropna().unique().tolist()) if "region" in fb.columns else []
        msgs.append(f"Available scenarios: {', '.join(scens) if scens else 'n/a'}; regions: {', '.join(regs) if regs else 'n/a'}.")
    if sector and company:
        fc = fact_company
        if "sector" in fc.columns and "company" in fc.columns:
            fc = fc[(fc["sector"]==sector)&(fc["company"]==company)]
        elif isinstance(fc.index, pd.MultiIndex):
            try:
                fc = fc.xs((sector, company), level=("sector","company"))
            except Exception:
                pass
        yrs = set(fc["Year"].tolist()) if "Year" in fc.columns else (set(fc.index.get_level_values("year").tolist()) if "year" in fc.index.names else set())
        missing = sorted(list(set(range(y0,y1+1)) - yrs))
        if missing:
            msgs.append(f"Missing years {y0}-{y1}: {missing[:10]}{'...' if len(missing)>10 else ''}.")
    if region and "region" in fact_bench.columns:
        exact = not fact_bench[(fact_bench.get("sector",sector)==sector) & (fact_bench["region"]==region)].empty if sector else not fact_bench[fact_bench["region"]==region].empty
        if not exact:
            msgs.append(f"Exact region '{region}' not present; fallback to Global will apply unless disabled.")

    return Reply("diagnostics", " ".join(msgs) or "No issues detected.", ["resources/methodology.md"])
