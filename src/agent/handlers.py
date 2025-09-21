import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from .router import Reply

from ..analytics import (
    calculate_climate_risk_score, compute_transition_readiness, 
    generate_peer_comparison_metrics, detect_financial_outliers, 
    rank_outlier_significance
)
from ..business_intelligence import (
    create_sector_performance_summary, generate_climate_risk_heatmap,
    generate_executive_summary
)

# This is a very basic Intent based qna bot, which I had previously worked on
# and found user to integrate in given timeframe instead of an LLM based agent

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

def process_business_query(query: str, user_context: str, fact_company: pd.DataFrame, 
                         fact_benchmark: pd.DataFrame) -> Reply:

    q_lower = query.lower()

    if any(phrase in q_lower for phrase in ['best positioned', 'top performers', 'investment opportunities', 'outperforming']):
        return handle_investment_opportunity_query(query, fact_company, fact_benchmark)
    
    elif any(phrase in q_lower for phrase in ['falling behind', 'at risk', 'poor performance', 'concerning trends']):
        return handle_risk_assessment_query(query, fact_company, fact_benchmark)
    
    elif any(phrase in q_lower for phrase in ['compare', 'vs', 'versus', 'against', 'relative to']):
        return handle_peer_comparison_query(query, fact_company, fact_benchmark)
    
    elif any(phrase in q_lower for phrase in ['sector analysis', 'industry overview', 'sector trends']):
        return handle_sector_analysis_query(query, fact_company, fact_benchmark)
    
    elif any(phrase in q_lower for phrase in ['executive summary', 'overview', 'high level', 'dashboard']):
        return handle_executive_summary_query(query, fact_company, fact_benchmark)
    
    elif any(phrase in q_lower for phrase in ['outliers', 'unusual', 'anomalies', 'significant deviations']):
        return handle_outlier_detection_query(query, fact_company, fact_benchmark)
    
    return Reply(
        "business_query",
        "I can help you with investment analysis, risk assessment, peer comparisons, sector analysis, executive summaries, and outlier detection. "
        "Try queries like: 'Which energy companies are best positioned for 1.5°C?', 'Show risk assessment for auto sector', or 'Compare Apple vs tech sector'.",
        ["resources/methodology.md"]
    )

def handle_investment_opportunity_query(query: str, fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame) -> Reply:
    
    sector = extract_sector_from_query(query)
    if not sector:
        return Reply("business_query", "Please specify a sector for investment analysis (e.g., 'Energy', 'Utilities', 'Auto').", [])
    
    scenarios = extract_scenarios_from_query(query)
    if not scenarios:
        scenarios = ["1.5°C", "Below 2°C"]  # Default to ambitious scenarios
    
    try:
        dashboard = create_sector_performance_summary(fact_company, fact_benchmark, sector, scenarios)
        
        if dashboard.sector_rankings.empty:
            return Reply("business_query", f"No data available for investment analysis in {sector} sector.", [])
        
        top_performers = dashboard.sector_rankings.head(5)
        
        improving_companies = []
        if not dashboard.trend_analysis.empty:
            improving_companies = dashboard.trend_analysis[
                dashboard.trend_analysis['trend_category'] == 'improving'
            ]['company'].tolist()
        
        investment_recommendations = []
        for _, company_row in top_performers.iterrows():
            company = company_row['company']
            
            trend_info = "improving trajectory" if company in improving_companies else "stable/declining trajectory"
            risk_level = "Low" if company_row.get('latest_intensity', 0) < top_performers['latest_intensity'].median() else "Medium"
            
            investment_recommendations.append(f"**{company}**: Rank #{company_row['sector_rank']} in {sector}, {trend_info}, Risk: {risk_level}")
        
        response_text = f"## Investment Opportunities in {sector} Sector\n\n"
        response_text += f"**Analysis Date**: {dashboard.last_updated[:10]}\n"
        response_text += f"**Scenarios Analyzed**: {', '.join(scenarios)}\n\n"
        response_text += "### Top Investment Candidates:\n"
        response_text += "\n".join(investment_recommendations)
        
        if improving_companies:
            response_text += f"\n\n### Companies with Positive Climate Trends:\n"
            response_text += ", ".join(improving_companies[:10])
        
        response_text += f"\n\n*Based on analysis of {len(dashboard.sector_rankings)} companies in {sector} sector.*"
        
        return Reply("business_query", response_text, ["resources/methodology.md"])
        
    except Exception as e:
        return Reply("business_query", f"Error analyzing investment opportunities in {sector}: {str(e)}", [])

def handle_risk_assessment_query(query: str, fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame) -> Reply:
    
    sector = extract_sector_from_query(query)
    if not sector:
        return Reply("business_query", "Please specify a sector for risk assessment (e.g., 'Oil & Gas', 'Mining', 'Cement').", [])
    
    try:
        dashboard = create_sector_performance_summary(fact_company, fact_benchmark, sector)
        
        if dashboard.risk_distribution.empty:
            return Reply("business_query", f"No risk data available for {sector} sector.", [])
        
        high_risk = dashboard.risk_distribution[dashboard.risk_distribution['risk_category'] == 'HIGH_RISK']
        medium_risk = dashboard.risk_distribution[dashboard.risk_distribution['risk_category'] == 'MEDIUM_RISK']
        low_risk = dashboard.risk_distribution[dashboard.risk_distribution['risk_category'] == 'LOW_RISK']
        
        declining_companies = []
        if not dashboard.trend_analysis.empty:
            declining_companies = dashboard.trend_analysis[
                dashboard.trend_analysis['trend_category'] == 'declining'
            ]['company'].tolist()
        
        response_text = f"## Climate Risk Assessment: {sector} Sector\n\n"
        response_text += f"**Total Companies Analyzed**: {len(dashboard.risk_distribution)}\n\n"
        
        response_text += "### Risk Distribution:\n"
        response_text += f"- **High Risk**: {len(high_risk)} companies ({len(high_risk)/len(dashboard.risk_distribution)*100:.1f}%)\n"
        response_text += f"- **Medium Risk**: {len(medium_risk)} companies ({len(medium_risk)/len(dashboard.risk_distribution)*100:.1f}%)\n"
        response_text += f"- **Low Risk**: {len(low_risk)} companies ({len(low_risk)/len(dashboard.risk_distribution)*100:.1f}%)\n\n"
        
        if not high_risk.empty:
            response_text += "### High Risk Companies (Immediate Attention Required):\n"
            high_risk_sorted = high_risk.sort_values('confidence_level', ascending=False)
            for _, row in high_risk_sorted.head(5).iterrows():
                confidence = f"Confidence: {row['confidence_level']:.1%}" if 'confidence_level' in row else ""
                response_text += f"- **{row['company']}**: {confidence}\n"
        
        if declining_companies:
            response_text += f"\n### Companies with Declining Trends:\n"
            response_text += ", ".join(declining_companies[:8])
        
        risk_percentage = len(high_risk) / len(dashboard.risk_distribution) * 100
        if risk_percentage > 40:
            response_text += f"\n\n**RISK ALERT**: {risk_percentage:.1f}% of {sector} companies are in HIGH_RISK category - consider sector rebalancing."
        
        return Reply("business_query", response_text, ["resources/methodology.md"])
        
    except Exception as e:
        return Reply("business_query", f"Error conducting risk assessment for {sector}: {str(e)}", [])

def handle_peer_comparison_query(query: str, fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame) -> Reply:
    company = extract_company_from_query(query, fact_company)
    sector = extract_sector_from_query(query)
    
    if not company and not sector:
        return Reply("business_query", "Please specify a company and/or sector for peer comparison.", [])
    
    try:
        if company and not sector:
            if hasattr(fact_company.index, 'get_level_values'):
                company_data = fact_company.xs(company, level='company', drop_level=False)
                sector = company_data.index.get_level_values('sector')[0]
            else:
                company_data = fact_company[fact_company['company'] == company]
                sector = company_data['sector'].iloc[0] if not company_data.empty else None
        
        if not sector:
            return Reply("business_query", f"Could not determine sector for {company}.", [])
        
        dashboard = create_sector_performance_summary(fact_company, fact_benchmark, sector)
        
        if dashboard.sector_rankings.empty:
            return Reply("business_query", f"No peer data available for comparison in {sector} sector.", [])
        
        response_text = f"## Peer Comparison Analysis\n\n"
        
        if company:
            company_data = dashboard.sector_rankings[dashboard.sector_rankings['company'] == company]
            if not company_data.empty:
                rank = company_data['sector_rank'].iloc[0]
                total_companies = len(dashboard.sector_rankings)
                percentile = ((total_companies - rank + 1) / total_companies) * 100
                
                response_text += f"### {company} vs {sector} Sector Peers\n\n"
                response_text += f"**Sector Rank**: #{rank} out of {total_companies} companies\n"
                response_text += f"**Percentile**: {percentile:.1f}th percentile\n"
                response_text += f"**Latest Intensity**: {company_data['latest_intensity'].iloc[0]:.2f}\n\n"
                
                response_text += "### Sector Leaders (Top 3):\n"
                for _, row in dashboard.sector_rankings.head(3).iterrows():
                    indicator = "**YOUR COMPANY**" if row['company'] == company else ""
                    response_text += f"{row['sector_rank']}. {row['company']}: {row['latest_intensity']:.2f} {indicator}\n"
                
                response_text += "\n### Sector Laggards (Bottom 3):\n"
                for _, row in dashboard.sector_rankings.tail(3).iterrows():
                    indicator = "**YOUR COMPANY**" if row['company'] == company else ""
                    response_text += f"{row['sector_rank']}. {row['company']}: {row['latest_intensity']:.2f} {indicator}\n"
            else:
                response_text += f"**{company}** not found in {sector} sector analysis.\n\n"
        
        response_text += f"\n### {sector} Sector Overview:\n"
        response_text += f"**Total Companies**: {len(dashboard.sector_rankings)}\n"
        if not dashboard.sector_rankings.empty:
            median_intensity = dashboard.sector_rankings['latest_intensity'].median()
            response_text += f"**Sector Median Intensity**: {median_intensity:.2f}\n"
        
        return Reply("business_query", response_text, ["resources/methodology.md"])
        
    except Exception as e:
        return Reply("business_query", f"Error conducting peer comparison: {str(e)}", [])

def handle_sector_analysis_query(query: str, fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame) -> Reply:
    
    sector = extract_sector_from_query(query)
    if not sector:
        return Reply("business_query", "Please specify a sector for analysis (e.g., 'Energy', 'Utilities', 'Transport').", [])
    
    try:
        dashboard = create_sector_performance_summary(fact_company, fact_benchmark, sector)
        
        response_text = f"# {sector} Sector Analysis\n\n"
        
        if not dashboard.data_coverage.empty:
            coverage = dashboard.data_coverage['coverage_percentage'].iloc[0]
            response_text += f"**Data Coverage**: {coverage:.1f}% ({dashboard.data_coverage['companies_with_data'].iloc[0]} of {dashboard.data_coverage['total_companies'].iloc[0]} companies)\n\n"
        
        if not dashboard.trend_analysis.empty:
            improving = len(dashboard.trend_analysis[dashboard.trend_analysis['trend_category'] == 'improving'])
            declining = len(dashboard.trend_analysis[dashboard.trend_analysis['trend_category'] == 'declining'])
            stable = len(dashboard.trend_analysis[dashboard.trend_analysis['trend_category'] == 'stable'])
            
            total_with_trends = improving + declining + stable
            if total_with_trends > 0:
                if improving > declining:
                    sector_direction = "improving"
                elif declining > improving:
                    sector_direction = "declining"
                else:
                    sector_direction = "mixed"
                
                response_text += f"## Sector Trajectory: **{sector_direction.upper()}**\n\n"
                response_text += f"- **Improving**: {improving} companies\n"
                response_text += f"- **Declining**: {declining} companies\n"
                response_text += f"- **Stable**: {stable} companies\n\n"
        
        if not dashboard.sector_rankings.empty:
            response_text += "## Performance Leaders:\n"
            for _, row in dashboard.sector_rankings.head(3).iterrows():
                response_text += f"**{row['company']}**: {row['latest_intensity']:.2f} (Rank #{row['sector_rank']})\n"
            
            response_text += "\n## Performance Laggards:\n"
            for _, row in dashboard.sector_rankings.tail(3).iterrows():
                response_text += f"**{row['company']}**: {row['latest_intensity']:.2f} (Rank #{row['sector_rank']})\n"
        
        if not dashboard.risk_distribution.empty:
            high_risk_pct = len(dashboard.risk_distribution[dashboard.risk_distribution['risk_category'] == 'HIGH_RISK']) / len(dashboard.risk_distribution) * 100
            response_text += f"\n## Risk Profile:\n"
            response_text += f"**High Risk Companies**: {high_risk_pct:.1f}% of sector\n"
        
        return Reply("business_query", response_text, ["resources/methodology.md"])
        
    except Exception as e:
        return Reply("business_query", f"Error analyzing {sector} sector: {str(e)}", [])

def handle_executive_summary_query(query: str, fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame) -> Reply:
    try:
        summary = generate_executive_summary(fact_company, fact_benchmark)
        
        response_text = "# Executive Climate Risk Summary\n\n"
        response_text += f"**Analysis Date**: {summary['analysis_date'][:10]}\n"
        response_text += f"**Sectors Analyzed**: {summary['total_sectors_analyzed']}\n\n"
        
        if summary['key_findings']:
            response_text += "## Key Findings:\n"
            for finding in summary['key_findings']:
                response_text += f"- **{finding['sector']}**: {finding['companies_analyzed']} companies analyzed, "
                response_text += f"{finding['high_risk_percentage']}% high risk, "
                response_text += f"{finding['data_coverage']:.1f}% data coverage\n"
        
        if summary['risk_alerts']:
            response_text += "\n## Risk Alerts:\n"
            for alert in summary['risk_alerts']:
                response_text += f"- **{alert['sector']}**: {alert['message']}\n"
        
        if summary['opportunities']:
            response_text += "\n## Investment Opportunities:\n"
            for opp in summary['opportunities']:
                response_text += f"- **{opp['sector']}**: {opp['message']}\n"
        
        response_text += "\n*This summary provides high-level insights for strategic decision-making. Contact the team for detailed sector analysis.*"
        
        return Reply("business_query", response_text, ["resources/methodology.md"])
        
    except Exception as e:
        return Reply("business_query", f"Error generating executive summary: {str(e)}", [])

def handle_outlier_detection_query(query: str, fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame) -> Reply:
    sector = extract_sector_from_query(query)
    if not sector:
        return Reply("business_query", "Please specify a sector for outlier analysis (e.g., 'Energy', 'Mining', 'Transport').", [])
    
    try:
        outlier_analysis = detect_financial_outliers(fact_company, fact_benchmark, sector)
        ranked_outliers = rank_outlier_significance(outlier_analysis, f"{sector} investment analysis")
        
        response_text = f"# Outlier Analysis: {sector} Sector\n\n"
        response_text += f"**Companies Analyzed**: {outlier_analysis.total_companies_analyzed}\n"
        response_text += f"**Detection Method**: {outlier_analysis.method_used}\n"
        response_text += f"**Confidence Level**: {outlier_analysis.confidence_level:.1%}\n\n"
        
        if not ranked_outliers.high_opportunity.empty:
            response_text += "## High Opportunity Outliers:\n"
            for _, row in ranked_outliers.high_opportunity.head(5).iterrows():
                response_text += f"**{row['company']}**: {row['details']} (Significance: {row['significance']:.1f})\n"
        
        if not ranked_outliers.high_risk.empty:
            response_text += "\n## High Risk Outliers:\n"
            for _, row in ranked_outliers.high_risk.head(5).iterrows():
                response_text += f"**{row['company']}**: {row['details']} (Significance: {row['significance']:.1f})\n"
        
        if not ranked_outliers.volatile.empty:
            response_text += "\n## High Volatility Outliers:\n"
            for _, row in ranked_outliers.volatile.head(3).iterrows():
                response_text += f"**{row['company']}**: {row['details']}\n"
        
        response_text += f"\n*Outlier significance scores help prioritize investment research and risk management focus.*"
        
        return Reply("business_query", response_text, ["resources/methodology.md"])
        
    except Exception as e:
        return Reply("business_query", f"Error detecting outliers in {sector}: {str(e)}", [])

def extract_sector_from_query(query: str) -> Optional[str]:
    sector_keywords = {
        'energy': ['energy', 'oil', 'gas', 'petroleum', 'exxon', 'chevron', 'bp', 'shell'],
        'utilities': ['utilities', 'utility', 'electric', 'power', 'grid'],
        'transport': ['transport', 'transportation', 'airlines', 'shipping', 'aviation'],
        'auto': ['auto', 'automotive', 'car', 'vehicle', 'tesla', 'ford', 'gm'],
        'cement': ['cement', 'concrete', 'construction', 'building materials'],
        'steel': ['steel', 'metal', 'iron', 'mining'],
        'chemicals': ['chemicals', 'chemical', 'petrochemical'],
        'aviation': ['aviation', 'airlines', 'aircraft', 'boeing', 'airbus']
    }
    
    query_lower = query.lower()
    
    for sector, keywords in sector_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return sector.title()
    
    sector_names = ['Energy', 'Utilities', 'Transport', 'Auto', 'Cement', 'Steel', 'Chemicals', 'Aviation', 'Mining']
    for sector in sector_names:
        if sector.lower() in query_lower:
            return sector
    
    return None

def extract_company_from_query(query: str, fact_company: pd.DataFrame) -> Optional[str]:
    
    if hasattr(fact_company.index, 'get_level_values'):
        companies = fact_company.index.get_level_values('company').unique().tolist()
    else:
        companies = fact_company['company'].unique().tolist() if 'company' in fact_company.columns else []
    
    query_lower = query.lower()
    
    for company in companies:
        if company.lower() in query_lower:
            return company
    
    company_abbrevs = {
        'tesla': 'Tesla',
        'apple': 'Apple',
        'microsoft': 'Microsoft', 
        'exxon': 'ExxonMobil',
        'chevron': 'Chevron',
        'bp': 'BP'
    }
    
    for abbrev, full_name in company_abbrevs.items():
        if abbrev in query_lower and full_name in companies:
            return full_name
    
    return None

def extract_scenarios_from_query(query: str) -> List[str]:
    scenario_keywords = {
        '1.5°C': ['1.5', '1.5°c', '1.5 degree', 'ambitious', 'paris'],
        'Below 2°C': ['below 2', 'under 2', '2°c', 'climate aligned'],
        'National Pledges': ['pledge', 'commitment', 'national', 'current policy']
    }
    
    query_lower = query.lower()
    scenarios = []
    
    for scenario, keywords in scenario_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            scenarios.append(scenario)
    
    return scenarios if scenarios else ["1.5°C", "Below 2°C"]
