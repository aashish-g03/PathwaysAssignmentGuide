import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from .analytics import view_pathway, view_sector_companies, calculate_climate_risk_score


@dataclass
class SectorDashboard:
    sector_rankings: pd.DataFrame
    trend_analysis: pd.DataFrame
    risk_distribution: pd.DataFrame
    data_coverage: pd.DataFrame
    last_updated: str

@dataclass 
class RiskHeatmap:
    heatmap_data: pd.DataFrame
    risk_scores: pd.DataFrame
    sector_averages: pd.DataFrame
    scenario_impact: pd.DataFrame

@dataclass
class TrendAnalysis:
    improving_companies: pd.DataFrame
    declining_companies: pd.DataFrame
    stable_companies: pd.DataFrame
    trend_statistics: pd.DataFrame
    sector_trajectories: pd.DataFrame

def create_sector_performance_summary(fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame,
                                    sector: str, scenarios: List[str] = None) -> SectorDashboard:
    if scenarios is None:
        scenarios = ["1.5°C", "Below 2°C"]
    
    # Validate that sector exists in benchmark data
    if hasattr(fact_benchmark.index, 'get_level_values'):
        available_benchmark_sectors = fact_benchmark.index.get_level_values('sector').unique()
    else:
        available_benchmark_sectors = fact_benchmark['sector'].unique() if 'sector' in fact_benchmark.columns else []
    
    if sector not in available_benchmark_sectors:
        return SectorDashboard(
            sector_rankings=pd.DataFrame(),
            trend_analysis=pd.DataFrame(), 
            risk_distribution=pd.DataFrame(),
            data_coverage=pd.DataFrame([{
                'sector': sector,
                'total_companies': 0,
                'companies_with_data': 0,
                'coverage_percentage': 0.0,
                'has_benchmark_data': False
            }]),
            last_updated=pd.Timestamp.now().isoformat()
        )
    
    sector_companies = view_sector_companies(fact_company, sector)
    if not sector_companies:
        return SectorDashboard(
            sector_rankings=pd.DataFrame(),
            trend_analysis=pd.DataFrame(), 
            risk_distribution=pd.DataFrame(),
            data_coverage=pd.DataFrame([{
                'sector': sector,
                'total_companies': 0,
                'companies_with_data': 0,
                'coverage_percentage': 0.0,
                'has_benchmark_data': True
            }]),
            last_updated=pd.Timestamp.now().isoformat()
        )
    
    company_summaries = []
    risk_scores = []
    trend_data = []
    
    year_range = (2015, 2035)
    
    for company in sector_companies:
        try:
            company_df, scen_map, _ = view_pathway(
                fact_company, fact_benchmark, sector, company, "Global",
                scenarios, year_range, exact_region=False
            )
            
            if company_df.empty or not scen_map:
                continue
            
            latest_year = company_df['year'].max()
            latest_intensity = company_df[company_df['year'] == latest_year]['intensity'].iloc[0]
            
            valid_data = company_df.dropna(subset=['intensity'])
            if len(valid_data) >= 2:
                trend_slope = np.polyfit(valid_data['year'], valid_data['intensity'], 1)[0]
            else:
                trend_slope = 0.0
            
            if scen_map:
                primary_scenario = "1.5°C" if "1.5°C" in scen_map else list(scen_map.keys())[0]
                benchmark_data = scen_map[primary_scenario]
                risk_score = calculate_climate_risk_score(company_df, benchmark_data)
                
                risk_scores.append({
                    'company': company,
                    'overall_risk_score': risk_score.overall_score,
                    'risk_category': risk_score.risk_category,
                    'confidence_level': risk_score.confidence_level
                })
            
            company_summaries.append({
                'company': company,
                'latest_intensity': latest_intensity,
                'trend_slope': trend_slope,
                'data_years': len(valid_data),
                'latest_year': latest_year
            })
            
            if trend_slope < -0.05:
                trend_category = 'improving'
            elif trend_slope > 0.05:  
                trend_category = 'declining'
            else:
                trend_category = 'stable'
            
            trend_data.append({
                'company': company,
                'trend_category': trend_category,
                'trend_slope': trend_slope
            })
            
        except Exception:
            continue
    
    sector_rankings = pd.DataFrame(company_summaries)
    if not sector_rankings.empty:
        sector_rankings = sector_rankings.sort_values('latest_intensity')
        sector_rankings['sector_rank'] = range(1, len(sector_rankings) + 1)
    
    trend_analysis = pd.DataFrame(trend_data)
    risk_distribution = pd.DataFrame(risk_scores)
    
    data_coverage = pd.DataFrame([{
        'sector': sector,
        'total_companies': len(sector_companies),
        'companies_with_data': len(company_summaries),
        'coverage_percentage': (len(company_summaries) / len(sector_companies)) * 100 if sector_companies else 0,
        'scenarios_analyzed': len(scenarios)
    }])
    
    return SectorDashboard(
        sector_rankings=sector_rankings,
        trend_analysis=trend_analysis,
        risk_distribution=risk_distribution,
        data_coverage=data_coverage,
        last_updated=pd.Timestamp.now().isoformat()
    )

def generate_climate_risk_heatmap(fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame,
                                sectors: List[str] = None, scenarios: List[str] = None) -> RiskHeatmap:
    if sectors is None:
        if hasattr(fact_company.index, 'get_level_values'):
            sectors = fact_company.index.get_level_values('sector').unique().tolist()
        else:
            sectors = fact_company['sector'].unique().tolist() if 'sector' in fact_company.columns else []
    
    if scenarios is None:
        scenarios = ["1.5°C", "Below 2°C", "National Pledges"]
    
    heatmap_data = []
    sector_averages = []
    risk_scores_all = []
    
    for sector in sectors:
        sector_risks = []
        sector_companies = view_sector_companies(fact_company, sector)
        
        if not sector_companies:
            continue
        
        for scenario in scenarios:
            scenario_risks = []
            
            for company in sector_companies[:10]:
                try:
                    company_df, scen_map, _ = view_pathway(
                        fact_company, fact_benchmark, sector, company, "Global",
                        [scenario], (2015, 2035), exact_region=False
                    )
                    
                    if company_df.empty or scenario not in scen_map:
                        continue
                    
                    benchmark_data = scen_map[scenario]
                    risk_score = calculate_climate_risk_score(company_df, benchmark_data)
                    
                    scenario_risks.append(risk_score.overall_score)
                    risk_scores_all.append({
                        'sector': sector,
                        'company': company,
                        'scenario': scenario,
                        'risk_score': risk_score.overall_score,
                        'risk_category': risk_score.risk_category
                    })
                    
                except Exception:
                    continue
            
            avg_risk = np.mean(scenario_risks) if scenario_risks else 0.0
            heatmap_data.append({
                'sector': sector,
                'scenario': scenario,
                'average_risk_score': avg_risk,
                'companies_analyzed': len(scenario_risks)
            })
            
            sector_risks.extend(scenario_risks)
        
        if sector_risks:
            sector_averages.append({
                'sector': sector,
                'overall_risk_score': np.mean(sector_risks),
                'risk_std': np.std(sector_risks),
                'companies_analyzed': len(set([r['company'] for r in risk_scores_all if r['sector'] == sector]))
            })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    if not heatmap_df.empty:
        heatmap_pivot = heatmap_df.pivot(index='sector', columns='scenario', values='average_risk_score')
    else:
        heatmap_pivot = pd.DataFrame()
    
    scenario_impact = []
    if not heatmap_df.empty:
        for scenario in scenarios:
            scenario_data = heatmap_df[heatmap_df['scenario'] == scenario]
            if not scenario_data.empty:
                scenario_impact.append({
                    'scenario': scenario,
                    'average_risk': scenario_data['average_risk_score'].mean(),
                    'risk_range': scenario_data['average_risk_score'].max() - scenario_data['average_risk_score'].min(),
                    'sectors_analyzed': len(scenario_data)
                })
    
    return RiskHeatmap(
        heatmap_data=heatmap_pivot,
        risk_scores=pd.DataFrame(risk_scores_all),
        sector_averages=pd.DataFrame(sector_averages),
        scenario_impact=pd.DataFrame(scenario_impact)
    )


def generate_executive_summary(fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame,
                             sectors: List[str] = None) -> Dict[str, Any]:
    if sectors is None:
        # Only use sectors that have both company data AND benchmark data
        if hasattr(fact_company.index, 'get_level_values'):
            company_sectors = set(fact_company.index.get_level_values('sector').unique())
        else:
            company_sectors = set(fact_company['sector'].unique()) if 'sector' in fact_company.columns else set()
        
        if hasattr(fact_benchmark.index, 'get_level_values'):
            benchmark_sectors = set(fact_benchmark.index.get_level_values('sector').unique())
        else:
            benchmark_sectors = set(fact_benchmark['sector'].unique()) if 'sector' in fact_benchmark.columns else set()
        
        # Only analyze sectors with both company and benchmark data
        valid_sectors = company_sectors & benchmark_sectors
        
        if not valid_sectors:
            sectors = []
        else:
            # Get sector counts only for valid sectors
            if hasattr(fact_company.index, 'get_level_values'):
                sector_filter = fact_company.index.get_level_values('sector').isin(valid_sectors)
                sector_counts = fact_company[sector_filter].index.get_level_values('sector').value_counts()
            else:
                sector_filter = fact_company['sector'].isin(valid_sectors)
                sector_counts = fact_company[sector_filter]['sector'].value_counts()
            
            sectors = sector_counts.head(5).index.tolist()
    
    executive_summary = {
        'total_sectors_analyzed': len(sectors),
        'analysis_date': pd.Timestamp.now().isoformat(),
        'key_findings': [],
        'risk_alerts': [],
        'opportunities': []
    }
    
    for sector in sectors:
        try:
            dashboard = create_sector_performance_summary(fact_company, fact_benchmark, sector)
            
            if dashboard.risk_distribution.empty:
                continue
            
            high_risk_count = len(dashboard.risk_distribution[
                dashboard.risk_distribution['risk_category'] == 'HIGH_RISK'
            ])
            
            total_companies = len(dashboard.risk_distribution)
            risk_percentage = (high_risk_count / total_companies) * 100 if total_companies > 0 else 0
            
            executive_summary['key_findings'].append({
                'sector': sector,
                'companies_analyzed': total_companies,
                'high_risk_percentage': round(risk_percentage, 1),
                'data_coverage': dashboard.data_coverage['coverage_percentage'].iloc[0] if not dashboard.data_coverage.empty else 0
            })
            
            if risk_percentage > 50:
                executive_summary['risk_alerts'].append({
                    'sector': sector,
                    'alert_type': 'high_sector_risk',
                    'message': f"{sector} sector has {risk_percentage:.1f}% companies in HIGH_RISK category"
                })
            
            if not dashboard.trend_analysis.empty:
                improving_count = len(dashboard.trend_analysis[
                    dashboard.trend_analysis['trend_category'] == 'improving'
                ])
                
                if improving_count > 0:
                    executive_summary['opportunities'].append({
                        'sector': sector,
                        'opportunity_type': 'improving_trends',
                        'companies_improving': improving_count,
                        'message': f"{improving_count} companies in {sector} showing positive climate trends"
                    })
                    
        except Exception:
            continue
    
    return executive_summary
