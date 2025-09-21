import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from .io_utils import join_company_benchmark

def view_sector_companies_by_country(fact_company, sector, country=None):
    query = fact_company.index.get_level_values('sector') == sector
    sector_data = fact_company.loc[query]
    
    if country and country != "Global" and "geography" in sector_data.columns:
        sector_data = sector_data[sector_data["geography"] == country]
    
    return sorted(sector_data.index.get_level_values('company').unique().tolist())


def view_country_options_direct(fact_company, sector):
    sector_data = fact_company[fact_company.index.get_level_values('sector') == sector]
    
    if "geography" not in sector_data.columns:
        return ["Global"]
    
    countries = sector_data["geography"].dropna().unique().tolist()
    return ["Global"] + sorted(countries)

def view_country_options(fact_company, sector, company_region=None):
    sector_data = fact_company[fact_company.index.get_level_values('sector') == sector]
    
    if company_region and company_region != "Global (All)" and "companyregion" in sector_data.columns:
        sector_data = sector_data[sector_data["companyregion"] == company_region]
    
    if "geography" not in sector_data.columns:
        return ["(Any)"]
    
    countries = sector_data["geography"].dropna().unique().tolist()
    return ["(Any)"] + sorted(countries)

def view_pathway(fact_company, fact_benchmark, sector, company, region, scenarios, year_range, subsector=None, exact_region=False):
    
    # Join company and scenario data
    company_df, scen_map = join_company_benchmark(
        fact_company, fact_benchmark, sector, company, 
        region, scenarios, year_range, subsector, exact_region
    )
    
    bands = calculate_envelope_bands(scen_map, company_df, year_range)
    
    return company_df, scen_map, bands

def calculate_envelope_bands(scen_map, company_df, year_range):
    
    y_start, y_end = year_range
    years = list(range(y_start, y_end + 1))
    
    bands = {}
    
    # Below 1.5°C
    if '1.5°C' in scen_map and not scen_map['1.5°C'].empty:
        green_data = scen_map['1.5°C']
        
        company_min = company_df['intensity'].min() if not company_df.empty else 0.0
        green_min = green_data['benchmark'].min() if not green_data.empty else 0.0
        baseline = max(0.0, min(green_min, company_min))
        
        green_reindexed = green_data.set_index('year').reindex(years).reset_index()
        bands['below_1_5'] = pd.DataFrame({
            'year': years,
            'lower': baseline,
            'upper': green_reindexed['benchmark'].ffill().bfill()
        })
    
    # Between 1.5°C and 2°C
    twoc_scenarios = [s for s in scen_map.keys() if '2°C' in s or 'Below 2°C' in s]
    green_available = '1.5°C' in scen_map and not scen_map['1.5°C'].empty
    
    if green_available and twoc_scenarios:
        green_data = scen_map['1.5°C'].set_index('year').reindex(years)
        
        twoc_frames = []
        for scenario in twoc_scenarios:
            if scenario in scen_map and not scen_map[scenario].empty:
                scenario_data = scen_map[scenario].set_index('year').reindex(years)
                twoc_frames.append(scenario_data['benchmark'])
        
        if twoc_frames:
            twoc_combined = pd.concat(twoc_frames, axis=1)
            twoc_min = twoc_combined.min(axis=1)
            twoc_max = twoc_combined.max(axis=1)
            
            bands['between_1_5_2'] = pd.DataFrame({
                'year': years,
                'lower': np.minimum(green_data['benchmark'], twoc_min).ffill().bfill(),
                'upper': np.maximum(green_data['benchmark'], twoc_max).ffill().bfill()
            })
    
    # Above 2°C
    pledge_scenarios = [s for s in scen_map.keys() if 'Pledge' in s]
    
    if twoc_scenarios and pledge_scenarios:
        twoc_frames = []
        for scenario in twoc_scenarios:
            if scenario in scen_map and not scen_map[scenario].empty:
                scenario_data = scen_map[scenario].set_index('year').reindex(years)
                twoc_frames.append(scenario_data['benchmark'])
        
        pledge_frames = []
        for scenario in pledge_scenarios:
            if scenario in scen_map and not scen_map[scenario].empty:
                scenario_data = scen_map[scenario].set_index('year').reindex(years)
                pledge_frames.append(scenario_data['benchmark'])
        
        if twoc_frames and pledge_frames:
            twoc_combined = pd.concat(twoc_frames, axis=1)
            twoc_upper = twoc_combined.max(axis=1)
            
            pledge_combined = pd.concat(pledge_frames, axis=1)
            pledge_min = pledge_combined.min(axis=1)
            pledge_max = pledge_combined.max(axis=1)
            
            bands['above_2'] = pd.DataFrame({
                'year': years,
                'lower': np.minimum(twoc_upper, pledge_min).ffill().bfill(),
                'upper': np.maximum(twoc_upper, pledge_max).ffill().bfill()
            })
    
    return bands


def view_sector_companies(fact_company, sector):
    sector_filter = fact_company.index.get_level_values('sector') == sector
    companies = fact_company.loc[sector_filter].index.get_level_values('company').unique()
    return sorted(companies)

def view_company_subsectors(fact_company, sector, company):
    company_filter = (
        (fact_company.index.get_level_values('sector') == sector) &
        (fact_company.index.get_level_values('company') == company)
    )
    company_records = fact_company.loc[company_filter].reset_index()
    
    if 'subsector' in company_records.columns:
        subsectors = company_records['subsector'].dropna().unique()
        return sorted(subsectors)
    return []

def view_sector_regions_scenarios(fact_benchmark, sector):
    sector_filter = fact_benchmark.index.get_level_values('sector') == sector
    
    if sector_filter.any():
        sector_data = fact_benchmark.loc[sector_filter]
        regions = sorted(sector_data.index.get_level_values('region').unique())
        scenarios = sorted(sector_data.index.get_level_values('scenario').unique())
        return regions, scenarios
    return [], []

def view_company_year_bounds(fact_company, sector, company, subsector=None):
    company_filter = (
        (fact_company.index.get_level_values('sector') == sector) &
        (fact_company.index.get_level_values('company') == company)
    )
    
    company_data = fact_company.loc[company_filter].reset_index()
    
    if subsector and 'subsector' in company_data.columns:
        company_data = company_data[company_data['subsector'] == subsector]
    
    if company_data.empty:
        return None, None
    
    year_bounds = company_data['year'].agg(['min', 'max'])
    return int(year_bounds['min']), int(year_bounds['max'])


@dataclass
class RiskScore:
    overall_score: float
    trajectory_score: float  
    alignment_score: float
    data_quality_score: float
    risk_category: str
    confidence_level: float


def calculate_climate_risk_score(company_data: pd.DataFrame, benchmark_data: pd.DataFrame, 
                               target_scenario: str = "1.5°C") -> RiskScore:
    if company_data.empty or benchmark_data.empty:
        return RiskScore(0.0, 0.0, 0.0, 0.0, "HIGH_RISK", 0.0)
    
    valid_company_points = company_data['intensity'].notna().sum()
    total_company_points = len(company_data)
    data_quality_score = valid_company_points / total_company_points if total_company_points > 0 else 0.0
    
    if data_quality_score < 0.5:
        return RiskScore(0.0, 0.0, 0.0, data_quality_score, "HIGH_RISK", 0.0)
    
    valid_data = company_data.dropna(subset=['intensity']).sort_values('year')
    if len(valid_data) < 2:
        trajectory_score = 0.0
    else:
        years = valid_data['year'].values
        intensities = valid_data['intensity'].values
        slope = np.polyfit(years, intensities, 1)[0]
        trajectory_score = max(0.0, min(1.0, -slope * 10))
    
        merged_data = company_data.merge(benchmark_data, on='year', how='inner')
    if merged_data.empty:
        alignment_score = 0.0
    else:
        distance = merged_data['intensity'] - merged_data['benchmark']
        avg_distance = distance.mean()
        alignment_score = max(0.0, min(1.0, 1.0 - (avg_distance / merged_data['benchmark'].mean())))
    
    overall_score = (
        0.4 * alignment_score +
        0.3 * trajectory_score + 
        0.3 * data_quality_score
    )
    
    if overall_score >= 0.75:
        risk_category = "LOW_RISK"
    elif overall_score >= 0.5:
        risk_category = "MEDIUM_RISK"
    else:
        risk_category = "HIGH_RISK"
    
    confidence_level = data_quality_score * (1.0 if len(valid_data) >= 5 else 0.5)
    
    return RiskScore(
        overall_score=round(overall_score, 3),
        trajectory_score=round(trajectory_score, 3),
        alignment_score=round(alignment_score, 3),
        data_quality_score=round(data_quality_score, 3),
        risk_category=risk_category,
        confidence_level=round(confidence_level, 3)
    )
    
@dataclass 
class OutlierAnalysis:
    statistical_outliers: pd.DataFrame
    trend_outliers: pd.DataFrame  
    performance_outliers: pd.DataFrame
    volatility_outliers: pd.DataFrame
    method_used: str
    confidence_level: float
    total_companies_analyzed: int

@dataclass
class RankedOutliers:
    high_opportunity: pd.DataFrame
    high_risk: pd.DataFrame
    volatile: pd.DataFrame
    significance_scores: pd.DataFrame
    business_context: str

def detect_financial_outliers(fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame,
                            sector: str, scenarios: List[str] = None,
                            method: str = 'multi_metric', confidence: float = 0.95) -> OutlierAnalysis:
    if scenarios is None:
        scenarios = ["1.5°C", "Below 2°C"]
    
    sector_companies = view_sector_companies(fact_company, sector)
    if not sector_companies:
        return OutlierAnalysis(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 
                             pd.DataFrame(), method, confidence, 0)
    
    year_range = (2015, 2035)
    
    company_analyses = []
    
    for company in sector_companies:
        try:
            company_df, scen_map, _ = view_pathway(
                fact_company, fact_benchmark, sector, company, "Global",
                scenarios, year_range, exact_region=False
            )
            
            if company_df.empty or not scen_map:
                continue
                
            analysis = calculate_company_outlier_metrics(company, company_df, scen_map)
            if analysis:
                company_analyses.append(analysis)
                
        except Exception:
            continue
    
    if not company_analyses:
        return OutlierAnalysis(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                             pd.DataFrame(), method, confidence, 0)
    
    metrics_df = pd.DataFrame(company_analyses)
    
    statistical_outliers = detect_statistical_outliers(metrics_df, confidence)
    trend_outliers = detect_trend_outliers(metrics_df, confidence)
    performance_outliers = detect_performance_outliers(metrics_df, confidence)
    volatility_outliers = detect_volatility_outliers(metrics_df, confidence)
    
    return OutlierAnalysis(
        statistical_outliers=statistical_outliers,
        trend_outliers=trend_outliers,
        performance_outliers=performance_outliers,
        volatility_outliers=volatility_outliers,
        method_used=method,
        confidence_level=confidence,
        total_companies_analyzed=len(company_analyses)
    )

def calculate_company_outlier_metrics(company: str, company_df: pd.DataFrame, 
                                    scen_map: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
    if company_df.empty or not scen_map:
        return None
    
    valid_data = company_df.dropna(subset=['intensity'])
    if len(valid_data) < 2:
        return None
    
    intensity_mean = valid_data['intensity'].mean()
    intensity_std = valid_data['intensity'].std()
    intensity_trend = np.polyfit(valid_data['year'], valid_data['intensity'], 1)[0]
    
    benchmark_scenario = "1.5°C" if "1.5°C" in scen_map else list(scen_map.keys())[0]
    benchmark_data = scen_map[benchmark_scenario]
    
    benchmark_deviation = 0.0
    cbd_score = 0.0
    
    if not benchmark_data.empty:
        merged = company_df.merge(benchmark_data, on='year', how='inner')
        if not merged.empty:
            benchmark_deviation = (merged['intensity'] - merged['benchmark']).mean()
            cbd_score = np.trapezoid(
                merged['intensity'].values - merged['benchmark'].values,
                merged['year'].values
            )
    
    volatility = intensity_std / intensity_mean if intensity_mean != 0 else 0.0
    
    data_quality = len(valid_data) / len(company_df)
    
    return {
        'company': company,
        'intensity_mean': intensity_mean,
        'intensity_std': intensity_std,
        'intensity_trend': intensity_trend,
        'benchmark_deviation': benchmark_deviation,
        'cbd_score': cbd_score,
        'volatility': volatility,
        'data_quality': data_quality,
        'data_points': len(valid_data)
    }

def detect_statistical_outliers(metrics_df: pd.DataFrame, confidence: float) -> pd.DataFrame:
    
    outliers = []
    z_threshold = 2.0 if confidence < 0.95 else 2.5
    
    for metric in ['intensity_mean', 'intensity_trend', 'benchmark_deviation', 'cbd_score']:
        if metric not in metrics_df.columns:
            continue
            
        values = metrics_df[metric].dropna()
        if len(values) < 3:
            continue
        
        z_scores = np.abs((values - values.mean()) / values.std())
        z_outliers = metrics_df.loc[values.index[z_scores > z_threshold]]
        
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = metrics_df[(values < lower_bound) | (values > upper_bound)]
        
        for _, row in z_outliers.iterrows():
            outliers.append({
                'company': row['company'],
                'metric': metric,
                'value': row[metric],
                'method': 'z_score',
                'z_score': z_scores[row.name],
                'outlier_type': 'high' if row[metric] > values.mean() else 'low'
            })
        
        for _, row in iqr_outliers.iterrows():
            outliers.append({
                'company': row['company'],
                'metric': metric,
                'value': row[metric],
                'method': 'iqr',
                'z_score': None,
                'outlier_type': 'high' if row[metric] > values.median() else 'low'
            })
    
    return pd.DataFrame(outliers).drop_duplicates(['company', 'metric', 'method'])

def detect_trend_outliers(metrics_df: pd.DataFrame, confidence: float) -> pd.DataFrame:
    
    if 'intensity_trend' not in metrics_df.columns:
        return pd.DataFrame()
    
    trends = metrics_df['intensity_trend'].dropna()
    if len(trends) < 3:
        return pd.DataFrame()
    
    median_trend = trends.median()
    trend_std = trends.std()
    threshold = 2.0 * trend_std
    
    outlier_companies = metrics_df[
        np.abs(metrics_df['intensity_trend'] - median_trend) > threshold
    ].copy()
    
    outlier_companies['trend_deviation'] = outlier_companies['intensity_trend'] - median_trend
    outlier_companies['trend_type'] = outlier_companies['trend_deviation'].apply(
        lambda x: 'improving_fast' if x < -threshold else 'worsening_fast'
    )
    
    return outlier_companies[['company', 'intensity_trend', 'trend_deviation', 'trend_type']]

def detect_performance_outliers(metrics_df: pd.DataFrame, confidence: float) -> pd.DataFrame:
    
    if 'benchmark_deviation' not in metrics_df.columns:
        return pd.DataFrame()
    
    deviations = metrics_df['benchmark_deviation'].dropna()
    if len(deviations) < 3:
        return pd.DataFrame()
    
    q5 = deviations.quantile(0.05)
    q95 = deviations.quantile(0.95)
    
    performance_outliers = metrics_df[
        (metrics_df['benchmark_deviation'] <= q5) | 
        (metrics_df['benchmark_deviation'] >= q95)
    ].copy()
    
    performance_outliers['performance_category'] = performance_outliers['benchmark_deviation'].apply(
        lambda x: 'high_performer' if x <= q5 else 'poor_performer'
    )
    
    return performance_outliers[['company', 'benchmark_deviation', 'cbd_score', 'performance_category']]

def detect_volatility_outliers(metrics_df: pd.DataFrame, confidence: float) -> pd.DataFrame:
    
    if 'volatility' not in metrics_df.columns:
        return pd.DataFrame()
    
    volatilities = metrics_df['volatility'].dropna()
    if len(volatilities) < 3:
        return pd.DataFrame()
    
    high_vol_threshold = volatilities.quantile(0.9)
    high_vol_outliers = metrics_df[metrics_df['volatility'] > high_vol_threshold].copy()
    
    high_vol_outliers['volatility_type'] = 'high_volatility'
    
    return high_vol_outliers[['company', 'volatility', 'intensity_std', 'volatility_type']]

def rank_outlier_significance(outliers: OutlierAnalysis, business_context: str) -> RankedOutliers:
    
    all_outliers = []
    
    for _, row in outliers.statistical_outliers.iterrows():
        metric = row['metric']
        outlier_type = row['outlier_type']
        
        if metric == 'intensity_mean':
            detail = "Unusually low emissions intensity" if outlier_type == 'low' else "Unusually high emissions intensity"
        elif metric == 'intensity_trend':
            detail = "Rapidly improving trajectory" if outlier_type == 'low' else "Rapidly worsening trajectory"
        elif metric == 'benchmark_deviation':
            detail = "Far ahead of climate targets" if outlier_type == 'low' else "Far behind climate targets"
        elif metric == 'cbd_score':
            detail = "Strong cumulative performance vs benchmarks" if outlier_type == 'low' else "Poor cumulative performance vs benchmarks"
        else:
            detail = f"Statistical anomaly in {metric}"
            
        all_outliers.append({
            'company': row['company'],
            'outlier_source': 'statistical',
            'significance': abs(row.get('z_score', 1.0)),
            'details': detail
        })
    
    for _, row in outliers.trend_outliers.iterrows():
        significance = abs(row['trend_deviation']) * 10
        trend_type = row['trend_type']
        
        if trend_type == 'improving_fast':
            detail = "Accelerating climate performance improvement"
        elif trend_type == 'worsening_fast':
            detail = "Concerning performance decline"
        else:
            detail = f"Unusual trend pattern: {trend_type}"
            
        all_outliers.append({
            'company': row['company'],
            'outlier_source': 'trend',
            'significance': significance,
            'details': detail
        })
    
    for _, row in outliers.performance_outliers.iterrows():
        significance = abs(row['benchmark_deviation']) * 5
        perf_category = row['performance_category']
        
        if perf_category == 'high_performer':
            detail = "Top climate performance vs sector benchmarks"
        elif perf_category == 'poor_performer':
            detail = "Significantly underperforming climate benchmarks"
        else:
            detail = f"Performance anomaly: {perf_category}"
            
        all_outliers.append({
            'company': row['company'],
            'outlier_source': 'performance',
            'significance': significance,
            'details': detail
        })
    
    for _, row in outliers.volatility_outliers.iterrows():
        detail = "High data volatility - inconsistent reporting or operational changes"
        all_outliers.append({
            'company': row['company'],
            'outlier_source': 'volatility',
            'significance': row['volatility'] * 3,
            'details': detail
        })
    
    if not all_outliers:
        return RankedOutliers(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                            pd.DataFrame(), business_context)
    
    significance_df = pd.DataFrame(all_outliers)
    
    company_significance = significance_df.groupby('company').agg({
        'significance': 'sum',
        'details': lambda x: ' | '.join(x)
    }).reset_index()
    
    company_significance = company_significance.sort_values('significance', ascending=False)
    
    performance_data = outliers.performance_outliers.set_index('company') if not outliers.performance_outliers.empty else pd.DataFrame()
    trend_data = outliers.trend_outliers.set_index('company') if not outliers.trend_outliers.empty else pd.DataFrame()
    
    high_opportunity = []
    high_risk = []
    volatile = []
    
    for _, row in company_significance.iterrows():
        company = row['company']
        
        is_high_performer = (company in performance_data.index and 
                           performance_data.loc[company, 'performance_category'] == 'high_performer')
        is_improving = (company in trend_data.index and 
                       trend_data.loc[company, 'trend_type'] == 'improving_fast')
        
        is_poor_performer = (company in performance_data.index and 
                           performance_data.loc[company, 'performance_category'] == 'poor_performer')
        is_worsening = (company in trend_data.index and 
                       trend_data.loc[company, 'trend_type'] == 'worsening_fast')
        
        is_volatile = 'volatility' in row['details']
        
        if is_high_performer or is_improving:
            high_opportunity.append(row)
        elif is_poor_performer or is_worsening:
            high_risk.append(row)
        elif is_volatile:
            volatile.append(row)
    
    return RankedOutliers(
        high_opportunity=pd.DataFrame(high_opportunity).head(10),
        high_risk=pd.DataFrame(high_risk).head(10),
        volatile=pd.DataFrame(volatile).head(10),
        significance_scores=company_significance,
        business_context=business_context
    )
