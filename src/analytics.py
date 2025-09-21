import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from .io_utils import join_company_benchmark

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

def view_outliers(fact_company, fact_benchmark, sector, scenario, region, year_range, exact_region=False, k=10):
    
    y_start, y_end = year_range
    
    # Get sector companies
    sector_filter = fact_company.index.get_level_values('sector') == sector
    year_filter = (
        (fact_company.index.get_level_values('year') >= y_start) &
        (fact_company.index.get_level_values('year') <= y_end)
    )
    
    sector_companies = fact_company.loc[sector_filter & year_filter].reset_index()
    
    # Get benchmark data
    bench_filter = (
        (fact_benchmark.index.get_level_values('sector') == sector) &
        (fact_benchmark.index.get_level_values('scenario') == scenario) &
        (fact_benchmark.index.get_level_values('year') >= y_start) &
        (fact_benchmark.index.get_level_values('year') <= y_end)
    )
    
    if region and not exact_region:
        region_filter = bench_filter & (fact_benchmark.index.get_level_values('region') == region)
        benchmark_data = fact_benchmark.loc[region_filter].reset_index()
        
        if benchmark_data.empty:
            global_filter = bench_filter & (fact_benchmark.index.get_level_values('region') == 'Global')
            benchmark_data = fact_benchmark.loc[global_filter].reset_index()
    else:
        benchmark_data = fact_benchmark.loc[bench_filter].reset_index()
    
    if benchmark_data.empty:
        empty_result = pd.DataFrame(columns=['company', 'cbd', 'z_score'])
        return empty_result, empty_result
    
    # Join company and benchmark data
    joined_table = sector_companies.merge(
        benchmark_data[['year', 'benchmark']],
        on='year',
        how='left'
    )
    
    # Calculate CBD
    company_scores = calculate_cbd_vectorized(joined_table)
    
    # Statistical analysis
    if len(company_scores) > 1:
        cbd_mean = company_scores['cbd'].mean()
        cbd_std = company_scores['cbd'].std(ddof=1)
        company_scores['z_score'] = (company_scores['cbd'] - cbd_mean) / cbd_std
    else:
        company_scores['z_score'] = 0.0
    
    # Select top and bottom performers
    best_performers = company_scores.nsmallest(k, 'cbd')
    worst_performers = company_scores.nlargest(k, 'cbd')
    
    return best_performers, worst_performers

def calculate_cbd_vectorized(joined_table):
    """Calculate CBD using vectorized operations equivalent to window functions"""
    
    # Group by company and calculate trapezoid areas
    company_scores = []
    
    for company_name, company_group in joined_table.groupby('company'):
        company_sorted = company_group.sort_values('year').reset_index(drop=True)
        
        if len(company_sorted) < 2:
            continue
            
        intensity_diff = company_sorted['intensity'] - company_sorted['benchmark']
        
        company_sorted['diff'] = intensity_diff
        company_sorted['diff_next'] = company_sorted['diff'].shift(-1)
        company_sorted['year_next'] = company_sorted['year'].shift(-1)
        
        company_sorted['segment_area'] = (
            (company_sorted['diff'] + company_sorted['diff_next']) / 2.0 *
            (company_sorted['year_next'] - company_sorted['year'])
        )
        
        cbd_value = company_sorted['segment_area'].sum()
        
        company_scores.append({
            'company': company_name,
            'cbd': cbd_value
        })
    
    return pd.DataFrame(company_scores).dropna()

def sector_availability(fact_benchmark):
    
    availability = (
        fact_benchmark.reset_index()
        .groupby('sector')
        .agg({
            'scenario': lambda x: sorted(x.unique()),
            'region': lambda x: sorted(x.unique()),
            'year': ['min', 'max']
        })
        .reset_index()
    )
    
    availability.columns = ['sector', 'scenarios', 'regions', 'year_min', 'year_max']
    
    return availability

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

@dataclass
class ReadinessMetric:
    readiness_score: float
    gap_to_target: float
    years_to_alignment: Optional[int]
    trajectory_slope: float
    benchmark_scenario: str

@dataclass
class PeerAnalysis:
    peer_rank: int
    total_peers: int
    percentile: float
    performance_vs_median: float
    sector_leaders: List[str]
    sector_laggards: List[str]

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
        trajectory_score = max(0.0, min(1.0, -slope * 10))  # Normalize and invert
    
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

def compute_transition_readiness(sector: str, company: str, scenarios: List[str],
                               fact_company: pd.DataFrame, fact_benchmark: pd.DataFrame,
                               target_year: int = 2030) -> ReadinessMetric:
    
    company_df, scen_map, _ = view_pathway(
        fact_company, fact_benchmark, sector, company, "Global", 
        scenarios, (2015, target_year), exact_region=False
    )
    
    if company_df.empty or not scen_map:
        return ReadinessMetric(0.0, float('inf'), None, 0.0, "N/A")
    
    benchmark_scenario = "1.5°C" if "1.5°C" in scen_map else list(scen_map.keys())[0]
    benchmark_data = scen_map[benchmark_scenario]
    
    if benchmark_data.empty:
        return ReadinessMetric(0.0, float('inf'), None, 0.0, benchmark_scenario)
    
    latest_year = company_df['year'].max()
    target_benchmark = benchmark_data[benchmark_data['year'] == target_year]
    current_company = company_df[company_df['year'] == latest_year]
    
    if target_benchmark.empty or current_company.empty:
        return ReadinessMetric(0.0, float('inf'), None, 0.0, benchmark_scenario)
    
    target_intensity = target_benchmark['benchmark'].iloc[0]
    current_intensity = current_company['intensity'].iloc[0]
    gap_to_target = current_intensity - target_intensity
    
    valid_data = company_df.dropna(subset=['intensity']).sort_values('year')
    if len(valid_data) >= 2:
        years = valid_data['year'].values
        intensities = valid_data['intensity'].values
        trajectory_slope = np.polyfit(years, intensities, 1)[0]
    else:
        trajectory_slope = 0.0
    
    years_to_alignment = None
    if trajectory_slope < 0 and gap_to_target > 0:
        years_to_alignment = int(gap_to_target / abs(trajectory_slope))
        years_to_alignment = min(years_to_alignment, 50)
    
    readiness_score = 0.0
    if gap_to_target <= 0:
        readiness_score = 1.0
    elif trajectory_slope < 0:
        years_remaining = target_year - latest_year
        required_slope = gap_to_target / years_remaining if years_remaining > 0 else float('inf')
        if abs(trajectory_slope) >= required_slope:
            readiness_score = 0.8
        else:
            readiness_score = min(0.7, abs(trajectory_slope) / required_slope)
    else:
        readiness_score = max(0.1, 1.0 / (1.0 + gap_to_target))
    
    return ReadinessMetric(
        readiness_score=round(readiness_score, 3),
        gap_to_target=round(gap_to_target, 3),
        years_to_alignment=years_to_alignment,
        trajectory_slope=round(trajectory_slope, 4),
        benchmark_scenario=benchmark_scenario
    )

def generate_peer_comparison_metrics(company: str, sector_peers: pd.DataFrame,
                                   benchmark_data: pd.DataFrame,
                                   comparison_year: int = 2023) -> PeerAnalysis:
    if sector_peers.empty:
        return PeerAnalysis(0, 0, 0.0, 0.0, [], [])
    
    year_data = sector_peers[sector_peers['year'] == comparison_year]
    if year_data.empty:
        latest_year = sector_peers['year'].max()
        year_data = sector_peers[sector_peers['year'] == latest_year]
    
    if year_data.empty:
        return PeerAnalysis(0, 0, 0.0, 0.0, [], [])
    
    company_data = year_data[year_data['company'] == company]
    if company_data.empty:
        return PeerAnalysis(0, len(year_data), 0.0, 0.0, [], [])
    
    company_intensity = company_data['intensity'].iloc[0]
    if pd.isna(company_intensity):
        return PeerAnalysis(0, len(year_data), 0.0, 0.0, [], [])
    
    valid_peers = year_data.dropna(subset=['intensity']).sort_values('intensity')
    total_peers = len(valid_peers)
    
    if total_peers == 0:
        return PeerAnalysis(0, 0, 0.0, 0.0, [], [])
    
    peer_rank = (valid_peers['intensity'] <= company_intensity).sum()
    percentile = (peer_rank / total_peers) * 100
    
    median_intensity = valid_peers['intensity'].median()
    performance_vs_median = company_intensity - median_intensity
    
    top_quartile = int(total_peers * 0.25)
    bottom_quartile = int(total_peers * 0.75)
    
    sector_leaders = valid_peers.head(max(1, top_quartile))['company'].tolist()
    sector_laggards = valid_peers.tail(max(1, total_peers - bottom_quartile))['company'].tolist()
    
    return PeerAnalysis(
        peer_rank=peer_rank,
        total_peers=total_peers,
        percentile=round(percentile, 1),
        performance_vs_median=round(performance_vs_median, 3),
        sector_leaders=sector_leaders,
        sector_laggards=sector_laggards
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
