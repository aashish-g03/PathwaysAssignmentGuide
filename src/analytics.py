import pandas as pd
import numpy as np
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
