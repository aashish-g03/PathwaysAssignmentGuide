import numpy as np
import pandas as pd

def calculate_cumulative_benchmark_deviation(company_data: pd.DataFrame, benchmark_data: pd.DataFrame, start_year=2020, end_year=2035) -> float:
    company_filtered = company_data.query('Year >= @start_year and Year <= @end_year').sort_values('Year')
    if company_filtered.empty: 
        return float('nan')
    
    if company_filtered['Year'].duplicated().any():
        company_aggregated = company_filtered.groupby('Year', as_index=False)['Intensity'].mean()
    else:
        company_aggregated = company_filtered
    
    if benchmark_data['Year'].duplicated().any():
        benchmark_aggregated = benchmark_data.groupby('Year', as_index=False)['Benchmark'].mean()
    else:
        benchmark_aggregated = benchmark_data
        
    benchmark_aligned = (
        benchmark_aggregated
        .set_index('Year')
        .reindex(company_aggregated['Year'])
        .reset_index()
        .rename(columns={'Benchmark': 'benchmark_value'})
    )
    
    if benchmark_aligned['benchmark_value'].isna().all(): 
        return float('nan')
    
    intensity_diff = company_aggregated['Intensity'].values - benchmark_aligned['benchmark_value'].values
    
    return float(np.trapz(intensity_diff, company_aggregated['Year'].values))

def calculate_sector_outliers(company_table: pd.DataFrame, benchmark_table: pd.DataFrame, sector: str, scenario='1.5°C', region_pref=None, k=10, start=2020, end=2035):
    sector_companies = company_table.query('Sector == @sector')
    
    benchmark_query_conditions = ['Sector == @sector']
    
    if 'Scenario' in benchmark_table.columns:
        benchmark_query_conditions.append('Scenario == @scenario')
    
    base_benchmark = benchmark_table.query(' and '.join(benchmark_query_conditions))
    
    if region_pref and 'Region' in benchmark_table.columns:
        regional_benchmark = base_benchmark.query('Region == @region_pref')
        
        if regional_benchmark.empty:
            global_benchmark = base_benchmark.query('Region == "Global"')
            final_benchmark = global_benchmark
        else:
            final_benchmark = regional_benchmark
    else:
        final_benchmark = base_benchmark
    
    company_scores = []
    for company_name, company_group in sector_companies.groupby('Company'):
        company_subset = company_group[['Year', 'Intensity']]
        benchmark_subset = final_benchmark[['Year', 'Benchmark']]
        
        cbd_value = calculate_cumulative_benchmark_deviation(
            company_subset, benchmark_subset, 
            start_year=start, end_year=end
        )
        company_scores.append({'Company': company_name, 'CBD': cbd_value})
    
    results_table = pd.DataFrame(company_scores).dropna()
    
    if results_table.empty:
        empty_result = pd.DataFrame(columns=['Company', 'CBD', 'z_score'])
        return empty_result, empty_result
    
    if len(results_table) > 1:
        cbd_mean = results_table['CBD'].mean()
        cbd_std = results_table['CBD'].std(ddof=1)
        results_table = results_table.assign(
            z_score=(results_table['CBD'] - cbd_mean) / cbd_std
        )
    else:
        results_table = results_table.assign(z_score=0.0)
    
    best_performers = results_table.nsmallest(k, 'CBD')
    worst_performers = results_table.nlargest(k, 'CBD')
    
    return best_performers, worst_performers
