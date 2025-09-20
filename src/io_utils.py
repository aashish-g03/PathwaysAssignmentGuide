import pandas as pd
import numpy as np
import os
from pathlib import Path

def find_year_columns(df: pd.DataFrame) -> list:
    year_cols = []
    for col in df.columns:
        try:
            year = int(str(col))
            if 1990 <= year <= 2100:
                year_cols.append(col)
        except (ValueError, TypeError):
            continue
    return sorted(year_cols)

def build_tables(cp_csv_path: str, sb_csv_path: str) -> tuple:
    
    cp_raw = pd.read_csv(cp_csv_path)
    
    year_cols = find_year_columns(cp_raw)
    id_cols = [c for c in ['Company Name', 'Company', 'Sector', 'Subsector', 'Unit', 'Geography'] 
               if c in cp_raw.columns]
    
    fact_company = cp_raw.melt(
        id_vars=id_cols, 
        value_vars=year_cols, 
        var_name='year', 
        value_name='intensity'
    )
    
    fact_company['year'] = fact_company['year'].astype('int16')
    fact_company['intensity'] = pd.to_numeric(fact_company['intensity'], errors='coerce').astype('float32')
    
    if 'Company Name' in fact_company.columns:
        fact_company = fact_company.rename(columns={'Company Name': 'Company'})
    
    for col in ['Company', 'Sector', 'Subsector', 'Unit']:
        if col in fact_company.columns:
            fact_company[col] = fact_company[col].astype('category')
    
    fact_company.columns = [c.lower() for c in fact_company.columns]
    
    sb_raw = pd.read_csv(sb_csv_path)
    
    if 'Release date' in sb_raw.columns:
        sb_raw['Release date'] = pd.to_datetime(sb_raw['Release date'], format='%d/%m/%Y')
        group_cols = ['Sector name' if 'Sector name' in sb_raw.columns else 'Sector',
                     'Scenario name' if 'Scenario name' in sb_raw.columns else 'Scenario', 
                     'Region']
        sb_raw = sb_raw.sort_values('Release date', ascending=False).drop_duplicates(
            subset=group_cols, keep='first'
        )
    
    column_mapping = {
        'Sector name': 'Sector',
        'Scenario name': 'Scenario'
    }
    sb_raw = sb_raw.rename(columns=column_mapping)
    
    if 'Scenario' in sb_raw.columns:
        sb_raw['Scenario'] = (
            sb_raw['Scenario'].astype(str)
            .str.replace(r'(\d+\.?\d*)\s+Degrees\s+(\([^)]+\))', r'\1°C \2', case=False, regex=True)
            .str.replace(r'(\d+\.?\d*)\s+Degrees', r'\1°C', case=False, regex=True)
            .str.replace(r'(Below\s+\d+\.?\d*)\s+Degrees', r'\1°C', case=False, regex=True)
        )
    
    # Find year columns
    bench_year_cols = find_year_columns(sb_raw)
    bench_id_cols = [c for c in ['Sector', 'Region', 'Scenario', 'Unit'] if c in sb_raw.columns]
    
    fact_benchmark = sb_raw.melt(
        id_vars=bench_id_cols,
        value_vars=bench_year_cols,
        var_name='year',
        value_name='benchmark'
    )
    
    fact_benchmark['year'] = fact_benchmark['year'].astype('int16')
    fact_benchmark['benchmark'] = pd.to_numeric(fact_benchmark['benchmark'], errors='coerce').astype('float32')
    
    for col in ['Sector', 'Region', 'Scenario', 'Unit']:
        if col in fact_benchmark.columns:
            fact_benchmark[col] = fact_benchmark[col].astype('category')
    
    fact_benchmark.columns = [c.lower() for c in fact_benchmark.columns]
    
    # Set indexes for performance
    fact_company = fact_company.set_index(['sector', 'company', 'year']).sort_index()
    fact_benchmark = fact_benchmark.set_index(['sector', 'region', 'scenario', 'year']).sort_index()
    
    os.makedirs('data', exist_ok=True)
    fact_company.to_parquet('data/fact_company.parquet')
    fact_benchmark.to_parquet('data/fact_benchmark.parquet')
    
    return fact_company, fact_benchmark

def load_tables() -> tuple:
    company_path = Path('data/fact_company.parquet')
    benchmark_path = Path('data/fact_benchmark.parquet')
    
    if not (company_path.exists() and benchmark_path.exists()):
        return build_tables(
            'data/Company_Latest_Assessments_5.0.csv',
            'data/Sector_Benchmarks_19092025.csv'
        )
    
    fact_company = pd.read_parquet('data/fact_company.parquet')
    fact_benchmark = pd.read_parquet('data/fact_benchmark.parquet')
    
    return fact_company, fact_benchmark

def join_company_benchmark(fact_company, fact_benchmark, sector, company, region, scenarios, year_range, subsector=None, exact_region=False):
    
    y_start, y_end = year_range
    
    company_filter = (
        (fact_company.index.get_level_values('sector') == sector) &
        (fact_company.index.get_level_values('company') == company) &
        (fact_company.index.get_level_values('year') >= y_start) &
        (fact_company.index.get_level_values('year') <= y_end)
    )
    
    company_series = fact_company.loc[company_filter].reset_index()
    
    if subsector and 'subsector' in company_series.columns:
        company_series = company_series[company_series['subsector'] == subsector]
    
    scen_lines = {}
    
    for scenario in scenarios:
        bench_filter = (
            (fact_benchmark.index.get_level_values('sector') == sector) &
            (fact_benchmark.index.get_level_values('scenario') == scenario) &
            (fact_benchmark.index.get_level_values('year') >= y_start) &
            (fact_benchmark.index.get_level_values('year') <= y_end)
        )
        
        if region and not exact_region:
            region_filter = bench_filter & (fact_benchmark.index.get_level_values('region') == region)
            scenario_data = fact_benchmark.loc[region_filter].reset_index()
            
            if scenario_data.empty:
                global_filter = bench_filter & (fact_benchmark.index.get_level_values('region') == 'Global')
                scenario_data = fact_benchmark.loc[global_filter].reset_index()
        else:
            scenario_data = fact_benchmark.loc[bench_filter].reset_index()
        
        if not scenario_data.empty:
            scenario_agg = (
                scenario_data
                .groupby('year', as_index=False)['benchmark']
                .mean()
            )
            scen_lines[scenario] = scenario_agg
        else:
            scen_lines[scenario] = pd.DataFrame(columns=['year', 'benchmark'])
    
    return company_series, scen_lines

def load_company_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    years = find_year_columns(raw)
    id_cols = [c for c in ['Company','Sector','Geography','Region','Subsector','Unit'] if c in raw.columns]
    long = raw.melt(id_vars=id_cols, value_vars=years, var_name='Year', value_name='Intensity')
    long['Year'] = long['Year'].astype(int)
    return long

def load_benchmark_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if 'Release date' in raw.columns:
        raw['Release date'] = pd.to_datetime(raw['Release date'], format='%d/%m/%Y')
        group_cols = ['Sector name' if 'Sector name' in raw.columns else 'Sector',
                     'Scenario name' if 'Scenario name' in raw.columns else 'Scenario', 
                     'Region']
        raw = raw.sort_values('Release date', ascending=False).drop_duplicates(subset=group_cols, keep='first')
    
    ren = {}
    if 'Sector name' in raw.columns: ren['Sector name'] = 'Sector'
    if 'Scenario name' in raw.columns: ren['Scenario name'] = 'Scenario'
    raw = raw.rename(columns=ren)
    
    if 'Scenario' in raw.columns:
        raw['Scenario'] = (raw['Scenario'].astype(str)
            .str.replace(r'(\d+\.?\d*)\s+Degrees\s+(\([^)]+\))', r'\1°C \2', case=False, regex=True)
            .str.replace(r'(\d+\.?\d*)\s+Degrees', r'\1°C', case=False, regex=True)
            .str.replace(r'(Below\s+\d+\.?\d*)\s+Degrees', r'\1°C', case=False, regex=True)
        )
    
    years = find_year_columns(raw)
    id_cols = [c for c in ['Sector','Region','Scenario','Unit'] if c in raw.columns]
    long = raw.melt(id_vars=id_cols, value_vars=years, var_name='Year', value_name='Benchmark')
    long['Year'] = long['Year'].astype(int)
    return long
