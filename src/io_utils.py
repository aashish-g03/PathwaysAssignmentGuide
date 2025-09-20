import re
import pandas as pd
from typing import List

YEAR_REGEX = re.compile(r'^(19|20)\d{2}$')

def find_year_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if YEAR_REGEX.fullmatch(str(c))]

def load_company_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    # Rename Company Name to Company
    if 'Company Name' in raw.columns:
        raw = raw.rename(columns={'Company Name': 'Company'})
        raw['Company'] = raw['Company'].str.strip()
    # Rename CP Unit to Unit
    if 'CP Unit' in raw.columns:
        raw = raw.rename(columns={'CP Unit': 'Unit'})
    # Find years
    years = find_year_columns(raw)
    id_cols = [c for c in ['Company','Sector','Geography','Region','Subsector','Unit'] if c in raw.columns]
    long = raw.melt(id_vars=id_cols, value_vars=years, var_name='Year', value_name='Intensity')
    long['Year'] = long['Year'].astype(int)

    if 'Geography' in long.columns and 'Region' not in long.columns:
        pass
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
            .str.replace('International Pledges', 'National Pledges', case=False)
        )
    
    years = find_year_columns(raw)
    id_cols = [c for c in ['Sector','Region','Scenario','Unit'] if c in raw.columns]
    long = raw.melt(id_vars=id_cols, value_vars=years, var_name='Year', value_name='Benchmark')
    long['Year'] = long['Year'].astype(int)
    return long

def join_company_benchmark(company_long: pd.DataFrame, bench_long: pd.DataFrame) -> pd.DataFrame:
    base = company_long.merge(bench_long[['Sector','Region','Scenario','Unit','Year','Benchmark']],
                              on=['Sector','Year'], how='left', suffixes=('','_bench'))
    return base
