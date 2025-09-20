import os
import streamlit as st
import pandas as pd

from src.io_utils import load_company_csv, load_benchmark_csv
from src.plot_utils import pathway_figure

st.set_page_config(
    page_title='Company Pathways - TPI Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

DATA_COMPANY = 'data/Company_Latest_Assessments_5.0.csv'
DATA_BENCH   = 'data/Sector_Benchmarks_19092025.csv'

@st.cache_data(show_spinner=True)
def load_data(cp_path: str, sb_path: str):
    cp_long = load_company_csv(cp_path)
    sb_long = load_benchmark_csv(sb_path)
    return cp_long, sb_long

st.title('Company Pathways')

# Load data
cp_long, sb_long = load_data(DATA_COMPANY, DATA_BENCH)

# Controls
sectors_with_data = sorted(list(set(cp_long['Sector'].dropna().unique()) & set(sb_long['Sector'].dropna().unique())))
if not sectors_with_data:
    st.error('No overlapping sectors between company and benchmark data.')
    st.stop()

col1, col2 = st.columns(2)
with col1:
    sector = st.selectbox('Sector', sectors_with_data)
    companies = sorted(cp_long[cp_long['Sector']==sector]['Company'].dropna().unique())
    company = st.selectbox('Company', companies)
    
    # Handle subsector selection for companies with multiple subsectors
    company_data = cp_long[(cp_long['Sector']==sector) & (cp_long['Company']==company)]
    subsectors = sorted(company_data['Subsector'].dropna().unique()) if 'Subsector' in company_data.columns else []
    
    subsector = None
    if len(subsectors) > 1:
        st.info(f"{company} operates in {len(subsectors)} subsectors")
        subsector = st.selectbox('Subsector', subsectors)
        st.caption(f"Showing data for: **{subsector}**")
    elif len(subsectors) == 1:
        subsector = subsectors[0]
        st.caption(f"Subsector: **{subsector}**")
    else:
        # Most companies don't have subsector breakdowns
        st.caption("Analyzing consolidated company-level data")

with col2:
    region = None
    if 'Region' in sb_long.columns:
        options = sorted(sb_long[sb_long['Sector']==sector]['Region'].dropna().unique())
        if options:
            region = st.selectbox('Benchmark Region', options, index=0)
    scenario_opts = sorted(sb_long[sb_long['Sector']==sector]['Scenario'].dropna().unique()) if 'Scenario' in sb_long.columns else []
    default_scenarios = [s for s in ['1.5°C','Below 2°C','National Pledges'] if s in scenario_opts]
    scenarios = st.multiselect('Scenarios', scenario_opts, default=default_scenarios or scenario_opts[:2])

# Filter data by sector and company
f_cp = cp_long[(cp_long['Sector']==sector) & (cp_long['Company']==company)]

# Apply subsector filtering if applicable
if subsector and 'Subsector' in f_cp.columns:
    f_cp = f_cp[f_cp['Subsector']==subsector]
    if len(f_cp) == 0:
        st.warning(f"No data found for {company} in {subsector} subsector.")
        st.stop()

if 'Year' in f_cp.columns:
    # year slider bounds:
    y_min, y_max = int(f_cp['Year'].min()), int(f_cp['Year'].max())
    year_range = st.slider('Year range', y_min, y_max, (max(y_min,2015), min(y_max,2035)))
    f_cp = f_cp[(f_cp['Year']>=year_range[0]) & (f_cp['Year']<=year_range[1])]

# Build scenario series
scenario_map = {}
for s in scenarios:
    q = (sb_long['Sector']==sector) & (sb_long['Scenario']==s)
    if region and 'Region' in sb_long.columns:
        # prefer exact region; fallback to Global
        sset = sb_long[q & (sb_long['Region']==region)]
        if sset.empty:
            sset = sb_long[q & (sb_long['Region']=='Global')]
    else:
        sset = sb_long[q]
    sset_clean = sset[['Year','Benchmark']].dropna().groupby('Year')['Benchmark'].mean().reset_index()
    scenario_map[s] = sset_clean

unit_hint = ''
if 'Unit' in f_cp.columns:
    u = f_cp['Unit'].dropna().astype(str)
    if not u.empty:
        unit_hint = u.iloc[0]

# Create display name with subsector info
display_name = f"{company} ({subsector})" if subsector else company
fig = pathway_figure(f_cp[['Year','Intensity']], scenario_map, unit_hint, display_name)
st.plotly_chart(fig, use_container_width=True)
