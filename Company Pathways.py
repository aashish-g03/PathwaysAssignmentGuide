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

# Download section
st.subheader('Downloads')
col1, col2, col3 = st.columns(3)

with col1:
    # Company data download
    company_download = f_cp[['Year', 'Intensity']].copy()
    company_download['Company'] = display_name
    company_download['Sector'] = sector
    if subsector:
        company_download['Subsector'] = subsector
    csv_data = company_download.to_csv(index=False)
    st.download_button(
        label="Download Company Data CSV",
        data=csv_data,
        file_name=f"{company.replace(' ', '_')}_pathways_data.csv",
        mime="text/csv"
    )

with col2:
    # Benchmark data download
    if scenario_map:
        benchmark_download = pd.DataFrame()
        for scenario_name, scenario_data in scenario_map.items():
            if not scenario_data.empty:
                scenario_df = scenario_data.copy()
                scenario_df['Scenario'] = scenario_name
                scenario_df['Sector'] = sector
                if region:
                    scenario_df['Region'] = region
                benchmark_download = pd.concat([benchmark_download, scenario_df], ignore_index=True)
        
        if not benchmark_download.empty:
            csv_data = benchmark_download.to_csv(index=False)
            st.download_button(
                label="Download Benchmark Data CSV",
                data=csv_data,
                file_name=f"{sector.replace(' ', '_')}_benchmark_data.csv",
                mime="text/csv"
            )

with col3:
    # Combined data download
    if scenario_map and not f_cp.empty:
        combined_data = f_cp[['Year', 'Intensity']].copy()
        combined_data['Company'] = display_name
        combined_data['Sector'] = sector
        combined_data['Scenario'] = 'Actual'
        combined_data['Data_Type'] = 'Company'
        if subsector:
            combined_data['Subsector'] = subsector
        if region:
            combined_data['Region'] = region
        
        benchmark_combined = pd.DataFrame()
        for scenario_name, scenario_data in scenario_map.items():
            if not scenario_data.empty:
                scenario_df = scenario_data.copy()
                scenario_df = scenario_df.rename(columns={'Benchmark': 'Intensity'})
                scenario_df['Company'] = display_name
                scenario_df['Sector'] = sector
                scenario_df['Scenario'] = scenario_name
                scenario_df['Data_Type'] = 'Benchmark'
                if subsector:
                    scenario_df['Subsector'] = subsector
                if region:
                    scenario_df['Region'] = region
                benchmark_combined = pd.concat([benchmark_combined, scenario_df], ignore_index=True)
        
        if not benchmark_combined.empty:
            all_data = pd.concat([combined_data, benchmark_combined], ignore_index=True)
            csv_data = all_data.to_csv(index=False)
            st.download_button(
                label="Download All Data CSV",
                data=csv_data,
                file_name=f"{company.replace(' ', '_')}_complete_analysis.csv",
                mime="text/csv"
            )

# Methodology Notes
with st.expander("Methodology Notes"):
    st.markdown("""
    **Data Sources & Processing:**
    - **Company Data**: TPI Company Latest Assessments v5.0 - carbon intensity trajectories by sector
    - **Benchmark Data**: TPI Sector Benchmarks (Sept 2025) - climate scenario pathways aligned with temperature goals
    
    **Scenario Normalization:**
    - Scenario labels standardized: "1.5 Degrees" → "1.5°C", "International Pledges" → "National Pledges"
    - All temperature scenarios expressed in consistent °C notation for clarity
    
    **Unit Handling:**
    - Company and benchmark data automatically matched by sector and unit type
    - Y-axis displays detected unit from company data (e.g., tCO2e/MWh, kgCO2e/tonne)
    
    **Region Fallback Logic:**
    - Benchmark data matched by exact region selection when available
    - Automatically falls back to "Global" benchmarks if selected region unavailable for specific scenarios
    
    **Carbon Budget Deviation (CBD):**
    - Visual zones represent alignment ranges: Green (1.5°C line), Amber (between 1.5°C-2°C), Red (above 2°C toward pledges)
    - Company trajectory compared against scenario bands to assess climate alignment
    
    **Year Range Defaults:**
    - Default analysis window: 2015-2035 (adjustable via slider)
    - Focuses on near-term transition period most relevant for corporate climate action
    """)

st.divider()
