import os
import streamlit as st
import pandas as pd

from src.io_utils import load_tables
from src.analytics import (view_pathway, view_sector_companies, view_company_year_bounds,
                         view_company_subsectors, view_sector_regions_scenarios,
                         view_sector_companies_by_country, view_country_options_direct)
from src.plot_utils import pathway_figure

st.set_page_config(
    page_title='Company Climate Performance - TPI Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

DATA_COMPANY = 'data/Company_Latest_Assessments_5.0.csv'
DATA_BENCH   = 'data/Sector_Benchmarks_19092025.csv'

@st.cache_data(show_spinner=True)
def load_fact_tables():
    return load_tables()

st.title('Company Climate Performance')
st.caption("Track how companies are progressing toward their climate goals")

# Load data from CSV files in form of fact tables
fact_company, fact_benchmark = load_fact_tables()

company_sectors = fact_company.index.get_level_values('sector').unique()
benchmark_sectors = fact_benchmark.index.get_level_values('sector').unique()

sectors_with_data = sorted(list(company_sectors))
sectors_with_benchmarks = sorted(list(set(company_sectors) & set(benchmark_sectors)))


col1, col2 = st.columns(2)
with col1:
    sector = st.selectbox('Sector', sectors_with_data)
    country = st.selectbox("Country", view_country_options_direct(fact_company, sector), index=0)
    
    companies = view_sector_companies_by_country(fact_company, sector, country)
    if not companies:
        st.error(f"No companies found for the selected filters.")
        st.stop()
        
    company = st.selectbox('Company', companies)
    subsectors = view_company_subsectors(fact_company, sector, company)
    
    subsector = None
    if len(subsectors) > 1:
        st.info(f"{company} operates in {len(subsectors)} subsectors")
        subsector = st.selectbox('Subsector', subsectors)
        st.caption(f"Showing data for: **{subsector}**")
    elif len(subsectors) == 1:
        subsector = subsectors[0]
        st.caption(f"Subsector: **{subsector}**")
    else:
        st.caption("Analyzing consolidated company-level data")

with col2:
    has_benchmark_data = sector in sectors_with_benchmarks
    
    if has_benchmark_data:
        region_options, scenario_opts = view_sector_regions_scenarios(fact_benchmark, sector)
        
        region = None
        if region_options:
            region = st.selectbox('Benchmark Region', region_options, index=0)
        
        default_scenarios = [s for s in ['1.5°C','Below 2°C','National Pledges'] if s in scenario_opts]
        scenarios = st.multiselect('Scenarios', scenario_opts, default=default_scenarios or scenario_opts[:2])
    else:
        st.warning(f"**{sector}** sector has company data but no benchmark scenarios available.")
        st.info("You can still view company data, but pathway comparisons won't be possible.")
        region = None
        scenarios = []

y_min, y_max = view_company_year_bounds(fact_company, sector, company, subsector)

if y_min is None or y_max is None:
    if subsector:
        st.warning(f"No data found for {company} in {subsector} subsector.")
    else:
        st.warning(f"No data found for {company} in {sector} sector.")
    st.stop()

year_range = st.slider('Year range', y_min, y_max, (max(y_min,2015), min(y_max,2035)))

company_df, scenario_map, bands = view_pathway(
    fact_company, fact_benchmark, sector, company, 
    region, scenarios, year_range, subsector, exact_region=False
)

unit_hint = ''
if 'unit' in company_df.columns:
    units = company_df['unit'].dropna().astype(str)
    if not units.empty:
        unit_hint = units.iloc[0]

display_name = f"{company} ({subsector})" if subsector else company
badge_scenarios = ', '.join(scenarios) if scenarios else 'None'
badge_region = region if region else 'Global'

y_start, y_end = year_range
st.caption(f"**Sector:** {sector} • **Country:** {country} • "
          f"**Benchmark:** {badge_region} • **Scenarios:** {badge_scenarios} • **Years:** {y_start}–{y_end}")

visualization_data = company_df[['year', 'intensity']].rename(columns={'year': 'Year', 'intensity': 'Emissions per unit'})
scenario_map_viz = {k: v.rename(columns={'year': 'Year', 'benchmark': 'Benchmark'}) for k, v in scenario_map.items()}

fig = pathway_figure(visualization_data, scenario_map_viz, unit_hint, display_name)
st.plotly_chart(fig, use_container_width=True)

st.subheader('Downloads')
col1, col2, col3 = st.columns(3)

with col1:
    if not company_df.empty:
        company_download = company_df[['year', 'intensity']].rename(columns={'year': 'Year', 'intensity': 'Emissions per unit'})
        company_download = company_download.assign(
            Company=display_name,
            Sector=sector
        )
        if subsector:
            company_download = company_download.assign(Subsector=subsector)
    csv_data = company_download.to_csv(index=False)
    st.download_button(
        label="Download Company Data CSV",
        data=csv_data,
        file_name=f"{company.replace(' ', '_')}_pathways_data.csv",
        mime="text/csv"
    )

with col2:
    if scenario_map:
        benchmark_frames = []
        for scenario_name, scenario_data in scenario_map.items():
            if not scenario_data.empty:
                enhanced_scenario = (
                    scenario_data
                    .rename(columns={'year': 'Year', 'benchmark': 'Benchmark'})
                    .assign(Scenario=scenario_name, Sector=sector)
                )
                if region:
                    enhanced_scenario = enhanced_scenario.assign(Region=region)
                benchmark_frames.append(enhanced_scenario)
        
        if benchmark_frames:
            benchmark_download = pd.concat(benchmark_frames, ignore_index=True)
        else:
            benchmark_download = pd.DataFrame()
        
        if not benchmark_download.empty:
            csv_data = benchmark_download.to_csv(index=False)
            st.download_button(
                label="Download Benchmark Data CSV",
                data=csv_data,
                file_name=f"{sector.replace(' ', '_')}_benchmark_data.csv",
                mime="text/csv"
            )

with col3:
    if scenario_map and not company_df.empty:
        company_standardized = (
            company_df[['year', 'intensity']]
            .rename(columns={'year': 'Year', 'intensity': 'Emissions per unit'})
            .assign(
                Company=display_name,
                Sector=sector,
                Scenario='Actual',
                Data_Type='Company'
            )
        )
        if subsector:
            company_standardized = company_standardized.assign(Subsector=subsector)
        if region:
            company_standardized = company_standardized.assign(Region=region)
        
        benchmark_frames = []
        for scenario_name, scenario_data in scenario_map.items():
            if not scenario_data.empty:
                scenario_standardized = (
                    scenario_data
                    .rename(columns={'year': 'Year', 'benchmark': 'Emissions per unit'})
                    .assign(
                        Company=display_name,
                        Sector=sector,
                        Scenario=scenario_name,
                        Data_Type='Benchmark'
                    )
                )
                if subsector:
                    scenario_standardized = scenario_standardized.assign(Subsector=subsector)
                if region:
                    scenario_standardized = scenario_standardized.assign(Region=region)
                benchmark_frames.append(scenario_standardized)
        
        if benchmark_frames:
            benchmark_unified = pd.concat(benchmark_frames, ignore_index=True)
            all_data = pd.concat([company_standardized, benchmark_unified], ignore_index=True)
        else:
            all_data = company_standardized
        
        csv_data = all_data.to_csv(index=False)
        st.download_button(
            label="Download All Data CSV",
            data=csv_data,
            file_name=f"{company.replace(' ', '_')}_complete_analysis.csv",
            mime="text/csv"
        )

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
