import os
import streamlit as st
import pandas as pd
from src.io_utils import load_tables
from src.analytics import view_outliers

st.set_page_config(
    page_title='Outlier Analysis - TPI Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

DATA_COMPANY = 'data/Company_Latest_Assessments_5.0.csv'
DATA_BENCH   = 'data/Sector_Benchmarks_19092025.csv'

@st.cache_data(show_spinner=True)
def load_fact_tables():
    return load_tables()

fact_company, fact_benchmark = load_fact_tables()

st.title('Company Outlier Analysis')

company_sectors = fact_company.index.get_level_values('sector').unique()
benchmark_sectors = fact_benchmark.index.get_level_values('sector').unique()
sectors_with_benchmarks = sorted(list(set(company_sectors) & set(benchmark_sectors)))

if not sectors_with_benchmarks:
    st.error('No sectors have both company and benchmark data for outlier analysis.')
    st.stop()

st.info(f"Showing {len(sectors_with_benchmarks)} sectors with benchmark data out of {len(company_sectors)} total sectors.")

c1, c2, c3, c4 = st.columns([2,2,1,1])
sector = c1.selectbox('Sector', sectors_with_benchmarks)

sector_bench_filter = fact_benchmark.index.get_level_values('sector') == sector
sector_scenarios = fact_benchmark.loc[sector_bench_filter].index.get_level_values('scenario').unique()
scenario_opts = sorted(sector_scenarios)

scenario = c2.selectbox('Scenario', scenario_opts, index=0 if scenario_opts else None)

region_pref = None
sector_regions = fact_benchmark.loc[sector_bench_filter].index.get_level_values('region').unique()
regions = sorted(sector_regions)
if regions:
    region_pref = c3.selectbox('Region preference', regions, index=0)

k = int(c4.number_input('Top K', min_value=3, max_value=30, value=10, step=1))

c5, c6 = st.columns(2)
start = int(c5.number_input('Start year', value=2020))
end   = int(c6.number_input('End year',   value=2035))

st.subheader('Results')

good_performers, poor_performers = view_outliers(
    fact_company, fact_benchmark, 
    sector=sector, scenario=scenario, 
    region=region_pref, year_range=(start, end), 
    exact_region=False, k=k
)
left, right = st.columns(2)
with left:
    st.subheader('Best Aligned Companies (Lower CBD)')
    st.success('Companies performing better than sector benchmark')

    if not good_performers.empty:
        good_display = (
            good_performers
            .reset_index(drop=True)
            .assign(cbd=lambda df: df['cbd'].round(3) if 'cbd' in df.columns else df.get('cbd', pd.Series()))
        )
        good_display.index = good_display.index + 1
        st.dataframe(good_display, use_container_width=True)
    else:
        st.info('No data available for selected criteria')

with right:
    st.subheader('Worst Aligned Companies (Higher CBD)')
    st.warning('Companies lagging behind sector benchmark')

    if not poor_performers.empty:
        poor_display = (
            poor_performers
            .reset_index(drop=True)
            .assign(cbd=lambda df: df['cbd'].round(3) if 'cbd' in df.columns else df.get('cbd', pd.Series()))
        )
        poor_display.index = poor_display.index + 1
        st.dataframe(poor_display, use_container_width=True)
    else:
        st.info('No data available for selected criteria')

st.caption("""
**CBD** = Cumulative Benchmark Deviation (area under company-benchmark curve over selected years)  
**Z** = Z-score (standard deviations from sector average). Values beyond ±2 indicate statistical outliers.  
**Negative CBD = better alignment** (company performing better than benchmark).
""")

st.subheader('Downloads')
col1, col2, col3 = st.columns(3)

with col1:
    if not good_performers.empty:
        good_download = (
            good_performers
            .assign(
                Analysis_Type='Best_Aligned',
                Sector=sector,
                Scenario=scenario,
                Analysis_Period=f"{start}-{end}"
            )
        )
        if region_pref:
            good_download = good_download.assign(Region=region_pref)
        csv_data = good_download.to_csv(index=False)
        st.download_button(
            label="Download Best Aligned CSV",
            data=csv_data,
            file_name=f"best_aligned_{sector.replace(' ', '_')}_{start}_{end}.csv",
            mime="text/csv"
        )

with col2:
    if not poor_performers.empty:
        poor_download = (
            poor_performers
            .assign(
                Analysis_Type='Worst_Aligned',
                Sector=sector,
                Scenario=scenario,
                Analysis_Period=f"{start}-{end}"
            )
        )
        if region_pref:
            poor_download = poor_download.assign(Region=region_pref)
        
        csv_data = poor_download.to_csv(index=False)
        st.download_button(
            label="Download Worst Aligned CSV",
            data=csv_data,
            file_name=f"worst_aligned_{sector.replace(' ', '_')}_{start}_{end}.csv",
            mime="text/csv"
        )

with col3:
    if not good_performers.empty and not poor_performers.empty:
        outlier_frames = [
            good_performers.assign(Analysis_Type='Best_Aligned'),
            poor_performers.assign(Analysis_Type='Worst_Aligned')
        ]
        combined_outliers = (
            pd.concat(outlier_frames, ignore_index=True)
            .assign(
                Sector=sector,
                Scenario=scenario,
                Analysis_Period=f"{start}-{end}"
            )
        )
        if region_pref:
            combined_outliers = combined_outliers.assign(Region=region_pref)
        csv_data = combined_outliers.to_csv(index=False)
        st.download_button(
            label="Download All Outliers CSV",
            data=csv_data,
            file_name=f"all_outliers_{sector.replace(' ', '_')}_{start}_{end}.csv",
            mime="text/csv"
        )
