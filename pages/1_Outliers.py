import os
import streamlit as st
import pandas as pd
from src.io_utils import load_company_csv, load_benchmark_csv
from src.outliers import sector_outliers

st.set_page_config(
    page_title='Outlier Analysis - TPI Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

DATA_COMPANY = 'data/Company_Latest_Assessments_5.0.csv'
DATA_BENCH   = 'data/Sector_Benchmarks_19092025.csv'

@st.cache_data(show_spinner=True)
def load_data(cp_path: str, sb_path: str):
    return load_company_csv(cp_path), load_benchmark_csv(sb_path)

# Load data
cp, sb = load_data(DATA_COMPANY, DATA_BENCH)

st.title('Company Outlier Analysis')

sectors_with_data = sorted(list(set(cp['Sector'].dropna().unique()) & set(sb['Sector'].dropna().unique())))
if not sectors_with_data:
    st.error('No overlapping sectors between company and benchmark data.')
    st.stop()

c1, c2, c3, c4 = st.columns([2,2,1,1])
sector = c1.selectbox('Sector', sectors_with_data)
scenario_opts = sorted(sb[sb['Sector']==sector]['Scenario'].dropna().unique()) if 'Scenario' in sb.columns else []
scenario = c2.selectbox('Scenario', scenario_opts, index=0 if scenario_opts else None)
region_pref = None
regions = sorted(sb[sb['Sector']==sector]['Region'].dropna().unique())
if regions:
    region_pref = c3.selectbox('Region preference', regions, index=0)
k = int(c4.number_input('Top K', min_value=3, max_value=30, value=10, step=1))

c5, c6 = st.columns(2)
start = int(c5.number_input('Start year', value=2020))
end   = int(c6.number_input('End year',   value=2035))

st.subheader('Results')

good, bad = sector_outliers(cp, sb, sector=sector, scenario=scenario, region_pref=region_pref, k=k, start=start, end=end)
left, right = st.columns(2)
with left:
    st.subheader('Best Aligned Companies (Lower CBD)')
    st.success('Companies performing better than sector benchmark')
    good_display = good.reset_index(drop=True)
    good_display.index = good_display.index + 1
    if 'CBD' in good_display.columns:
        good_display['CBD'] = good_display['CBD'].round(3)
    st.dataframe(good_display, width='stretch')
with right:
    st.subheader('Worst Aligned Companies (Higher CBD)')
    st.warning('Companies lagging behind sector benchmark')  
    bad_display = bad.reset_index(drop=True)
    bad_display.index = bad_display.index + 1
    if 'CBD' in bad_display.columns:
        bad_display['CBD'] = bad_display['CBD'].round(3)
    st.dataframe(bad_display, width='stretch')

st.caption("""
**CBD** = Cumulative Benchmark Deviation (area under company-benchmark curve over selected years)  
**z** = Z-score (standard deviations from sector average). Values beyond ±2 indicate statistical outliers.  
**Negative CBD = better alignment** (company performing better than benchmark).
""")

# Download section for outliers
st.subheader('Downloads')
col1, col2, col3 = st.columns(3)

with col1:
    if not good.empty:
        good_download = good.copy()
        good_download['Analysis_Type'] = 'Best_Aligned'
        good_download['Sector'] = sector
        good_download['Scenario'] = scenario
        if region_pref:
            good_download['Region'] = region_pref
        good_download['Analysis_Period'] = f"{start}-{end}"
        csv_data = good_download.to_csv(index=False)
        st.download_button(
            label="Download Best Aligned CSV",
            data=csv_data,
            file_name=f"best_aligned_{sector.replace(' ', '_')}_{start}_{end}.csv",
            mime="text/csv"
        )

with col2:
    if not bad.empty:
        bad_download = bad.copy()
        bad_download['Analysis_Type'] = 'Worst_Aligned'
        bad_download['Sector'] = sector
        bad_download['Scenario'] = scenario
        if region_pref:
            bad_download['Region'] = region_pref
        bad_download['Analysis_Period'] = f"{start}-{end}"
        csv_data = bad_download.to_csv(index=False)
        st.download_button(
            label="Download Worst Aligned CSV",
            data=csv_data,
            file_name=f"worst_aligned_{sector.replace(' ', '_')}_{start}_{end}.csv",
            mime="text/csv"
        )

with col3:
    if not good.empty and not bad.empty:
        combined_outliers = pd.concat([
            good.assign(Analysis_Type='Best_Aligned'),
            bad.assign(Analysis_Type='Worst_Aligned')
        ], ignore_index=True)
        combined_outliers['Sector'] = sector
        combined_outliers['Scenario'] = scenario
        if region_pref:
            combined_outliers['Region'] = region_pref
        combined_outliers['Analysis_Period'] = f"{start}-{end}"
        csv_data = combined_outliers.to_csv(index=False)
        st.download_button(
            label="Download All Outliers CSV",
            data=csv_data,
            file_name=f"all_outliers_{sector.replace(' ', '_')}_{start}_{end}.csv",
            mime="text/csv"
        )
