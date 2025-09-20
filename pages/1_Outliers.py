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
    st.dataframe(good_display, width='stretch')
with right:
    st.subheader('Worst Aligned Companies (Higher CBD)')
    st.warning('Companies lagging behind sector benchmark')  
    bad_display = bad.reset_index(drop=True)
    bad_display.index = bad_display.index + 1
    st.dataframe(bad_display, width='stretch')

st.caption("""
**CBD** = Cumulative Benchmark Deviation (area under company-benchmark curve over selected years)  
**z** = Z-score (standard deviations from sector average). Values beyond ±2 indicate statistical outliers.
""")
