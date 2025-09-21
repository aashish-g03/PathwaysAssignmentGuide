import os
import streamlit as st
import pandas as pd
from src.io_utils import load_tables
from src.analytics import view_country_options_direct, detect_financial_outliers, rank_outlier_significance
from src.outliers import sector_outliers

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
st.caption("Find companies that stand out from their industry peers - both the best and worst performers")

company_sectors = fact_company.index.get_level_values('sector').unique()
benchmark_sectors = fact_benchmark.index.get_level_values('sector').unique()
sectors_with_benchmarks = sorted(list(set(company_sectors) & set(benchmark_sectors)))

if not sectors_with_benchmarks:
    st.warning('Sorry, we need more data to find outliers right now. Please check back later.')
    st.stop()

st.info(f"Showing {len(sectors_with_benchmarks)} sectors with benchmark data out of {len(company_sectors)} total sectors.")

c1, c2, c3 = st.columns(3)
sector = c1.selectbox('Industry', sectors_with_benchmarks)

sector_bench_filter = fact_benchmark.index.get_level_values('sector') == sector
sector_scenarios = fact_benchmark.loc[sector_bench_filter].index.get_level_values('scenario').unique()
scenario_opts = sorted(sector_scenarios)

scenario = c2.selectbox('Climate Target', scenario_opts, index=0 if scenario_opts else None, help="Different temperature goals - 1.5°C is most ambitious")
country = c3.selectbox("Country", view_country_options_direct(fact_company, sector), index=0)

region_pref = "Global"

c4, c5, c6 = st.columns(3)
k = int(c4.number_input('Show top', min_value=3, max_value=30, value=10, step=1))

start = int(c5.number_input('From year', value=2020))
end   = int(c6.number_input('To year', value=2035))

st.subheader('Results')

good_performers, poor_performers = sector_outliers(
    cp_long=fact_company, sb_long=fact_benchmark, 
    sector=sector, scenario=scenario,
    company_region=None,
    country=None if country=="Global" else country,
    start=start, end=end, k=k
)

st.caption(f"Comparing {sector} companies in {country} against {scenario} climate target from {start} to {end}")
left, right = st.columns(2)
with left:
    st.subheader('Best Performing Companies')
    st.success('Companies doing well compared to their industry peers')

    if not good_performers.empty:
        good_display = (
            good_performers
            .reset_index(drop=True)
            .assign(cbd=lambda df: df['cbd'].round(3))
            .assign(z_score=lambda df: df['z'].round(4))
            .drop(columns=['z'])
        )
        good_display = good_display.rename(columns={
            'company': 'Company',
            'cbd': 'Climate Gap Score',
            'z_score': 'Performance Score'
        })
        good_display.index = good_display.index + 1
        st.dataframe(good_display, use_container_width=True)
    else:
        st.info('No data available for selected criteria')

with right:
    st.subheader('Worst Performing Companies')
    st.warning('Companies falling behind their industry peers')

    if not poor_performers.empty:
        poor_display = (
            poor_performers
            .reset_index(drop=True)
            .assign(cbd=lambda df: df['cbd'].round(3))
            .assign(z_score=lambda df: df['z'].round(4))
            .drop(columns=['z'])
        )
        poor_display = poor_display.rename(columns={
            'company': 'Company',
            'cbd': 'Climate Gap Score',
            'z_score': 'Performance Score'
        })
        poor_display.index = poor_display.index + 1
        st.dataframe(poor_display, use_container_width=True)
    else:
        st.info('No data available for selected criteria')

st.caption("""
**Climate Gap Score**: How far a company is from its climate target. Lower numbers are better. We are using Cumulative Budget Deviation (CBD) to measure this.  
**Performance Score**: How a company compares to industry average. Higher absolute values indicate outliers. We are using Z-score to measure this.
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
            label="Download Best Performers",
            data=csv_data,
            file_name=f"best_performers_{sector.replace(' ', '_')}_{start}_{end}.csv",
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
            label="Download Worst Performers",
            data=csv_data,
            file_name=f"worst_performers_{sector.replace(' ', '_')}_{start}_{end}.csv",
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
            label="Download All Outliers",
            data=csv_data,
            file_name=f"all_outliers_{sector.replace(' ', '_')}_{start}_{end}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.subheader("Investment Analysis")
st.caption("Find companies with the highest investment potential and risk")

use_advanced = st.checkbox("Show investment analysis", value=False, help="Uses additional financial metrics to identify opportunities and risks")

if use_advanced:
    try:
        with st.spinner("Analyzing investment opportunities..."):
            outlier_analysis = detect_financial_outliers(
                fact_company, fact_benchmark, sector, 
                scenarios=[scenario], confidence=0.95
            )
            ranked_outliers = rank_outlier_significance(
                outlier_analysis, f"{sector} investment analysis"
            )
        
        st.write("**Investment Opportunities**")
        if not ranked_outliers.high_opportunity.empty:
            opp_df = ranked_outliers.high_opportunity[['company', 'significance', 'details']].head(5)
            opp_display = opp_df.rename(columns={
                'company': 'Company',
                'significance': 'Significance',
                'details': 'Details'
            }).copy()
            opp_display['Significance'] = opp_display['Significance'].round(4)
            opp_display.index = range(1, len(opp_display) + 1)
            st.dataframe(opp_display, use_container_width=True)
        else:
            st.info("No clear investment opportunities found")
        
        st.write("**High Risk Companies**")
        if not ranked_outliers.high_risk.empty:
            risk_df = ranked_outliers.high_risk[['company', 'significance', 'details']].head(5)
            risk_display = risk_df.rename(columns={
                'company': 'Company',
                'significance': 'Significance', 
                'details': 'Details'
            }).copy()
            risk_display['Significance'] = risk_display['Significance'].round(4)
            risk_display.index = range(1, len(risk_display) + 1)
            st.dataframe(risk_display, use_container_width=True)
        else:
            st.info("No high risk companies identified")
        
        if not ranked_outliers.volatile.empty:
            st.write("**Unpredictable Companies**")
            vol_df = ranked_outliers.volatile[['company', 'details']].head(3)
            vol_display = vol_df.rename(columns={
                'company': 'Company',
                'details': 'Details'
            }).copy()
            vol_display.index = range(1, len(vol_display) + 1)
            st.dataframe(vol_display, use_container_width=True)
        
        st.write(f"**Summary**: Analyzed {outlier_analysis.total_companies_analyzed} companies to find investment opportunities and risks")
        
    except Exception as e:
        st.error(f"Could not complete investment analysis: {str(e)}")
        st.info("Try using the basic outlier analysis above")
