import streamlit as st
import pandas as pd
from src.io_utils import load_tables
from src.business_intelligence import create_sector_performance_summary, generate_climate_risk_heatmap, generate_executive_summary

st.set_page_config(
    page_title='Industry Analysis - TPI Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

@st.cache_data(show_spinner=True)
def load_fact_tables():
    return load_tables()

fact_company, fact_benchmark = load_fact_tables()

st.title('Industry Analysis')
st.caption('Analyze climate performance across different industries and find trends')

tab1, tab2, tab3 = st.tabs(["Overview", "Industry Analysis", "Risk Comparison"])

with tab1:
    st.subheader("Market Overview")
    st.caption("Get a quick summary of climate performance across all industries")
    
    if st.button("Generate Overview", type="primary"):
        with st.spinner("Analyzing market data..."):
            try:
                summary = generate_executive_summary(fact_company, fact_benchmark)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Industries Analyzed", summary['total_sectors_analyzed'])
                with col2:
                    st.metric("Data as of", summary['analysis_date'][:10])
                
                if summary['key_findings']:
                    st.subheader("Industry Summary")
                    findings_df = pd.DataFrame(summary['key_findings'])
                    findings_display = findings_df.rename(columns={
                        'sector': 'Industry',
                        'companies_analyzed': 'Companies Tracked',
                        'high_risk_percentage': 'Companies at Risk (%)',
                        'data_coverage': 'Data Available (%)'
                    }).copy()
                    findings_display.index = range(1, len(findings_display) + 1)
                    st.dataframe(findings_display, use_container_width=True)
                
                if summary['risk_alerts']:
                    st.subheader("Industries Needing Attention")
                    for alert in summary['risk_alerts']:
                        st.warning(f"**{alert['sector']}**: {alert['message']}")
                
                if summary['opportunities']:
                    st.subheader("Industries Doing Well")
                    for opp in summary['opportunities']:
                        st.success(f"**{opp['sector']}**: {opp['message']}")
                        
            except Exception as e:
                st.error(f"Could not generate overview: {str(e)}")

with tab2:
    st.subheader("Deep Dive into One Industry")
    st.caption("Pick an industry to see detailed performance metrics and trends")
    
    company_sectors = fact_company.index.get_level_values('sector').unique()
    benchmark_sectors = fact_benchmark.index.get_level_values('sector').unique()
    available_sectors = sorted(list(set(company_sectors) & set(benchmark_sectors)))
    
    selected_sector = st.selectbox("Choose Industry", available_sectors, help="Select an industry like Steel, Airlines, or Oil & Gas")
    
    if st.button("Analyze This Industry", type="primary"):
        with st.spinner(f"Analyzing {selected_sector} industry..."):
            try:
                dashboard = create_sector_performance_summary(fact_company, fact_benchmark, selected_sector)
                
                if not dashboard.data_coverage.empty:
                    coverage = dashboard.data_coverage.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Companies", int(coverage['total_companies']))
                    with col2:
                        st.metric("Companies with Data", int(coverage['companies_with_data']))
                    with col3:
                        st.metric("Data Coverage", f"{coverage['coverage_percentage']:.1f}%")
                
                if not dashboard.sector_rankings.empty:
                    st.subheader("Company Rankings")
                    st.caption("Top 10 companies in this industry ranked by climate performance")
                    rankings_display = dashboard.sector_rankings[['company', 'sector_rank', 'latest_intensity', 'latest_year']].head(10)
                    rankings_display = rankings_display.rename(columns={
                        'company': 'Company',
                        'sector_rank': 'Industry Rank',
                        'latest_intensity': 'Latest Emissions per unit',
                        'latest_year': 'Latest Year'
                    }).copy()
                    rankings_display.index = range(1, len(rankings_display) + 1)
                    st.dataframe(rankings_display, use_container_width=True)
                
                if not dashboard.trend_analysis.empty:
                    st.subheader("How Companies Are Trending")
                    st.caption("Number of companies getting better, worse, or staying the same")
                    col1, col2, col3 = st.columns(3)
                    
                    improving = len(dashboard.trend_analysis[dashboard.trend_analysis['trend_category'] == 'improving'])
                    declining = len(dashboard.trend_analysis[dashboard.trend_analysis['trend_category'] == 'declining'])
                    stable = len(dashboard.trend_analysis[dashboard.trend_analysis['trend_category'] == 'stable'])
                    
                    with col1:
                        st.metric("Getting Better", improving, delta=improving if improving > 0 else None)
                    with col2:
                        st.metric("Getting Worse", declining, delta=-declining if declining > 0 else None)
                    with col3:
                        st.metric("No Change", stable)
                
                if not dashboard.risk_distribution.empty:
                    st.subheader("Risk Levels")
                    st.caption("How many companies fall into each risk category")
                    risk_counts = dashboard.risk_distribution['risk_category'].value_counts()
                    st.bar_chart(risk_counts)
                    
            except Exception as e:
                st.error(f"Could not analyze {selected_sector} industry: {str(e)}")

with tab3:
    st.subheader("Compare Industries")
    st.caption("See which industries have the highest and lowest climate risks")
    
    num_sectors = st.slider("Number of industries to compare", 3, 8, 5, help="Choose how many industries to include in the comparison")
    
    if st.button("Compare Industries", type="primary"):
        with st.spinner("Comparing industries..."):
            try:
                top_sectors = list(company_sectors[:num_sectors])
                heatmap = generate_climate_risk_heatmap(fact_company, fact_benchmark, sectors=top_sectors)
                
                if not heatmap.heatmap_data.empty:
                    st.subheader("Risk Comparison Chart")
                    st.caption("Red means higher risk, green means lower risk")
                    st.dataframe(heatmap.heatmap_data.style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)
                
                if not heatmap.sector_averages.empty:
                    st.subheader("Industry Risk Summary")
                    avg_display = heatmap.sector_averages[['sector', 'overall_risk_score', 'companies_analyzed']]
                    avg_display = avg_display.rename(columns={
                        'sector': 'Industry',
                        'overall_risk_score': 'Risk Level',
                        'companies_analyzed': 'Companies Included'
                    }).copy()
                    avg_display.index = range(1, len(avg_display) + 1)
                    st.dataframe(avg_display, use_container_width=True)
                
                if not heatmap.scenario_impact.empty:
                    st.subheader("Climate Target Impact")
                    st.caption("How different climate goals affect risk levels")
                    scenario_impact_display = heatmap.scenario_impact[['scenario', 'average_risk', 'risk_range', 'sectors_analyzed']]
                    scenario_impact_display = scenario_impact_display.rename(columns={
                        'scenario': 'Climate Target',
                        'average_risk': 'Average Risk',
                        'risk_range': 'Risk Range',
                        'sectors_analyzed': 'Industries Included'
                    }).copy()
                    scenario_impact_display.index = range(1, len(scenario_impact_display) + 1)
                    st.dataframe(scenario_impact_display, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Could not compare industries: {str(e)}")

st.markdown("---")
st.caption("This page helps you quickly understand climate performance patterns across different industries.")
