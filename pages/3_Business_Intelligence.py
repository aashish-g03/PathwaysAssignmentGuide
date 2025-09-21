import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.io_utils import load_tables
from src.business_intelligence import create_sector_performance_summary, generate_climate_risk_heatmap, generate_executive_summary

st.set_page_config(
    page_title='Business Intelligence - TPI Dashboard',
    layout='wide',
    initial_sidebar_state='expanded'
)

@st.cache_data(show_spinner=True)
def load_fact_tables():
    return load_tables()

fact_company, fact_benchmark = load_fact_tables()

st.title('Climate Risk Business Intelligence')
st.caption('Pre-computed analytics for investment teams and risk management')

tab1, tab2, tab3 = st.tabs(["Executive Summary", "Sector Analysis", "Risk Heatmap"])

with tab1:
    st.subheader("Executive Climate Risk Summary")
    
    if st.button("Generate Executive Summary", type="primary"):
        with st.spinner("Generating executive summary..."):
            try:
                summary = generate_executive_summary(fact_company, fact_benchmark)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sectors Analyzed", summary['total_sectors_analyzed'])
                with col2:
                    st.metric("Analysis Date", summary['analysis_date'][:10])
                
                if summary['key_findings']:
                    st.subheader("Key Findings")
                    findings_df = pd.DataFrame(summary['key_findings'])
                    findings_display = findings_df.rename(columns={
                        'sector': 'Sector',
                        'companies_analyzed': 'Companies Analyzed',
                        'high_risk_percentage': 'High Risk Percentage',
                        'data_coverage': 'Data Coverage'
                    }).copy()
                    findings_display.index = range(1, len(findings_display) + 1)
                    st.dataframe(findings_display, use_container_width=True)
                
                if summary['risk_alerts']:
                    st.subheader("Risk Alerts")
                    for alert in summary['risk_alerts']:
                        st.warning(f"**{alert['sector']}**: {alert['message']}")
                
                if summary['opportunities']:
                    st.subheader("Investment Opportunities")
                    for opp in summary['opportunities']:
                        st.success(f"**{opp['sector']}**: {opp['message']}")
                        
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

with tab2:
    st.subheader("Sector Performance Analysis")
    
    company_sectors = fact_company.index.get_level_values('sector').unique()
    benchmark_sectors = fact_benchmark.index.get_level_values('sector').unique()
    available_sectors = sorted(list(set(company_sectors) & set(benchmark_sectors)))
    
    selected_sector = st.selectbox("Select Sector", available_sectors)
    
    if st.button("Analyze Sector", type="primary"):
        with st.spinner(f"Analyzing {selected_sector} sector..."):
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
                    st.subheader("Sector Rankings")
                    rankings_display = dashboard.sector_rankings[['company', 'sector_rank', 'latest_intensity', 'latest_year']].head(10)
                    rankings_display = rankings_display.rename(columns={
                        'company': 'Company',
                        'sector_rank': 'Sector Rank',
                        'latest_intensity': 'Latest Intensity',
                        'latest_year': 'Latest Year'
                    }).copy()
                    rankings_display.index = range(1, len(rankings_display) + 1)
                    st.dataframe(rankings_display, use_container_width=True)
                
                if not dashboard.trend_analysis.empty:
                    st.subheader("Trend Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    improving = len(dashboard.trend_analysis[dashboard.trend_analysis['trend_category'] == 'improving'])
                    declining = len(dashboard.trend_analysis[dashboard.trend_analysis['trend_category'] == 'declining'])
                    stable = len(dashboard.trend_analysis[dashboard.trend_analysis['trend_category'] == 'stable'])
                    
                    with col1:
                        st.metric("Improving", improving, delta=improving if improving > 0 else None)
                    with col2:
                        st.metric("Declining", declining, delta=-declining if declining > 0 else None)
                    with col3:
                        st.metric("Stable", stable)
                
                if not dashboard.risk_distribution.empty:
                    st.subheader("Risk Distribution")
                    risk_counts = dashboard.risk_distribution['risk_category'].value_counts()
                    st.bar_chart(risk_counts)
                    
            except Exception as e:
                st.error(f"Error analyzing sector: {str(e)}")

with tab3:
    st.subheader("Climate Risk Heatmap")
    st.caption("Sector x Scenario risk analysis for portfolio management")
    
    num_sectors = st.slider("Number of top sectors to analyze", 3, 8, 5)
    
    if st.button("Generate Risk Heatmap", type="primary"):
        with st.spinner("Generating risk heatmap..."):
            try:
                top_sectors = list(company_sectors[:num_sectors])
                heatmap = generate_climate_risk_heatmap(fact_company, fact_benchmark, sectors=top_sectors)
                
                if not heatmap.heatmap_data.empty:
                    st.subheader("Risk Heatmap (Average Risk Scores)")
                    st.dataframe(heatmap.heatmap_data.style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)
                
                if not heatmap.sector_averages.empty:
                    st.subheader("Sector Risk Averages")
                    avg_display = heatmap.sector_averages[['sector', 'overall_risk_score', 'companies_analyzed']]
                    avg_display = avg_display.rename(columns={
                        'sector': 'Sector',
                        'overall_risk_score': 'Overall Risk Score',
                        'companies_analyzed': 'Companies Analyzed'
                    }).copy()
                    avg_display.index = range(1, len(avg_display) + 1)
                    st.dataframe(avg_display, use_container_width=True)
                
                if not heatmap.scenario_impact.empty:
                    st.subheader("Scenario Impact Analysis")
                    scenario_impact_display = heatmap.scenario_impact[['scenario', 'average_risk', 'risk_range', 'sectors_analyzed']]
                    scenario_impact_display = scenario_impact_display.rename(columns={
                        'scenario': 'Scenario',
                        'average_risk': 'Average Risk',
                        'risk_range': 'Risk Range',
                        'sectors_analyzed': 'Sectors Analyzed'
                    }).copy()
                    scenario_impact_display.index = range(1, len(scenario_impact_display) + 1)
                    st.dataframe(scenario_impact_display, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error generating heatmap: {str(e)}")

st.markdown("---")
st.caption("Business Intelligence features provide pre-computed analytics for faster decision-making in investment and risk management workflows.")
