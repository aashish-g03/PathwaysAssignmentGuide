import numpy as np
import plotly.graph_objs as go
import pandas as pd

def _band(fig, x, lower, upper, name, color='rgba(108, 117, 125, 0.3)', opacity=0.3):
    fig.add_trace(go.Scatter(x=x, y=upper, mode='lines', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=lower, mode='lines', line=dict(width=0, color='rgba(0,0,0,0)'),
                             fill='tonexty', name=name, fillcolor=color, 
                             opacity=opacity, hoverinfo='skip'))

def pathway_figure(df_company: pd.DataFrame, scenario_map: dict, unit_hint: str, company: str):
    fig = go.Figure()
    years = sorted(df_company['Year'].unique())

    green = scenario_map.get('1.5°C')
    below2 = scenario_map.get('Below 2°C')
    pledges = scenario_map.get('National Pledges')

    if green is not None and below2 is not None and not green.empty and not below2.empty:
        green_clean = green.groupby('Year')['Benchmark'].mean().reset_index()
        below2_clean = below2.groupby('Year')['Benchmark'].mean().reset_index()
        
        y1 = green_clean.set_index('Year').reindex(years)['Benchmark'].values
        y2 = below2_clean.set_index('Year').reindex(years)['Benchmark'].values
        _band(fig, years, np.minimum(y1,y2), np.maximum(y1,y2), 'Between 1.5°C and Below 2°C')

    if below2 is not None and pledges is not None and not below2.empty and not pledges.empty:
        below2_clean = below2.groupby('Year')['Benchmark'].mean().reset_index()
        pledges_clean = pledges.groupby('Year')['Benchmark'].mean().reset_index()
        
        y2 = below2_clean.set_index('Year').reindex(years)['Benchmark'].values
        y3 = pledges_clean.set_index('Year').reindex(years)['Benchmark'].values
        _band(fig, years, np.minimum(y2,y3), np.maximum(y2,y3), 'Above Below 2°C')

    cs = df_company.sort_values('Year')
    fig.add_trace(go.Scatter(x=cs['Year'], y=cs['Intensity'], mode='lines+markers', name=company))

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=f'Intensity ({unit_hint})' if unit_hint else 'Intensity',
        hovermode='x unified',
        margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#262730'),
        xaxis=dict(gridcolor='#E5E7EB', zeroline=False),
        yaxis=dict(gridcolor='#E5E7EB', zeroline=False)
    )
    return fig
