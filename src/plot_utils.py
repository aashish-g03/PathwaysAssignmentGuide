import numpy as np
import plotly.graph_objs as go
import pandas as pd

PALETTE = {
    "green": "#16A34A",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "gray": "#1F2937",
    "orange_light": "#FDBA74"
}

def _band(fig, x, lower, upper, name, color_hex, opacity=0.20):
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)
    mask = ~(np.isnan(lower) | np.isnan(upper))
    x_m = np.array(x)[mask]
    lo = lower[mask]; up = upper[mask]
    if len(x_m) == 0:
        return

    fig.add_trace(go.Scatter(
        x=x_m, y=lo, mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x_m, y=up, mode="lines",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        fill="tonexty",
        fillcolor=f"rgba({int(color_hex[1:3],16)},{int(color_hex[3:5],16)},{int(color_hex[5:7],16)},{opacity})",
        name=name, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x_m, y=up, mode="lines",
        line=dict(width=1, color=color_hex),
        showlegend=False, hoverinfo="skip"
    ))

def pathway_figure(df_company: pd.DataFrame, scenario_map: dict, unit_hint: str, company: str):
    fig = go.Figure()
    years = sorted(df_company["Year"].unique())
    cs = df_company.sort_values("Year")

    scenario_colors = {
        "1.5°C": PALETTE["green"],
        "Below 2°C": PALETTE["amber"],
        "2°C": PALETTE["orange_light"],
        "2°C (High Efficiency)": PALETTE["orange_light"],
        "2°C (Shift-Improve)": PALETTE["orange_light"],
        "Paris Pledges": PALETTE["red"],
        "National Pledges": PALETTE["red"],
        "International Pledges": PALETTE["red"]
    }

    green = scenario_map.get("1.5°C")
    below2 = scenario_map.get("Below 2°C")
    pledges = None
    for nm in ("National Pledges", "Paris Pledges", "International Pledges"):
        if nm in scenario_map and not scenario_map[nm].empty:
            pledges = scenario_map[nm]; break

    # Below 1.5°C
    if green is not None and not green.empty:
        g = green.groupby("Year")["Benchmark"].mean().reset_index()
        y1 = g.set_index("Year").reindex(years)["Benchmark"].values.astype(float)
        baseline = max(0.0, float(np.nanmin([np.nanmin(y1), cs["Intensity"].min()])))
        _band(fig, years, np.full_like(y1, baseline), y1, "Below 1.5°C", PALETTE["green"], opacity=0.12)

    # Between 1.5°C and Below 2°C
    if green is not None and not green.empty and below2 is not None and not below2.empty:
        g = green.groupby("Year")["Benchmark"].mean().reset_index()
        b2 = below2.groupby("Year")["Benchmark"].mean().reset_index()
        y1 = g.set_index("Year").reindex(years)["Benchmark"].values
        y2 = b2.set_index("Year").reindex(years)["Benchmark"].values
        _band(fig, years, np.minimum(y1, y2), np.maximum(y1, y2),
              "Between 1.5°C and 2°C", PALETTE["amber"], opacity=0.18)

    # Above 2°C
    if below2 is not None and not below2.empty and pledges is not None and not pledges.empty:
        b2 = below2.groupby("Year")["Benchmark"].mean().reset_index()
        pl = pledges.groupby("Year")["Benchmark"].mean().reset_index()
        y2 = b2.set_index("Year").reindex(years)["Benchmark"].values
        y3 = pl.set_index("Year").reindex(years)["Benchmark"].values
        _band(fig, years, np.minimum(y2, y3), np.maximum(y2, y3),
              "Above 2°C", PALETTE["red"], opacity=0.20)

    for scenario_name, scenario_data in scenario_map.items():
        if scenario_data is None or scenario_data.empty:
            continue
        s = scenario_data.groupby("Year")["Benchmark"].mean().reset_index()
        color = scenario_colors.get(scenario_name, "#6B7280")
        line_style = dict(color=color, width=1.5)
        if scenario_name == "1.5°C":
            line_style["dash"] = "dash"
        fig.add_trace(go.Scatter(
            x=s["Year"], y=s["Benchmark"], mode="lines",
            name=scenario_name, line=line_style
        ))

    fig.add_trace(go.Scatter(
        x=cs["Year"], y=cs["Intensity"],
        mode="lines+markers",
        name=company,
        line=dict(color=PALETTE["gray"], width=2.5),
        marker=dict(size=5, color=PALETTE["gray"])
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=f"Intensity ({unit_hint})" if unit_hint else "Intensity",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="simple_white",
        xaxis=dict(gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(gridcolor="#E5E7EB", zeroline=False)
    )
    return fig
