Methodology (summary)

Data → Tables
- Company fact table: (company, sector, year, intensity, unit[, geography])
- Benchmark fact table: (sector, region, scenario, year, benchmark, unit)
- Types: categorical dims, int16 year, float32 measures.

Views (functions)
- view_pathway(sector, company, region, scenarios, year_range, exact_region=False)
  Returns: company_df, scenario_map, bands
- view_outliers(sector, scenario, region, year_range, exact_region=False, k=10)
  Returns: best_df, worst_df

Region rule
- Prefer exact region; if unavailable, fall back to Global unless exact_region=True.

CBD
- Sum over trapezoids: ((c0-b0)+(c1-b1))/2 * (y1-y0) across adjacent years.
- Negative: company is below benchmark; positive: above.

Scenario normalization
- '1.5 Degrees' → '1.5°C'; 'Below 2 Degrees' → 'Below 2°C'; 'International/Paris Pledges' → 'National Pledges'.