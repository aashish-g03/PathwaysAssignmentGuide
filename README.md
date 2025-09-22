# TPI Data Assessment

This is a Streamlit, python application that helps users analyze and understand the data for TPI (Transition Pathway Initiative). This is a tool to help users understand the data and interact with it in a dashboard kind manner to make financial decisions. We are interacting with csv TPI Data in similar way as we interact with database in SQL. We are using pandas to read the csv files and perform operations on it.

## Sources

1. [`data/Company_Latest_Assessments_5.0.csv`](./data/Company_Latest_Assessments_5.0.csv) (Latest Assessment Data for Comapny wise emssions)
2. [`data/Sector_Benchmarks_19092025.csv`](./data/Sector_Benchmarks_19092025.csv) (Benchmark Data for Sector wise emissions as per temperature goals and Pledges)

## Pages

### 1. Company Climate Performance ([`Company Climate Performance.py`](./Company%20Climate%20Performance.py))

- This page allows users to view the climate performance of a specific company. User can select Sector, Company, Region, Benchmark Scenario and Year Window to view the data.
- We have plotted graph by using Years as X axis and Intensity i.e. Emissions per Unit as Y axis.
- We have layered the benchmarks in varying temperature outcomes, different colours below/between/above, as background to company data.
- User can also download the data as csv file on basis of selected filters.

### 2. Outliers ([`pages/1_Outliers.py`](./pages/1_Outliers.py))

- This page allows users to discover companies that are significant outliers compared to their sector peers, either for good or bad transition plans. User can select Sector, Company, Region, Benchmark Scenario and Year Window to view the data.
- We are calculating the outliers using Z score and then ranking them based on the significance of the outlier.
- User can also download the data as csv file on basis of selected filters.
- We are giving top K companies for each sector and scenario on basis of user selection and ranking is done on basis of standard deviation from the sector bencmarks (Z-Score).

### 3. Ask TPI Agent ([`pages/2_Ask_TPI_Agent.py`](./pages/2_Ask_TPI_Agent.py))

- This page allows users to ask questions to the TPI Agent. User can ask questions related to the data in the sources.
- This agent determined the intent from the user question and calls different handlign functions to get the expected result for displaying to the user.
- This is something inspired from Intent based chat bot I had worked on previously at Leena.
- Users can ask questions related to definitions, company data, sector data, business analytics, etc.
- For adding new intents or new handlers, we need to add them in the [`src/agent/handlers.py`](./src/agent/handlers.py), [`src/agent/agent_config.json`](./src/agent/agent_config.json) and [`src/agent/router.py`](./src/agent/router.py) files.

### 4. Industry Analysis ([`pages/3_Industry_Analysis.py`](./pages/3_Industry_Analysis.py))

- This page allows users to view the industry insights either as an overview of complete market, as sector wise deepdive or compare industry on basis of climate risk scores.
- We are calculating risk score on basis of deviation from the sector benchmarks.
- In Market Overview, we are showing the risk score for each sector and scenario.
- In Sector Deepdive, we are showing the risk score for each company in the selected sector and scenario.
- In Industry Comparison, we are showing the risk score for each company in the selected sector on basis of different scenarios.
- In Risk Comparison, we are showing the risk score for each sector on basis of different scenarios with different colours below/between/above, as background to company data.

## Folder Structure

```
PathwaysAssignmentGuide/
│
├── Company Climate Performance.py    # Main page - Company climate performance analysis
│
├── pages/
│   ├── 1_Outliers.py                # Outlier detection and analysis
│   ├── 2_Ask_TPI_Agent.py           # Question-answering agent interface
│   └── 3_Industry_Analysis.py       # Industry analysis and insights
│
├── src/
│   ├── io_utils.py                  # Data loading and CSV operations
│   ├── analytics.py                 # Core analytical functions and views
│   ├── business_intelligence.py     # Business intelligence and dashboards
│   ├── outliers.py                  # Outlier detection algorithms
│   ├── plot_utils.py                # Plotting and visualization utilities
│   │
│   └── agent/                       # Intent-based question answering system
│       ├── router.py                # Intent detection and routing
│       ├── handlers.py              # Query handlers for different intents
│       └── agent_config.json        # Agent configuration and keywords
│
├── data/
│   ├── Company_Latest_Assessments_5.0.csv    # Company emissions data
│   └── Sector_Benchmarks_19092025.csv        # Sector benchmark data
│
├── resources/
│   ├── methodology.md              # TPI methodology documentation
│   ├── data_dictionary.json        # Data field definitions
│   └── geo_regions.csv            # Geographic region mappings
│
├── requirements.txt                # Python package dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```
