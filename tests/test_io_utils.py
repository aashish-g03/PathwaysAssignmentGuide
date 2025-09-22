import pandas as pd
import numpy as np
import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from io_utils import find_year_columns, _load_geo_regions, load_tables

def test_find_year_columns():
    test_df = pd.DataFrame({
        'company': ['A', 'B'],
        'sector': ['Energy', 'Auto'],
        2020: [1.5, 2.1],
        2021: [1.4, 2.0],
        2022: [1.3, 1.9],
        'notes': ['good', 'ok'],
        1990: [2.0, 3.0], 
        2100: [0.5, 0.8], 
        1989: [9.9, 9.9], 
        2101: [0.1, 0.1]   
    })
    
    year_cols = find_year_columns(test_df)
    expected = [1990, 2020, 2021, 2022, 2100]
    
    assert year_cols == expected, f"Expected {expected}, got {year_cols}"
    assert len(year_cols) == 5


def test_find_year_columns_empty():
    test_df = pd.DataFrame({
        'company': ['A', 'B'],
        'sector': ['Energy', 'Auto'],
        'notes': ['good', 'ok']
    })
    
    year_cols = find_year_columns(test_df)
    assert year_cols == [], "Should return empty list when no year columns found"


def test_load_geo_regions():
    regions = _load_geo_regions("nonexistent.csv")
    assert regions == {}, "Should return empty dict for non-existent file"
    
    if os.path.exists("resources/geo_regions.csv"):
        regions = _load_geo_regions("resources/geo_regions.csv")
        assert isinstance(regions, dict), "Should return a dictionary"
        assert len(regions) > 0, "Should have some geographic region mappings"


def test_load_tables():
    try:
        fact_company, fact_benchmark = load_tables()
        
        assert fact_company is not None, "Company table should not be None"
        assert fact_benchmark is not None, "Benchmark table should not be None"
        assert isinstance(fact_company, pd.DataFrame), "Company data should be DataFrame"
        assert isinstance(fact_benchmark, pd.DataFrame), "Benchmark data should be DataFrame"
        
        assert len(fact_company) > 0, "Company table should have data"
        assert len(fact_benchmark) > 0, "Benchmark table should have data"
        
    except FileNotFoundError:
        pytest.skip("Data files not found - skipping integration test")
    except Exception as e:
        pytest.fail(f"Unexpected error loading tables: {e}")


# def test_company_data_structure():
#     test_df = pd.DataFrame({




if __name__ == "__main__":
    print("Running basic tests...")
    
    test_find_year_columns()
    print("Year column detection test passed")
    
    test_find_year_columns_empty()
    print("Empty year columns test passed")
    
    test_load_geo_regions()
    print("Geographic regions test passed")
    
    test_load_tables()
    print("Table loading test passed")
    
    # test_company_data_structure()
    # print("Company data structure test passed")
    
    print("IO utils tests completed")
