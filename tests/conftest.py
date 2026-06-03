# conftest.py — shared test fixtures

import pytest
import pandas as pd


@pytest.fixture
def sample_properties_df():
    """A small clean DataFrame that mimics real property data."""
    return pd.DataFrame({
        "price":       [1350000, 659999, 2500000, 890000, 1100000],
        "area_sqm":    [132, 65, 210, 95, 120],
        "bedrooms":    [3, 2, 4, 2, 3],
        "bathrooms":   [2, 1, 3, 1, 2],
        "location":    ["Wilgeheuwel", "Honeydew", "Ruimsig",
                        "Northgate", "Florida"],
        "crime_index": [3, 4, 2, 5, 6],
        "gautrain_km": [12.3, 10.1, 15.2, 8.4, 9.7],
    })


@pytest.fixture
def sample_raw_df():
    """Mimics the raw CSV before cleaning."""
    return pd.DataFrame({
        "price":         ["R 1 350 000", "R 659 999", "Not specified"],
        "area_sqm":      [132, 65, 80],
        "bedrooms":      [3, "Not specified", 2],
        "bathrooms":     ["Not specified", 1, 2],
        "location":      ["Wilgeheuwel", "Honeydew", "Roodepoort"],
        "property_type": ["Townhouse", "Apartment", "House"],
        "description":   ["desc1", "desc2", "desc3"],
        "listing_id":    [1, 2, 3],
        "agent":         ["Not specified", "Not specified", "Not specified"],
        "on_show":       [None, None, None],
        "special_tag":   [None, None, None],
        "url":           ["url1", "url2", "url3"],
    })