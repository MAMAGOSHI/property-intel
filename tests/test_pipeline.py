# test_pipeline.py — 10 unit tests covering cleaning, features, and sources

import pytest
import pandas as pd
import numpy as np
from property_intel.sources.kaggle_data import KagglePropertyData
from property_intel.sources.crime_data import CrimeStatsData
from property_intel.sources.gautrain_data import GautrainData
from property_intel.features import build_features
from property_intel.exceptions import CleaningError


def make_source():
    """Creates a KagglePropertyData without reading config or disk."""
    source = KagglePropertyData.__new__(KagglePropertyData)
    source.min_price = 100000
    source.max_price = 50000000
    source.min_area  = 10
    source.max_area  = 10000
    return source


def test_price_converts_to_float(sample_raw_df):
    """Price strings like 'R 1 350 000' must become 1350000.0"""
    cleaned = make_source()._clean(sample_raw_df)
    assert cleaned["price"].dtype == float
    assert 1350000.0 in cleaned["price"].values


def test_not_specified_price_dropped(sample_raw_df):
    """Rows where price is 'Not specified' must be dropped."""
    cleaned = make_source()._clean(sample_raw_df)
    assert len(cleaned) == 2


def test_price_below_minimum_dropped():
    """Prices below min_price must be removed."""
    source = make_source()
    df = pd.DataFrame({
        "price": ["R 50 000", "R 1 000 000"],
        "area_sqm": [80, 100], "bedrooms": [2, 3],
        "bathrooms": [1, 2], "location": ["Honeydew", "Ruimsig"],
        "property_type": ["Apartment", "House"],
        "description": ["d", "d"], "listing_id": [1, 2],
        "agent": [None, None], "on_show": [None, None],
        "special_tag": [None, None], "url": ["u", "u"],
    })
    cleaned = source._clean(df)
    assert len(cleaned) == 1
    assert cleaned["price"].iloc[0] == 1000000.0


def test_price_above_maximum_dropped():
    """Prices above max_price must be removed."""
    source = make_source()
    df = pd.DataFrame({
        "price": ["R 100 000 000", "R 1 000 000"],
        "area_sqm": [500, 100], "bedrooms": [5, 3],
        "bathrooms": [4, 2], "location": ["Ruimsig", "Honeydew"],
        "property_type": ["House", "House"],
        "description": ["d", "d"], "listing_id": [1, 2],
        "agent": [None, None], "on_show": [None, None],
        "special_tag": [None, None], "url": ["u", "u"],
    })
    cleaned = source._clean(df)
    assert len(cleaned) == 1


def test_crime_data_has_correct_columns():
    """CrimeStatsData must return location and crime_index columns."""
    df = CrimeStatsData().load()
    assert "location" in df.columns
    assert "crime_index" in df.columns


def test_crime_index_values_in_range():
    """All crime_index values must be between 1 and 10."""
    df = CrimeStatsData().load()
    assert df["crime_index"].between(1, 10).all()


def test_gautrain_has_correct_columns():
    """GautrainData must return location and gautrain_km columns."""
    df = GautrainData().load()
    assert "location" in df.columns
    assert "gautrain_km" in df.columns


def test_gautrain_distances_positive():
    """All distances must be greater than zero."""
    df = GautrainData().load()
    assert (df["gautrain_km"] > 0).all()


def test_build_features_raises_on_empty_dataframe():
    """build_features must raise CleaningError on empty input."""
    with pytest.raises(CleaningError):
        build_features(pd.DataFrame())


def test_build_features_drops_text_columns(sample_properties_df):
    """build_features must drop description, url, agent etc."""
    df = sample_properties_df.copy()
    df["description"]   = "some text"
    df["url"]           = "http://example.com"
    df["property_type"] = "House"
    result = build_features(df)
    assert "description" not in result.columns
    assert "url" not in result.columns