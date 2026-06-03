# features.py — turns the combined DataFrame into model-ready features

import pandas as pd
import logging
from pathlib import Path
from property_intel.exceptions import CleaningError

logger = logging.getLogger("property_intel")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the combined DataFrame for the model:
    - Fills missing bedrooms/bathrooms with the median
    - Adds price_per_sqm as an analysis column
    - One-hot encodes property_type and location
    - Drops columns the model cannot use
    - Saves the processed CSV
    """
    if df.empty:
        raise CleaningError("Cannot build features from an empty DataFrame")

    logger.info("Building features")

    # fill missing room counts with the median
    df["bedrooms"]  = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    # useful comparison column — but NOT used as a model feature
    df["price_per_sqm"] = df["price"] / df["area_sqm"]

    # one-hot encode property_type — converts text to 0/1 columns
    if "property_type" in df.columns:
        df = pd.get_dummies(df, columns=["property_type"], drop_first=True, dtype=int)

    # one-hot encode location — so the model knows which suburb matters
    if "location" in df.columns:
        df = pd.get_dummies(df, columns=["location"], drop_first=True, dtype=int)

    # drop columns the model cannot use
    cols_to_drop = ["description", "url", "agent",
                    "listing_id", "on_show", "special_tag"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # save processed data
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)
    logger.info(f"Features built — {df.shape[0]} rows, {df.shape[1]} columns")

    return df