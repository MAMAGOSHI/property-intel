# kaggle_data.py — loads and cleans the Roodepoort property listings CSV

import pandas as pd
import yaml
import logging
from pathlib import Path
from property_intel.sources.base import BaseDataSource
from property_intel.exceptions import DataSourceError

logger = logging.getLogger("property_intel")


class KagglePropertyData(BaseDataSource):
    """
    Loads the Roodepoort property listings CSV downloaded from Kaggle.
    Inherits from BaseDataSource — must implement load().
    """

    def __init__(self) -> None:
        super().__init__(name="KagglePropertyData")
        # read the file path from config.yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.file_path = Path(config["data"]["listings_file"])
        self.min_price = config["cleaning"]["min_price"]
        self.max_price = config["cleaning"]["max_price"]
        self.min_area = config["cleaning"]["min_area"]
        self.max_area = config["cleaning"]["max_area"]

    def load(self) -> pd.DataFrame:
        """
        Reads the CSV, cleans it, and returns a tidy DataFrame.
        """
        if not self.file_path.exists():
            raise DataSourceError(f"File not found: {self.file_path}")

        logger.info(f"Loading property data from {self.file_path}")
        df = pd.read_csv(self.file_path)
        logger.info(f"Loaded {len(df)} raw rows")

        df = self._clean(df)
        logger.info(f"{len(df)} rows remaining after cleaning")
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the raw CSV:
        - Converts price from 'R 1 350 000' to 1350000.0
        - Replaces 'Not specified' with NaN
        - Drops rows with no price, no area, no location
        - Removes price and area outliers
        """
        # replace "Not specified" with NaN everywhere
        df = df.replace("Not specified", pd.NA)

        # clean the price column — remove R, spaces, commas
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("R", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip()
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        # convert area and bedrooms to numbers
        df["area_sqm"] = pd.to_numeric(df["area_sqm"], errors="coerce")
        df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
        df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")

        # drop rows missing the three most important columns
        df = df.dropna(subset=["price", "area_sqm", "location"])

        # remove outliers using config thresholds
        df = df[df["price"].between(self.min_price, self.max_price)]
        df = df[df["area_sqm"].between(self.min_area, self.max_area)]

        # drop duplicates
        df = df.drop_duplicates()

        # clean up the location column
        df["location"] = df["location"].str.strip().str.title()

        return df