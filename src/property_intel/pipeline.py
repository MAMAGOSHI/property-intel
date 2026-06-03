# pipeline.py — combines all three data sources into one clean DataFrame

import pandas as pd
import logging
from pathlib import Path
from property_intel.sources.kaggle_data import KagglePropertyData
from property_intel.sources.crime_data import CrimeStatsData
from property_intel.sources.gautrain_data import GautrainData

logger = logging.getLogger("property_intel")


class Pipeline:
    """
    Loads all three data sources and merges them into one DataFrame.
    This is the single entry point for getting clean combined data.
    """

    def __init__(self) -> None:
        self.sources = {
            "properties": KagglePropertyData(),
            "crime":      CrimeStatsData(),
            "gautrain":   GautrainData(),
        }

    def run(self) -> pd.DataFrame:
        """
        Loads each source, merges on location, saves interim CSV,
        and returns the combined DataFrame.
        """
        logger.info("Pipeline starting")

        properties = self.sources["properties"].load()
        crime      = self.sources["crime"].load()
        gautrain   = self.sources["gautrain"].load()

        # merge crime index onto properties by suburb name
        df = properties.merge(crime, on="location", how="left")
        logger.info(f"After crime merge: {len(df)} rows")

        # merge gautrain distances onto properties by suburb name
        df = df.merge(gautrain, on="location", how="left")
        logger.info(f"After gautrain merge: {len(df)} rows")

        # fill suburbs not in our lookup tables with the median
        df["crime_index"] = df["crime_index"].fillna(df["crime_index"].median())
        df["gautrain_km"] = df["gautrain_km"].fillna(df["gautrain_km"].median())

        # save interim data so we can inspect it later
        Path("data/interim").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/interim/combined.csv", index=False)
        logger.info("Pipeline complete — saved to data/interim/combined.csv")

        return df