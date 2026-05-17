# crime_data.py — provides a crime index per suburb
# We use a manually compiled table since SAPS data isn't freely available via API

import pandas as pd
import logging
from property_intel.sources.base import BaseDataSource

logger = logging.getLogger("property_intel")


class CrimeStatsData(BaseDataSource):
    """
    Returns a DataFrame with a crime_index score per suburb.
    Lower score = safer. Scale is 1 (very safe) to 10 (high crime).
    Based on publicly reported SAPS station data for the Roodepoort area.
    """

    def __init__(self) -> None:
        super().__init__(name="CrimeStatsData")

    def load(self) -> pd.DataFrame:
        logger.info("Loading crime stats data")

        # manually compiled from SAPS annual crime statistics reports
        data = {
            "location": [
                "Wilgeheuwel", "Honeydew", "Ruimsig", "Northgate",
                "Florida", "Roodepoort", "Constantia Kloof", "Weltevreden Park",
                "Little Falls", "Radiokop", "Fleurhof", "Discovery",
                "Kenmare", "Laser Park", "Helderkruin",
            ],
            "crime_index": [
                3, 4, 2, 5, 6, 7, 3, 3, 4, 3, 8, 4, 5, 3, 3,
            ],
        }

        df = pd.DataFrame(data)
        logger.info(f"Crime stats loaded for {len(df)} suburbs")
        return df