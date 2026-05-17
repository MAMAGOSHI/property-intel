# gautrain_data.py — computes distance from each suburb to nearest Gautrain station

import pandas as pd
import numpy as np
import logging
from property_intel.sources.base import BaseDataSource

logger = logging.getLogger("property_intel")


class GautrainData(BaseDataSource):
    """
    Returns a DataFrame with the distance (in km) from each suburb
    to the nearest Gautrain station. Closer = more desirable = higher price.
    Uses the Haversine formula to compute distances from coordinates.
    """

    # Gautrain station coordinates
    STATIONS = {
        "Sandton":      (-26.1073, 28.0564),
        "Rosebank":     (-26.1459, 28.0436),
        "Park":         (-26.1929, 28.0437),
        "Rhodesfield":  (-26.1369, 28.2217),
        "Marlboro":     (-26.0892, 28.1025),
        "Midrand":      (-25.9986, 28.1286),
        "Centurion":    (-25.8602, 28.1889),
        "Pretoria":     (-25.7479, 28.1878),
    }

    # approximate suburb coordinates for the Roodepoort area
    SUBURBS = {
        "Wilgeheuwel":       (-26.0850, 27.9200),
        "Honeydew":          (-26.0700, 27.9500),
        "Ruimsig":           (-26.0950, 27.8800),
        "Northgate":         (-26.1000, 27.9800),
        "Florida":           (-26.1700, 27.9000),
        "Roodepoort":        (-26.1628, 27.8729),
        "Constantia Kloof":  (-26.1100, 27.9100),
        "Weltevreden Park":  (-26.1050, 27.9650),
        "Little Falls":      (-26.0900, 27.9300),
        "Radiokop":          (-26.0800, 27.9400),
        "Fleurhof":          (-26.2000, 27.9100),
        "Discovery":         (-26.1500, 27.9200),
        "Kenmare":           (-26.1200, 27.9000),
        "Laser Park":        (-26.0750, 27.9100),
        "Helderkruin":       (-26.1000, 27.9050),
    }

    def __init__(self) -> None:
        super().__init__(name="GautrainData")

    def _haversine(self, lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
        """
        Calculates the distance in km between two GPS coordinates.
        This is the standard formula used for distances on a sphere.
        """
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def _nearest_station_km(self, lat: float, lon: float) -> float:
        """Returns the distance in km to the closest Gautrain station."""
        distances = [
            self._haversine(lat, lon, slat, slon)
            for slat, slon in self.STATIONS.values()
        ]
        return round(min(distances), 2)

    def load(self) -> pd.DataFrame:
        logger.info("Computing Gautrain distances for all suburbs")
        rows = []
        for suburb, (lat, lon) in self.SUBURBS.items():
            dist = self._nearest_station_km(lat, lon)
            rows.append({"location": suburb, "gautrain_km": dist})

        df = pd.DataFrame(rows)
        logger.info(f"Gautrain distances computed for {len(df)} suburbs")
        return df