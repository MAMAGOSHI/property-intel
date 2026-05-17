# base.py — the abstract parent class all data sources must inherit from

from abc import ABC, abstractmethod
import pandas as pd


class BaseDataSource(ABC):
    """
    Every data source must inherit from this class and implement load().
    This guarantees that the Pipeline can treat all sources the same way.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load and return the data as a pandas DataFrame.
        Must be implemented by every child class.
        """
        pass

    def __repr__(self) -> str:
        return f"<DataSource: {self.name}>"