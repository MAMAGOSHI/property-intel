# exceptions.py — custom error classes for the whole project


class PropertyIntelError(Exception):
    """Base error for all project errors."""
    pass


class DataSourceError(PropertyIntelError):
    """Raised when a data source fails to load or is missing."""
    pass


class CleaningError(PropertyIntelError):
    """Raised when the cleaning pipeline hits an unexpected problem."""
    pass


class ModelError(PropertyIntelError):
    """Raised when model training or prediction fails."""
    pass