# logging_config.py — call setup_logging() once at the start of any script

import logging
import yaml
from pathlib import Path


def setup_logging() -> logging.Logger:
    """
    Reads log level and format from config.yaml and sets up the root logger.
    Returns a logger you can use immediately.
    """
    config_path = Path("config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    fmt = log_cfg.get("format", "%(asctime)s — %(levelname)s — %(message)s")

    logging.basicConfig(level=level, format=fmt)
    logger = logging.getLogger("property_intel")
    return logger