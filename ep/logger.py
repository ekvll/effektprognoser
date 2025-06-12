# -*- coding: utf-8 -*-
"""
Logger configuration for the Effektprognoser application.

This module sets up a logger named "effektprognoser" that outputs logs
to both the console and a log file ("effektprognoser.log").

Usage:
    from ep.logger import logger

    logger.info("This is an info message.")
    logger.error("This is an error message.")
"""

import logging
from pathlib import Path

from tqdm import tqdm

# Ensure the log directory exists
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logger
logger = logging.getLogger("ep")
logger.setLevel(logging.DEBUG)  # Set minimum log level

# Prevent adding handlers multiple times if module is reloaded
if not logger.hasHandlers():
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # File handler - logs to file with formatting
    file_handler = logging.FileHandler(log_dir / "ep.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Uncomment below to enable console logging
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)


def log_write(message: str) -> None:
    """
    Write a message to tqdm terminal output and to the log file at DEBUG level.

    Args:
        message (str): Message to log and print.
    """
    tqdm.write(message)
    logger.debug(message)
