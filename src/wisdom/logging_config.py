"""Structured logging setup for the Wisdom System."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the wisdom system logger."""
    logger = logging.getLogger("wisdom")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module."""
    return logging.getLogger(f"wisdom.{name}")
