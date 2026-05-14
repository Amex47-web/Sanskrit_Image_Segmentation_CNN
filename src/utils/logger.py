"""
logger.py — Logging Configuration
====================================

Sets up Python logging and optional TensorBoard writer.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


def setup_logger(
    name: str = "sanskrit_seg",
    log_dir: Optional[str | Path] = None,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name:        Logger name.
        log_dir:     Directory for log file (None = no file logging).
        log_level:   Logging level (e.g. logging.INFO).
        log_to_file: Whether to write logs to a file.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_tensorboard_writer(
    log_dir: str | Path,
    enabled: bool = True,
) -> Optional[SummaryWriter]:
    """
    Create a TensorBoard SummaryWriter.

    Args:
        log_dir: Directory for TensorBoard event files.
        enabled: If False, returns None.

    Returns:
        SummaryWriter instance, or None if disabled.
    """
    if not enabled:
        return None

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))
