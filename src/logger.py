from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_PATH = DEFAULT_LOG_DIR / "pipeline.log"


def setup_logging(
    name: str = "motor_insurance",
    level: int = logging.INFO,
    log_path: Path | None = None,
    console: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    resolved_log_path = log_path or DEFAULT_LOG_PATH
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "motor_insurance") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger
