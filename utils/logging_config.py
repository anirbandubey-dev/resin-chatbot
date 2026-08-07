"""Application logging configuration."""

import logging

from config import LOG_DIRECTORY, LOG_FILE


def configure_logging() -> None:
    """Configure a single UTF-8 file handler without duplicating rerun handlers."""
    root_logger = logging.getLogger()
    if any(handler.name == "supportgpt-file" for handler in root_logger.handlers):
        return

    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.name = "supportgpt-file"
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
