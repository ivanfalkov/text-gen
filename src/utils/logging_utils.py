"""Project-wide logging setup."""
from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"

# Сторонние библиотеки, которые на DEBUG/INFO заваливают консоль.
_NOISY_LOGGERS = (
    "httpcore",
    "httpx",
    "urllib3",
    "filelock",
    "fsspec",
    "datasets",
    "huggingface_hub",
    "transformers",
    "absl",
    "asyncio",
    "matplotlib",
    "PIL",
)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once. Safe to call multiple times."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        root.addHandler(handler)

    root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)