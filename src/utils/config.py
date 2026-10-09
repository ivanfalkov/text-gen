"""Config loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML config from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)