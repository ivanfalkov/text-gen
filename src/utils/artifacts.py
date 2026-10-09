"""Local artifact I/O."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_run_dir(root: Path, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("created run dir: %s", run_dir)
    return run_dir


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug("saved json: %s", path)


def save_split(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [c for c in ("id", "dialogue", "summary") if c in data.columns]
    data[columns].to_json(path, orient="records", lines=True, force_ascii=False)
    logger.info("saved split artifact: %d rows -> %s", len(data), path)


def save_predictions(data: pd.DataFrame, predictions: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({
        "id": data["id"] if "id" in data.columns else range(len(data)),
        "dialogue": data["dialogue"],
        "reference": data["summary"],
        "prediction": predictions,
    })
    output.to_json(path, orient="records", lines=True, force_ascii=False)
    logger.info("saved predictions artifact: %d rows -> %s", len(output), path)


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    save_json(metrics, path)
    logger.info("saved metrics artifact -> %s", path)


def save_config(config: dict[str, Any], path: Path) -> None:
    save_json(config, path)
    logger.debug("saved config artifact -> %s", path)