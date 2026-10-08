"""Pipeline steps with artifact saving + ClearML logging."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.utils import artifacts as artifact_utils
from src.utils import clearml_utils
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_yaml(path: Path) -> dict[str, Any]:
    logger.debug("loading yaml: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_dataset(
    data: pd.DataFrame,
    run_dir: Path,
    task=None,
    split: str | None = None,
) -> Path:
    """Save dataset locally + upload to ClearML."""
    path = run_dir / "dataset.jsonl"
    artifact_utils.save_dataset(data, path)
    clearml_utils.upload_artifact(task, "dataset", path)
    clearml_utils.log_scalar(task, "data", "n_rows", len(data))
    if split is not None:
        clearml_utils.set_description(
            task,
            f"baseline experiment (rule-based)\n"
            f"no training, no loss by design\n"
            f"split={split}\n"
            f"n_examples={len(data)}",
        )
    return path


def save_predictions(
    data: pd.DataFrame,
    predictions: list[str],
    run_dir: Path,
    task=None,
) -> Path:
    path = run_dir / "predictions.jsonl"
    artifact_utils.save_predictions(data, predictions, path)
    clearml_utils.upload_artifact(task, "predictions", path)
    return path


def evaluate_and_save_metrics(
    predictions: list[str],
    references: list[str],
    run_dir: Path,
    compute_metrics: Callable[[list[str], list[str]], dict[str, float]],
    task=None,
    split: str = "eval",
) -> dict[str, float]:
    metrics = compute_metrics(predictions, references)
    path = run_dir / "metrics.json"
    artifact_utils.save_metrics(metrics, path)
    clearml_utils.log_metrics(task, metrics, title=f"metrics/{split}")
    clearml_utils.upload_artifact(task, "metrics", path)
    return metrics


def save_config(config: dict[str, Any], run_dir: Path, task=None) -> Path:
    path = run_dir / "config.json"
    artifact_utils.save_config(config, path)
    clearml_utils.upload_artifact(task, "config", path)
    return path