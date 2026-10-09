"""Pipeline steps for experiments: data, predictions, metrics, config."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.data.loaders import load_split
from src.data.split import split_train_val
from src.utils import artifacts as artifact_utils
from src.utils import clearml_utils
from src.utils.artifacts import create_run_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def setup_experiment(config: dict[str, Any], config_path: Path) -> tuple[Any, Path]:
    experiment_name = config["experiment_name"]
    logger.info("starting experiment: %s", experiment_name)

    clearml_cfg = config.get("clearml", {})
    task = None
    if clearml_cfg.get("enabled", True):
        task = clearml_utils.init_task(
            project_name=clearml_cfg["project_name"],
            task_name=experiment_name,
            params=config,
        )

    artifacts_root = Path(config["paths"]["artifacts_dir"])
    run_dir = create_run_dir(artifacts_root, experiment_name)
    logger.info("run dir: %s", run_dir)

    artifact_utils.save_config(config, run_dir / "config.json")
    clearml_utils.upload_artifact(task, "config", run_dir / "config.json")
    return task, run_dir


def finish_experiment(task) -> None:
    if task is None:
        return
    logger.info("flushing ClearML task")
    task.flush(wait_for_uploads=True)
    logger.info("closing ClearML task")
    task.close()


def load_data(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load raw splits and produce {train, val, val_holdout}."""
    jsonl_dir = Path(config["paths"]["jsonl_dir"])
    splits_cfg = config["splits"]
    seed = config.get("seed", 42)

    raw_train = load_split(jsonl_dir, splits_cfg["train"])
    raw_val_holdout = load_split(jsonl_dir, splits_cfg["val_holdout"])
    logger.info("loaded raw train=%d, val_holdout=%d", len(raw_train), len(raw_val_holdout))

    train_df, val_df = split_train_val(
        raw_train,
        val_size=splits_cfg["val_size"],
        seed=seed,
    )

    return {
        "train": train_df,
        "val": val_df,
        "val_holdout": raw_val_holdout,
    }


def save_data_artifacts(data: dict[str, pd.DataFrame], run_dir: Path, task=None) -> None:
    for name, df in data.items():
        path = run_dir / f"data_{name}.jsonl"
        artifact_utils.save_split(df, path)
        clearml_utils.upload_artifact(task, f"data_{name}", path)
        clearml_utils.log_scalar(task, "data", f"n_rows_{name}", len(df))


def save_predictions_artifacts(
    data: dict[str, pd.DataFrame],
    predictions: dict[str, list[str]],
    run_dir: Path,
    task=None,
) -> None:
    for name, df in data.items():
        path = run_dir / f"predictions_{name}.jsonl"
        artifact_utils.save_predictions(df, predictions[name], path)
        clearml_utils.upload_artifact(task, f"predictions_{name}", path)


def evaluate_all(
    predictions: dict[str, list[str]],
    data: dict[str, pd.DataFrame],
    run_dir: Path,
    compute_metrics: Callable[[list[str], list[str]], dict[str, float]],
    task=None,
) -> dict[str, dict[str, float]]:
    all_metrics: dict[str, dict[str, float]] = {}
    for name, df in data.items():
        metrics = compute_metrics(predictions[name], df["summary"].tolist())
        all_metrics[name] = metrics
        artifact_utils.save_metrics(metrics, run_dir / f"metrics_{name}.json")
        clearml_utils.log_metrics(task, metrics, title=name)
        clearml_utils.upload_artifact(task, f"metrics_{name}", run_dir / f"metrics_{name}.json")
        logger.info("[%s] %s", name, metrics)
    return all_metrics