"""Entry point: rule-based baseline (first + last sentence) on train/val/test."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.experiments.runner import (
    evaluate_all,
    finish_experiment,
    load_data,
    save_data_artifacts,
    save_predictions_artifacts,
    setup_experiment,
)
from src.metrics.compute_metrics import compute_all
from src.models import baseline_rule
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/general_config.yaml"),
        help="General config (paths, clearml, splits).",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/baseline_rule.yaml"),
        help="Experiment-specific config.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    cfg = {**load_config(args.config), **load_config(args.experiment_config)}

    task, artifacts_dir = setup_experiment(cfg, args.experiment_config)

    data = load_data(cfg)
    save_data_artifacts(data, artifacts_dir, task)

    predictions = baseline_rule.predict(data, cfg)
    save_predictions_artifacts(data, predictions, artifacts_dir, task)

    evaluate_all(
        predictions=predictions,
        data=data,
        run_dir=artifacts_dir,
        compute_metrics=compute_all,
        task=task,
    )

    finish_experiment(task)


if __name__ == "__main__":
    main()