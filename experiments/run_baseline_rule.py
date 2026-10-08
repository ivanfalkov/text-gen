"""Entry point: rule-based baseline (first + last sentence)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.loaders import load_split
from src.experiments.runner import (
    evaluate_and_save_metrics,
    load_yaml,
    save_config,
    save_dataset,
    save_predictions,
)
from src.metrics.compute_metrics import compute_all
from src.models.baseline_rule import first_last_sentence
from src.utils import clearml_utils
from src.utils.artifacts import create_run_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/baseline_rule.yaml"),
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/general_config.yaml"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: use only N rows.")
    parser.add_argument("--split", type=str, default=None, help="Override evaluation split.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging for src.*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=logging.INFO)
    if args.verbose:
        logging.getLogger("src").setLevel(logging.DEBUG)

    experiment_config = load_yaml(args.experiment_config)
    project_config = load_yaml(args.project_config)
    experiment_name = experiment_config["experiment_name"]

    logger.info("starting experiment: %s", experiment_name)

    # ClearML — с самого начала
    task = None
    if (
        experiment_config.get("clearml", {}).get("enabled", True)
        and project_config.get("clearml", {}).get("enabled", False)
    ):
        task = clearml_utils.init_task(
            project_name=project_config["clearml"]["project_name"],
            task_name=experiment_name,
            params={
                "experiment": experiment_config,
                "project": project_config,
                "cli": {"limit": args.limit, "split": args.split},
            },
        )

    # run_dir сразу — все артефакты туда
    root = Path(project_config["paths"]["artifacts_dir"])
    run_dir = create_run_dir(root, experiment_name)
    logger.info("run dir: %s", run_dir)

    # config как первый артефакт
    save_config(
        {
            "experiment": experiment_config,
            "project": project_config,
            "cli": {"limit": args.limit, "split": args.split},
        },
        run_dir=run_dir,
        task=task,
    )

    # data
    split = args.split or project_config["evaluation"]["split"]
    jsonl_dir = Path(project_config["paths"]["jsonl_dir"])
    logger.info("using split='%s'", split)

    data = load_split(jsonl_dir, split)
    if args.limit:
        data = data.head(args.limit).copy()
        logger.info("limit applied: %d rows", len(data))
    logger.info("loaded data: %d rows", len(data))

    save_dataset(data, run_dir=run_dir, task=task, split=split)

    # predictions
    model_config = experiment_config["model"]
    logger.info("generating predictions with model=%s", model_config["name"])
    predictions = [
        first_last_sentence(
            dialogue,
            max_words=model_config["max_words"],
            sentence_splitter=model_config["sentence_splitter"],
        )
        for dialogue in data["dialogue"].tolist()
    ]
    logger.info("generated %d predictions", len(predictions))

    save_predictions(data, predictions, run_dir=run_dir, task=task)

    # metrics
    metrics = evaluate_and_save_metrics(
        predictions=predictions,
        references=data["summary"].tolist(),
        run_dir=run_dir,
        compute_metrics=compute_all,
        task=task,
        split=split,
    )
    logger.info("metrics: %s", metrics)

    logger.info("done. run_dir=%s", run_dir)
    if task is not None:
        task.close()


if __name__ == "__main__":
    main()