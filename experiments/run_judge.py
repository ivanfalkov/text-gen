"""Entry point: LLM-as-judge evaluation of all experiments."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.metrics.llm_judge import evaluate_predictions
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

JUDGE_DIR = Path("artifacts/judge")
JUDGE_RESULTS_PATH = Path("artifacts/judge_results.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/judge.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(level=logging.INFO)

    cfg = load_config(args.config)
    limit = args.limit or cfg["data"]["limit"]

    JUDGE_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    for name, path in cfg["prediction_files"].items():
        path = Path(path)
        if not path.exists():
            logger.warning("skip %s: file not found at %s", name, path)
            continue

        df = pd.read_json(path, orient="records", lines=True)
        logger.info("judging '%s': %d rows (limit=%s)", name, len(df), limit)

        averages, raw_records = evaluate_predictions(
            df, cfg, model_name=name, limit=limit,
        )
        results[name] = averages
        logger.info("[%s] %s", name, averages)

        model_file = JUDGE_DIR / f"{name}.jsonl"
        with open(model_file, "w", encoding="utf-8") as f:
            for record in raw_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("saved raw judge records -> %s", model_file)

    JUDGE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("saved judge results -> %s", JUDGE_RESULTS_PATH)


if __name__ == "__main__":
    main()