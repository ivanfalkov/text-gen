"""Entry point: LLM-as-judge evaluation of all experiments."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.metrics.llm_judge import (
    DailyRateLimitExceeded,
    Judge,
    log_record,
    make_record,
    summarize,
)
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


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save(
    records_by_model: dict[str, list[dict[str, Any]]],
    metric_names: list[str],
) -> None:
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    for name, records in records_by_model.items():
        _write_jsonl(JUDGE_DIR / f"{name}.jsonl", records)
    results = {name: summarize(recs, metric_names) for name, recs in records_by_model.items()}
    JUDGE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(level=logging.INFO)

    cfg = load_config(args.config)
    limit = args.limit or cfg["data"]["limit"]

    frames: list[tuple[str, pd.DataFrame]] = []
    for name, path in cfg["prediction_files"].items():
        path = Path(path)
        if not path.exists():
            logger.warning("skip %s: file not found at %s", name, path)
            continue
        df = pd.read_json(path, orient="records", lines=True).head(limit)
        logger.info("loaded '%s': %d rows (limit=%s)", name, len(df), limit)
        frames.append((name, df))

    if not frames:
        logger.warning("no prediction files to judge")
        return

    n = min(len(df) for _, df in frames)
    records_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in frames}
    judge = Judge(cfg)
    metrics = judge.metric_names

    logger.info("judging %d ids across %d models (round-robin)", n, len(frames))
    stopped = False
    try:
        for i in range(n):
            batch: dict[str, dict[str, Any]] = {}
            for name, df in frames:
                row = df.iloc[i]
                result = judge.score(row["dialogue"], row["prediction"])
                record = make_record(row, name, i, result, metrics)
                log_record(record, i + 1, n, metrics, judge.preview_chars)
                batch[name] = record
                time.sleep(judge.sleep_between)
            for name, record in batch.items():
                records_by_model[name].append(record)
            _save(records_by_model, metrics)
    except DailyRateLimitExceeded as e:
        stopped = True
        n_done = len(next(iter(records_by_model.values())))
        logger.warning("stopped: daily limit, n=%d (%s)", n_done, e)

    _save(records_by_model, metrics)
    n_done = len(next(iter(records_by_model.values())))
    for name, recs in records_by_model.items():
        logger.info("[%s] %s", name, summarize(recs, metrics))
    logger.info(
        "saved judge results -> %s (n=%d%s)",
        JUDGE_RESULTS_PATH,
        n_done,
        ", daily limit" if stopped else "",
    )


if __name__ == "__main__":
    main()
