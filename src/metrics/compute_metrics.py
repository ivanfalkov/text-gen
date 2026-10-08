"""Compute ROUGE and BLEU for summarization predictions."""
from __future__ import annotations

from typing import Iterable

import evaluate

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_rouge = None
_bleu = None


def _get_rouge():
    global _rouge
    if _rouge is None:
        logger.info("loading ROUGE metric")
        _rouge = evaluate.load("rouge")
    return _rouge


def _get_bleu():
    global _bleu
    if _bleu is None:
        logger.info("loading BLEU metric")
        _bleu = evaluate.load("bleu")
    return _bleu


def compute_all(
    predictions: Iterable[str],
    references: Iterable[str],
) -> dict[str, float]:
    """Compute ROUGE-1/2/L and BLEU. Returns a flat dict of floats."""
    predictions = [p if p else "" for p in predictions]
    references = [r if r else "" for r in references]

    logger.info("computing ROUGE and BLEU on %d examples", len(predictions))

    rouge = _get_rouge().compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    bleu = _get_bleu().compute(
        predictions=predictions,
        references=[[r] for r in references],
    )

    return {
        "rouge1": float(rouge["rouge1"]),
        "rouge2": float(rouge["rouge2"]),
        "rougeL": float(rouge["rougeL"]),
        "bleu": float(bleu["bleu"]),
    }