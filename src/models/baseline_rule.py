"""Rule-based summarization: first + last sentence of the dialogue."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, mode: str = "regex") -> list[str]:
    text = text.strip()
    if not text:
        return []

    if mode == "newline":
        parts = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]

    return parts


def truncate_words(text: str, max_words: int | None) -> str:
    if max_words is None:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def summarize(dialogue: str, max_words: int | None = 50, sentence_splitter: str = "regex") -> str:
    sentences = split_sentences(dialogue, mode=sentence_splitter)

    if not sentences:
        return ""
    if len(sentences) == 1:
        summary = sentences[0]
    else:
        summary = f"{sentences[0]} {sentences[-1]}"

    return truncate_words(summary, max_words)


def predict(data: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, list[str]]:
    """Generate summaries for every split in `data`."""
    model_cfg = config["model"]
    max_words = model_cfg["max_words"]
    sentence_splitter = model_cfg["sentence_splitter"]

    predictions: dict[str, list[str]] = {}
    for name, df in data.items():
        predictions[name] = [
            summarize(dialogue, max_words=max_words, sentence_splitter=sentence_splitter)
            for dialogue in df["dialogue"].tolist()
        ]
        logger.info("predicted split '%s': %d rows", name, len(predictions[name]))

    return predictions