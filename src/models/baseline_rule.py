"""Rule-based summarization: first + last sentence of the dialogue."""
from __future__ import annotations

import re


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str, mode: str = "regex") -> list[str]:
    """Split a dialogue into sentences.

    mode='regex'  -> split on . ! ? … followed by whitespace
    mode='newline'-> split on line breaks (each turn as one "sentence")
    """
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


def first_last_sentence(
    dialogue: str,
    max_words: int | None = 50,
    sentence_splitter: str = "regex",
) -> str:
    """Build a summary as 'first sentence + last sentence' of the dialogue."""
    sentences = split_sentences(dialogue, mode=sentence_splitter)

    if not sentences:
        return ""
    if len(sentences) == 1:
        summary = sentences[0]
    else:
        summary = f"{sentences[0]} {sentences[-1]}"

    return truncate_words(summary, max_words)