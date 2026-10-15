"""LLM-as-judge evaluation via Groq."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DIALOGUE_PREVIEW_CHARS = 200


JUDGE_PROMPT = """You are evaluating a dialogue summary.

Given the dialogue and a candidate summary, rate the candidate on a scale of 1-5 for each criterion:

- faithfulness: does the summary only contain facts that are present in the dialogue? (1 = many hallucinations, 5 = fully grounded in the dialogue)
- completeness: does the summary capture the key points of the dialogue? (1 = misses most key points, 5 = captures all key points)
- conciseness: is the summary appropriately brief without losing important information? (1 = too verbose or too short, 5 = well-balanced)

Dialogue:
{dialogue}

Candidate summary:
{summary}

IMPORTANT: Respond with ONLY a valid JSON object. Do NOT include any reasoning, explanations, or markdown formatting. Your entire response must be parseable as JSON.

{{"faithfulness": <int 1-5>, "completeness": <int 1-5>, "conciseness": <int 1-5>, "reasoning": "<short explanation>"}}"""


def _truncate(text: str, limit: int = DIALOGUE_PREVIEW_CHARS) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


class Judge:
    def __init__(self, config: dict[str, Any]):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            max_retries=0,  # свои retry ниже
        )
        self.model = config["judge"]["model"]
        self.max_retries = config["judge"].get("max_retries", 3)
        self.sleep_between = config["judge"].get("sleep_between", 4.0)

    def _call_api(self, dialogue: str, summary: str) -> dict:
        prompt = JUDGE_PROMPT.format(dialogue=dialogue, summary=summary)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_scores",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
                            "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
                            "conciseness": {"type": "integer", "minimum": 1, "maximum": 5},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["faithfulness", "completeness", "conciseness", "reasoning"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def score(self, dialogue: str, summary: str) -> dict:
        for attempt in range(self.max_retries):
            try:
                return self._call_api(dialogue, summary)
            except Exception as e:
                logger.warning(
                    "judge call failed (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.sleep_between * (2 ** attempt))
        return {
            "faithfulness": None,
            "completeness": None,
            "conciseness": None,
            "reasoning": "failed",
        }


def evaluate_predictions(
    predictions_df,
    config: dict[str, Any],
    model_name: str,
    limit: int | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Run judge on each row. Returns (averages, raw_records)."""
    judge = Judge(config)

    if limit:
        predictions_df = predictions_df.head(limit)
        logger.info("judge limit applied: %d rows", len(predictions_df))

    metric_names = ("faithfulness", "completeness", "conciseness")
    scores: dict[str, list[int]] = {name: [] for name in metric_names}
    raw_records: list[dict] = []
    failed = 0

    total = len(predictions_df)
    for i, (_, row) in enumerate(predictions_df.iterrows(), start=1):
        result = judge.score(row["dialogue"], row["prediction"])
        ok = result.get("faithfulness") is not None
        if ok:
            for name in metric_names:
                if result.get(name) is not None:
                    scores[name].append(result[name])
        else:
            failed += 1

        record = {
            "id": row.get("id", i),
            "model": model_name,
            "dialogue": row["dialogue"],
            "reference": row["reference"],
            "prediction": row["prediction"],
            "faithfulness": result.get("faithfulness"),
            "completeness": result.get("completeness"),
            "conciseness": result.get("conciseness"),
            "reasoning": result.get("reasoning", ""),
        }
        raw_records.append(record)

        logger.info(
            "\n─── [%s] row %d/%d ───\nDIALOGUE:   %s\nREFERENCE:  %s\nPREDICTION: %s\nSCORES:     faithfulness=%s completeness=%s conciseness=%s\nREASONING:  %s",
            model_name, i, total,
            _truncate(row["dialogue"]),
            row["reference"].strip(),
            row["prediction"].strip(),
            result.get("faithfulness"),
            result.get("completeness"),
            result.get("conciseness"),
            (result.get("reasoning") or "").strip(),
        )

        time.sleep(judge.sleep_between)

    def _mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    averages = {
        "judge_faithfulness": _mean(scores["faithfulness"]),
        "judge_completeness": _mean(scores["completeness"]),
        "judge_conciseness": _mean(scores["conciseness"]),
        "judge_failed": failed,
    }
    return averages, raw_records