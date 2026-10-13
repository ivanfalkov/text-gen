"""LLM inference via vLLM for summarization."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Summarizer:
    """Thin wrapper around vLLM for dialogue summarization."""

    def __init__(self, config: dict[str, Any]):
        from vllm import LLM, SamplingParams

        model_cfg = config["model"]
        gen_cfg = config["generation"]

        logger.info("loading model %s", model_cfg["name"])
        self.llm = LLM(
            model=model_cfg["name"],
            gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
            max_model_len=model_cfg["max_model_len"],
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=gen_cfg["temperature"],
            max_tokens=gen_cfg["max_tokens"],
        )
        self.prompt_template = config["prompt"]

    def predict(self, data: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, list[str]]:
        predictions: dict[str, list[str]] = {}
        for name, df in data.items():
            prompts = [self.prompt_template.format(dialogue=d) for d in df["dialogue"].tolist()]
            logger.info("generating summaries for split '%s' (%d rows)", name, len(prompts))
            outputs = self.llm.generate(prompts, self.sampling_params)
            predictions[name] = [o.outputs[0].text.strip() for o in outputs]
            logger.info("predicted split '%s': %d rows", name, len(predictions[name]))
        return predictions


def predict(data: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, list[str]]:
    """Entry point used by experiment scripts."""
    summarizer = Summarizer(config)
    return summarizer.predict(data, config)