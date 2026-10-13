"""LoRA fine-tuning and vLLM inference with the trained adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _format_prompt(dialogue: str, prompt_template: str) -> str:
    return prompt_template.format(dialogue=dialogue)


def train(
    data: dict[str, pd.DataFrame],
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Train LoRA adapter on the train split, using val for early stopping.

    Returns path to the saved adapter directory.
    """
    model_name = config["model"]["name"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    prompt_template = config["prompt"]

    logger.info("loading base model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="bfloat16",
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def to_text(df: pd.DataFrame) -> list[dict]:
        return [
            {"text": _format_prompt(row["dialogue"], prompt_template) + row["summary"] + tokenizer.eos_token}
            for _, row in df.iterrows()
        ]

    train_ds = Dataset.from_list(to_text(data["train"]))
    val_ds = Dataset.from_list(to_text(data["val"]))

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        report_to=train_cfg["report_to"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        remove_unused_columns=train_cfg["remove_unused_columns"],
        dataset_text_field="text",
        max_seq_length=config["model"]["max_model_len"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    logger.info("starting LoRA training")
    trainer.train()
    logger.info("training finished")

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("adapter saved to %s", adapter_dir)

    return adapter_dir


def predict_with_lora(
    data: dict[str, pd.DataFrame],
    config: dict[str, Any],
    adapter_dir: Path,
) -> dict[str, list[str]]:
    """Run vLLM inference with the trained LoRA adapter on all splits."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_name = config["model"]["name"]
    gen_cfg = config["generation"]
    prompt_template = config["prompt"]

    logger.info("loading vLLM with LoRA adapter from %s", adapter_dir)
    llm = LLM(
        model=model_name,
        enable_lora=True,
        max_lora_rank=config["lora"]["r"],
        gpu_memory_utilization=0.7,
        max_model_len=config["model"]["max_model_len"],
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=gen_cfg["temperature"],
        max_tokens=gen_cfg["max_tokens"],
    )
    lora_request = LoRARequest("summarizer", 1, str(adapter_dir))

    predictions: dict[str, list[str]] = {}
    for name, df in data.items():
        prompts = [_format_prompt(d, prompt_template) for d in df["dialogue"].tolist()]
        logger.info("generating with LoRA for split '%s' (%d rows)", name, len(prompts))
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        predictions[name] = [o.outputs[0].text.strip() for o in outputs]
        logger.info("predicted split '%s': %d rows", name, len(predictions[name]))

    return predictions