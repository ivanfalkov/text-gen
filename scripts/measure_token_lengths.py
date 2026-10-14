from transformers import AutoTokenizer
import json
from pathlib import Path

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

PROMPT_TEMPLATE = (
    "Summarize the following dialogue in 1-2 sentences. Be concise and capture the main points.\n\n"
    "Dialogue:\n{dialogue}\n\nSummary:"
)

def max_tokens(jsonl_path: Path) -> tuple[int, int]:
    max_len = 0
    p95_len = 0
    lengths = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = PROMPT_TEMPLATE.format(dialogue=row["dialogue"])
            n = len(tokenizer(text, truncation=False)["input_ids"])
            lengths.append(n)
    lengths.sort()
    return lengths[-1], lengths[int(len(lengths) * 0.95)]

for split in ["train", "validation", "test"]:
    path = Path(f"data/raw/jsonl/{split}.jsonl")
    if path.exists():
        mx, p95 = max_tokens(path)
        print(f"{split}: max={mx}, p95={p95}")