# project_notes.md

Внутренние заметки о проекте для быстрого погружения другого агента/разработчика. README — для пользователя, здесь — «что под капотом, где грабли, что помнить».

## О задаче

Суммаризация диалогов на датасете **DialogSum** (Kaggle, ~13k диалогов с референсными саммари). Цель — показать на практике, что LoRA-дообучение небольшой LLM даёт измеримый прирост качества относительно бейзлайнов.

Сравниваются три метода:
1. `baseline_rule` — эвристика «первое + последнее предложение».
2. `baseline_llm` — Qwen2.5-0.5B-Instruct без дообучения через vLLM.
3. `lora` — та же модель + LoRA-адаптер.

Плюс оценка через LLM-as-judge (`openai/gpt-oss-120b` на Groq).

## Окружение и ограничения

**ОС.** Работа идёт **в WSL2 (Ubuntu)** — не в Windows. Причина: **vLLM официально не поддерживает Windows**. `.venv` из Windows несовместим с Linux, поэтому используется `.venv_linux`.

**VRAM.** NVIDIA RTX 4060 Laptop, **8 GB**. Это накладывает жёсткие ограничения:
- Модель < 1B (по ТЗ и по памяти).
- LoRA вместо полного fine-tuning.
- При инференсе через vLLM — `gpu_memory_utilization=0.85`, `kv_cache_dtype="fp8"`, `max_num_seqs=8–16`, иначе падает с «No available memory for the cache blocks».

**CUDA.** `torch 2.6.0+cu124`, `vllm 0.8.5.post1`. Более новая vLLM (0.31+) требует torch 2.13, который на момент работы не опубликован — откатились до 0.8.5.

**Компилятор.** Triton JIT-компилирует CUDA-ядра при первом запуске vLLM — нужен `build-essential` (gcc). Без него — `Failed to find C compiler`.

**Системные библиотеки.** PyTorch 2.6+cu124 требует `libcudnn.so.9` и `libnccl.so.2`. В чистой WSL их нет — ставятся через репозиторий NVIDIA (`cuda-keyring`).

## Критичные пины версий

Эти версии не совместимы «по-другому» — проверено боем.

| Пакет | Версия | Почему так |
|---|---|---|
| `torch` | `==2.6.0` | vLLM 0.8.5 собран именно под неё |
| `vllm` | `==0.8.5.post1` | Более новые требуют torch 2.13 (нет в PyPI) |
| `transformers` | `>=4.51.1,<4.52` | vLLM 0.8.5 не работает с transformers 5.x (`all_special_tokens_extended`) |
| `numpy` | `>=1.26,<2.3` | numba (зависимость vLLM) требует `<2.3` |
| `pandas` | `>=2.2,<3` | pandas 3.x тянет numpy ≥2.3.3 на Python 3.14, конфликт |
| `pyarrow` | `<17` | старый `datasets` падает на `PyExtensionType` |
| `tokenizers` | `>=0.21,<0.22` | transformers 4.51 требует именно эту ветку |
| `datasets` | `>=3.0,<4` | совместим с transformers 4.51 и pyarrow <17 |
| `evaluate` | `>=0.4,<0.5` | та же причина |
| `peft` | `>=0.13,<0.14` | совместим с transformers 4.51 |
| `trl` | `>=0.12,<0.13` | совместим с peft 0.13 |

`requires-python = ">=3.12,<3.13"` — сужение, чтобы uv не пытался разрешать зависимости для 3.13/3.14 (там свои конфликты).

## Структура
configs/
general_config.yaml # пути, сплиты, clearml
experiments/
baseline_rule.yaml
baseline_llm.yaml
lora.yaml
judge.yaml
data/raw/jsonl/ # dialogsum.train.jsonl, validation.jsonl, test.jsonl
scripts/download_data.py # скачивание с Kaggle
experiments/
run_baseline_rule.py
run_baseline_llm.py
run_lora.py
run_judge.py
src/
data/loaders.py # load_split
data/split.py # split_train_val (отрезает val от train)
models/baseline_rule.py # эвристика + predict()
models/llm_inference.py # vLLM без адаптера
models/lora_finetune.py # train() + predict_with_lora()
metrics/compute_metrics.py # ROUGE + BLEU через evaluate
metrics/llm_judge.py # Groq + strict json_schema
experiments/runner.py # setup_experiment, load_data, save_*, evaluate_all
utils/config.py # load_config
utils/clearml_utils.py # init_task, log_metrics, upload_artifact, finish_experiment
utils/artifacts.py # create_run_dir, save_split, save_predictions, save_metrics
utils/logging_utils.py # setup_logging, get_logger
artifacts/ # результаты экспериментов

text

## Сплиты

Оригинальные сплиты DialogSum:
- `train.jsonl` — 12 460
- `validation.jsonl` — 500
- `test.jsonl` — 500, **но без `summary`** (не размечен). **Не используем.**

В пайплайне формируются три рабочих сплита (`load_data` в `runner.py`):
- `train` — 11 960 (из train.jsonl минус 500)
- `val` — 500 (отрезано от train, random seed=42, для in-training eval)
- `val_holdout` — 500 (официальная validation, финальная оценка)

## Данные и особенности

**Формат JSONL.** `dialogue` (многострочный, реплики `#Person1#`, `#Person2#`), `summary` (1–2 предложения).

**Дубликаты:**
- Одинаковый `dialogue`, разные `summary` — multi-reference, оставляем.
- Одинаковый `(dialogue, summary)` — полный дубликат, удаляем.
- Одинаковый `summary`, разные `dialogue` — оставляем.

**Длина токенов.** Промпт + диалог ≈ 400–800 токенов, редкие до 1500. `max_model_len=2048` в конфиге хватает с запасом.

## Модели

**Qwen2.5-0.5B-Instruct** — основная. 24 слоя, 14 attention heads, 2 KV heads (GQA), hidden 896. LoRA-параметры: 2 162 688 / 496 195 456 = **0.44%**.

LoRA-конфиг: `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=[q_proj, k_proj, v_proj, o_proj]`, `bias=none`. Обучение: 3 эпохи, `per_device_train_batch_size=4`, `grad_accum=4`, `lr=2e-4` (в YAML писать `0.0002`, не `2e-4` — парсер может отдать строку), `optim=adamw_8bit`, `gradient_checkpointing=true`.

**Обязательные фиксы в `lora_finetune.py`:**
- `model.enable_input_require_grads()` после `get_peft_model` — иначе `RuntimeError: element 0 of tensors does not require grad` при gradient checkpointing.
- `torch_dtype="bfloat16"` (не `dtype=`) — transformers 4.51 ещё не знает нового имени.
- `metric_for_best_model="eval_loss"`, `greater_is_better=False`, `load_best_model_at_end=True` — если `eval_rouge1` не считается, Trainer падает с `KeyError`.

**Модель-судья:** `openai/gpt-oss-120b` через Groq. `strict: true` + `additionalProperties: false` в `json_schema` — без этого reasoning-модель ломает JSON. `max_tokens=2048` — reasoning ест много токенов.

## Метрики

**ROUGE/BLEU** через `evaluate`. ROUGE — с `use_stemmer=True`, BLEU — `references=[[r] for r in refs]` (вложенный список).

**LLM-as-judge:** три метрики 1–5 — `faithfulness`, `completeness`, `conciseness`. Средние по модели сохраняются в `artifacts/judge_results.json`, сырые оценки по каждому примеру — в `artifacts/judge/<model>.jsonl`.

## Результаты (val_holdout, 500 примеров)

| Модель | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|
| baseline_rule | 0.2297 | 0.0533 | 0.1957 | 0.0696 |
| baseline_llm | 0.2922 | 0.0954 | 0.2208 | 0.0787 |
| **lora** | **0.4591** | **0.2043** | **0.3713** | **0.2294** |

LLM-as-judge (45 примеров на модель):

| Модель | Faithfulness | Completeness | Conciseness |
|---|---|---|---|
| baseline_rule | 3.69 | 1.11 | 1.91 |
| baseline_llm | 2.07 | 1.93 | 2.42 |
| **lora** | 3.20 | **2.02** | **3.20** |

LoRA > baseline_llm > baseline_rule по обеим системам оценки. baseline_llm страдает галлюцинациями (faithfulness 2.07) — ROUGE это не видит, судья ловит.

## ClearML

Креды в `~/clearml.conf` (копируется из Windows: `cp /mnt/c/Users/Ivan/clearml.conf ~/clearml.conf`).

Таск создаётся **в самом начале** (`setup_experiment`), артефакты логируются по ходу: `config` → `dataset` → `predictions` → `metrics`. Метрики под секциями `metrics/<split>` (`metrics/train`, `metrics/val`, `metrics/val_holdout`).

**Грабли:** `finish_experiment` вызывает `task.flush(wait_for_uploads=True)` — может зависнуть на 15+ минут, если сеть медленная. Нужен таймаут через `signal.alarm` или вообще убрать `wait_for_uploads` (артефакты всё равно локально лежат).

## Запуск

```bash
cd /mnt/c/Users/Ivan/Documents/projects/text-gen
source .venv_linux/bin/activate

uv run python experiments/run_baseline_rule.py
uv run python experiments/run_baseline_llm.py
uv run python experiments/run_lora.py
uv run python experiments/run_judge.py --limit 100
Флаг --limit N — для отладки на N примерах.

Грабли и решения (кратко)
Симптом	Причина	Решение
No solution found ... torch==2.13.0	vLLM 0.31 требует torch 2.13 (нет в PyPI)	Откат на vllm 0.8.5.post1
libcudnn.so.9: cannot open shared object file	cuDNN не установлен	cuda-keyring + libcudnn9-cuda-12
libnccl.so.2	NCCL не установлен	libnccl2 libnccl-dev
Failed to find C compiler	нет gcc в WSL	build-essential
No available memory for the cache blocks	мало VRAM под KV-cache	gpu_memory_utilization=0.85, kv_cache_dtype="fp8", max_num_seqs=8
all_special_tokens_extended	transformers 5.x с vllm 0.8.5	transformers<4.52
PyExtensionType	pyarrow ≥17 с старым datasets	pyarrow<17
dtype unexpected argument	transformers 4.51 не знает dtype	torch_dtype="bfloat16"
element 0 of tensors does not require grad	gradient checkpointing + LoRA	model.enable_input_require_grads()
KeyError: eval_rouge1	metric_for_best_model без compute_metrics	metric_for_best_model="eval_loss"
404 model_not_found на Groq	модель удалена	openai/gpt-oss-120b
json_validate_failed на Groq	нет strict schema	json_schema + strict: true + additionalProperties: false
429 на Groq	TPD 200 000 для gpt-oss-120b	--limit 50 или Cerebras
finish_experiment висит 15 мин	flush(wait_for_uploads=True) на медленной сети	таймаут или убрать параметр
Что осталось / идеи
Улучшения: r=32 или r=64, добавить MLP-слои в target_modules, 5 эпох, снять ограничение <1B и взять Qwen2.5-1.5B/3B.

LLM-as-judge прогнан только на 45–100 примерах (лимиты Groq TPD). Для полного прогона на 500 — либо Cerebras (1M токенов/день), либо несколько дней.

test.jsonl не используется — там нет summary. Если ТЗ потребует, искать альтернативный источник.

remove_unused_columns — стоит false из-за особенностей SFTTrainer с dataset_text_field. Предупреждение в логах, не ошибка.