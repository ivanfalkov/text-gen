# Dialog Summarization - генерация текста
## 1. О задаче.
В этой работе решается задача абстрактивной суммаризации диалогов на датасете [DialogSum](https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum/data) (~13k диалогов с референсными краткими саммари, сплиты train/val/test, формат JSONL с полями `dialogue` и `summary`).

Цель — показать полный цикл работы с генеративной LLM под конкретную задачу и измерить, насколько LoRA-дообучение улучшает качество относительно бейзлайнов.

Последовательно строятся:
- бейзлайн без ML — эвристика «первое + последнее предложение»;
- бейзлайн LLM без дообучения;
- LoRA-дообучение той же модели;
- оценка через LLM-as-judge.

Качество измеряется метриками ROUGE-1/2/L и BLEU, а также ручным просмотром примеров.
## 2. Данные и выводы по EDA
**Источник:** [DialogSum на Kaggle](https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum/data) — датасет для абстрактивной суммаризации диалогов, ~13k пар «диалог → краткое саммари», разделённых на train (~12.5k), val (~0.5k) и test (0.1k)

**Вывод EDA**
1. Пропусков в train, val, test нет   
2. Средняя длина диалога и саммари 131 и 28 соответвенно  
3. Есть одинаковые диалоги 2 штуки, но это нормально, так как саммари разное. Возможно разные разметчики сделали разное самари у одинаковых диалогов.  
4. Есть одинаковые саммари 24 шт, но диалоги у них разные. Это тоже нормально.  

## 3. Выводы экспериментов
### Обоснование выбора моделей
LLM для саммаризации - Qwen2.5-0.5B-Instruct — влезает в ограничение <1B и в 8 GB VRAM, instruct-версия следует промпту, официально поддерживается vLLM 0.8.5 (стабильная версия для WSL). Альтернативы хуже: TinyLlama устарел, Llama-3.2-1B вдвое тяжелее, OPT-125M слишком слаб для осмысленных саммари.

LLM as judge - openai/gpt-oss-120b — единственная бесплатная модель на Groq с поддержкой strict JSON Schema и качеством, достаточным для оценки суммаризации; OpenAI-совместимый API позволил обойтись без отдельного SDK. Llama-3.1/3.3 на Groq были удалены, Gemini требует другого SDK, локальные модели не тянут роль судьи на 8 GB VRAM.

vLLM вместо transformers — в 5–10 раз выше throughput за счёт PagedAttention и continuous batching, плюс из коробки умеет загружать PEFT-адаптеры через LoRARequest.

### Автоматические метрики (val_holdout, 500 примеров DialogSum)

| Модель | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|
| baseline_rule | 0.2297 | 0.0533 | 0.1957 | 0.0696 |
| baseline_llm (Qwen2.5-0.5B, zero-shot) | 0.2922 | 0.0954 | 0.2208 | 0.0787 |
| **lora (Qwen2.5-0.5B + LoRA, 3 эпохи)** | **0.4591** | **0.2043** | **0.3713** | **0.2294** |
### LLM-as-judge (45 примеров из val_holdout, openai/gpt-oss-120b через Groq)

| Модель | Faithfulness | Completeness | Conciseness |
|---|---|---|---|
| baseline_rule | **3.69** | 1.11 | 1.91 |
| baseline_llm | 2.07 | 1.93 | 2.42 |
| **lora** | 3.20 | **2.02** | **3.20** |

### Выводы
- LoRA — лучшая среди LLM по всем трём критериям судьи.
- Zero-shot LLM сильно галлюцинирует (faithfulness 2.07) — LoRA это лечит (3.20).
- baseline_rule выигрывает по faithfulness за счёт копирования, но проваливается по полноте и связности.
- ROUGE/BLEU и LLM-as-judge независимо подтверждают: LoRA > baseline_llm > baseline_rule.
- baseline_rule — эвристика на CPU, отвечает за миллисекунды без GPU. Среди LLM быстрее baseline_llm , но LoRA отстаёт лишь на секунды при 500 примерах и даёт +16.7 п.п. ROUGE-1, так что по соотношению качество/скорость выигрывает именно она.

**Как ещё можно улучшить результаты**
Дешёвые шаги — увеличить LoRA rank до 32–64 и добавить MLP-слои в target_modules, поднять число эпох до 5, расширить max_tokens при генерации. Более дорогой, но самый перспективный путь — снять ограничение <1B и взять Qwen2.5-1.5B или 3B с той же LoRA-схемой.


# 4. Инфраструктура и установка

## 4.1. WSL

Открыть PowerShell от имени администратора, выполнить `wsl --install`, перезагрузить компьютер. Открыть Ubuntu через меню Пуск, при первом запуске задать логин и пароль Linux (не связан с Windows-аккаунтом). Проверить, что WSL видит GPU: `nvidia-smi` — должна показаться RTX 4060 Laptop. Если пусто — обновить драйвер NVIDIA для Windows и перезапустить WSL: `wsl --shutdown` из PowerShell, потом заново открыть Ubuntu.

## 4.2. Установка uv в WSL

Установить uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`. Добавить в PATH: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc`. Установить Python 3.12: `uv python install 3.12`. Проверить: `uv --version`.

## 4.3. Системные библиотеки NVIDIA

Подключить репозиторий NVIDIA и установить cuDNN и NCCL:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
sudo apt install -y libnccl2 libnccl-dev
sudo ldconfig
```

Также поставить системный компилятор (нужен Triton для JIT-компиляции CUDA-ядер):

```bash
sudo apt install -y build-essential
```

## 4.4. Настройка окружения под Linux

Задать отдельный каталог для venv, чтобы не конфликтовал с Windows: `echo 'export UV_PROJECT_ENVIRONMENT=".venv_linux"' >> ~/.bashrc && source ~/.bashrc`. Перейти в проект: `cd /mnt/c/Users/Ivan/Documents/projects/text-gen` (здесь `/mnt/c/...` — доступ к диску `C:\` из Linux). Синхронизировать зависимости: `uv sync` — uv создаст `.venv_linux` и установит всё из `pyproject.toml`. Установить PyTorch с CUDA: `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`. Проверить GPU: `uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"` — ожидается `2.5.x+cu124`, `True`, `NVIDIA GeForce RTX 4060 Laptop GPU`. Установить vLLM: `uv pip install vllm`. Проверить импорт: `uv run python -c "import vllm; print(vllm.__version__)"`.

## 4.5. Переменные окружения

Создать `.env` в корне проекта:

```
KAGGLE_API_TOKEN=your_kaggle_token
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- `KAGGLE_API_TOKEN`— креды Kaggle (получить: https://www.kaggle.com/settings → API → Create New Token).
- `GROQ_API_KEY` — ключ Groq для LLM-as-judge (получить: https://console.groq.com → API Keys → Create API Key).

ClearML работает через `~/clearml.conf` (копируется из Windows: `cp /mnt/c/Users/Ivan/clearml.conf ~/clearml.conf`).

## 4.6. Загрузка данных

```bash
# Один раз: скачать датасет DialogSum с Kaggle
uv run python scripts/download_data.py
```

# 5. Запуск экспериментов

## 5.1. Rule-based baseline

Суммаризация эвристикой «первое + последнее предложение»:

```bash
uv run python experiments/run_baseline_rule.py
```

## 5.2. LLM baseline без дообучения

Qwen2.5-0.5B-Instruct через vLLM:

```bash
uv run python experiments/run_baseline_llm.py
```

## 5.3. LoRA дообучение

Обучение LoRA-адаптера + инференс через vLLM:

```bash
uv run python experiments/run_lora.py
```

## 5.4. LLM-as-judge

Оценка всех трёх моделей через Groq:

```bash
uv run python experiments/run_judge.py --limit 100
```

## 5.5. Артефакты

Каждый эксперимент сохраняет результаты в `artifacts/<exp_name>/<timestamp>/`:

- `config.json` — конфиг запуска.
- `data_{train,val,val_holdout}.jsonl` — данные, на которых считался эксперимент.
- `predictions_{train,val,val_holdout}.jsonl` — предсказания (диалог, reference, prediction).
- `metrics_{train,val,val_holdout}.json` — метрики по сплитам.
- `adapter/` — LoRA-адаптер (только для `lora`).

Для judge дополнительно:

- `artifacts/judge/<model>.jsonl` — сырые оценки судьи по каждому примеру.
- `artifacts/judge_results.json` — средние оценки по моделям.

# 6. Структура проекта

```
text-gen/
├── configs/
│   ├── general_config.yaml       # пути, сплиты, clearml
│   └── experiments/              # по одному YAML на эксперимент
│       ├── baseline_rule.yaml
│       ├── baseline_llm.yaml
│       ├── lora.yaml
│       └── judge.yaml
├── data/                         # raw, processed
├── scripts/                      # download_data, measure_token_lengths
├── experiments/                  # end-to-end раннеры
│   ├── run_baseline_rule.py
│   ├── run_baseline_llm.py
│   ├── run_lora.py
│   └── run_judge.py
├── src/
│   ├── data/                     # loaders, split
│   ├── preprocessing/            # tokenization, prompt formatting
│   ├── models/                   # baseline_rule, llm_inference, lora_finetune
│   ├── metrics/                  # compute_metrics, llm_judge
│   ├── experiments/              # runner, artifacts (общая инфраструктура)
│   └── utils/                    # config, logging, clearml_utils, artifacts
├── artifacts/                    # результаты экспериментов
```
