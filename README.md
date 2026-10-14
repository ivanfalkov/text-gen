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
## Результаты (val_holdout, 500 примеров DialogSum)

| Модель | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|---|---|---|---|---|
| baseline_rule | 0.2297 | 0.0533 | 0.1957 | 0.0696 |
| baseline_llm (Qwen2.5-0.5B, zero-shot) | 0.2922 | 0.0954 | 0.2208 | 0.0787 |
| **lora (Qwen2.5-0.5B + LoRA, 3 эпохи)** | **0.4591** | **0.2043** | **0.3713** | **0.2294** |

LoRA-дообучение дало относительно baseline_llm:
- ROUGE-1: **+57%**
- ROUGE-2: **+114%**
- ROUGE-L: **+68%**
- BLEU: **+191%**



# Infra
### WSL
ткрыть PowerShell от имени администратора, выполнить `wsl --install`, перезагрузить компьютер. Открыть Ubuntu через меню Пуск, при первом запуске задать логин и пароль Linux (не связан с Windows-аккаунтом). Проверить, что WSL видит GPU: `nvidia-smi` — должна показаться RTX 4060 Laptop. Если пусто — обновить драйвер NVIDIA для Windows и перезапустить WSL: `wsl --shutdown` из PowerShell, потом заново открыть Ubuntu.

**Установка uv в WSL**

Установить uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`. Добавить в PATH: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc`. uv venv --python 3.12 .Проверить: `uv --version`.
**подключить репозиторий NVIDIA и установить cuDNN**
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
sudo apt install -y libnccl2 libnccl-dev
sudo ldconfig

**Настройка окружения под Linux**

Задать отдельный каталог для venv, чтобы не конфликтовал с Windows: `echo 'export UV_PROJECT_ENVIRONMENT=".venv_linux"' >> ~/.bashrc && source ~/.bashrc`. Перейти в проект: `cd /mnt/c/Users/Ivan/Documents/projects/text-gen` (здесь `/mnt/c/...` — доступ к диску `C:\` из Linux). Синхронизировать зависимости: `uv sync` — uv создаст `.venv_linux` и установит всё из `pyproject.toml`. Установить PyTorch с CUDA: `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`. Проверить GPU: `uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"` — ожидается `2.5.x+cu124`, `True`, `NVIDIA GeForce RTX 4060 Laptop GPU`. Установить vLLM: `uv pip install vllm`. Проверить импорт: `uv run python -c "import vllm; print(vllm.__version__)"`.

**Ежедневный запуск**

Открыть Ubuntu (через меню Пуск или команду `wsl` в PowerShell), перейти в проект: `cd /mnt/c/Users/Ivan/Documents/projects/text-gen`, запустить нужный скрипт: `uv run python experiments/run_baseline_rule.py`. Для отладки с подробными логами: `uv run python experiments/run_baseline_rule.py --verbose`.