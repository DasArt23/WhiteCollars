# Запуск `doe_eval` на GPU-сервере

## 1. Установка окружения
```bash
# Установка uv (если не установлен)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Синхронизация зависимостей из uv.lock
make setup

# Проверка доступности GPU в PyTorch
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device count:', torch.cuda.device_count())"