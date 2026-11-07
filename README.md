# WhiteCollars

Нейросетевой синтез фазовых дифракционных оптических элементов (ДОЭ).

Модель совместно принимает **четыре изображения 128×128**, рассчитывает **одну
фазовую маску** и восстанавливает четыре изображения на расстояниях
24,1 / 25,3 / 26,5 / 27,7 мм. Результат: 8-битный grayscale PNG с фазой −π…+π,
четыре PNG восстановления и отчёт с SSIM и коэффициентом Пирсона.

**[RUN.md](RUN.md) — все команды подготовки, обучения, использования и сдачи.**

## Что реализовано

- `HoloUnrollNet`: нейросеть с развёрнутыми физическими итерациями и Fourier U-Net.
- Дифференцируемый single-FFT расчёт Френеля на PyTorch; независимый NumPy-эталон.
- Обучение через физику с учётом квантования фазы в 8 бит.
- Пирсон, глобальный SSIM и оконный SSIM; выбор лучших весов с учётом SSIM и Пирсона.
- Необязательное уточнение фазы после прямого прохода сети.
- Классический многоплоскостной алгоритм Герчберга–Сэкстона (GS) для сравнения.
- Генерация и проверка процедурных данных, CLI и Dockerfile для сдачи.

## Быстрый запуск на Tesla V100

Из корня репозитория, на Linux x86_64 с доступным NVIDIA GPU:

```bash
uv sync --extra v100 --frozen
make gpu-check UV_FLAGS="--extra v100"
make check UV_FLAGS="--extra v100"
```

Профиль V100 использует PyTorch 2.10.0 с CUDA 12.6. Поддержка Volta отсутствует
в новых CUDA 13 сборках — [матрица PyTorch](https://github.com/pytorch/pytorch/blob/main/RELEASE.md).
`gpu-check` проверяет реальные вычисления и градиент на GPU.
На CPU: `uv sync --frozen`, затем `make check`.

## Обучение

Подготовка расширенного набора выполняется один раз, в новую папку:

```bash
uv run --extra v100 --frozen python -m dataset.expand_grayscale \
  --out build/data_gray --extra-train 1000 --stress-count 100
```

Запуск обучения пользователем:

```bash
uv run --extra v100 --frozen project train \
  --manifest build/data_gray/manifest.json \
  --checkpoint checkpoints/best_model.pt \
  --device cuda --epochs 60 --batch-size 4 \
  --lr 0.002 --unroll-steps 6 --base-dim 48 --seed 3701 --polish-steps 25
```

Лучшие веса сохраняются в `checkpoints/best_model.pt`, журнал — в
`checkpoints/training_history.json`. Обученные веса в этой сессии не создавались.
Для исходного набора используйте `--manifest src/dataset/data/manifest.json`.

## Использование обученной модели

```bash
uv run --extra v100 --frozen project predict \
  --checkpoint checkpoints/best_model.pt --device cuda \
  --inputs a.png b.png c.png d.png --out outputs/example --polish-steps 25
```

Входные файлы должны быть PNG, 128×128, 8 бит, grayscale (режим Pillow `L`).
Порядок файлов соответствует четырём расстояниям. Без совместимого checkpoint
прогноз завершается ошибкой. `--polish-steps 0` отключает уточнение фазы.

В `outputs/example` сохраняются:

| Файл | Содержимое |
|---|---|
| `doe.png` | Одна фазовая маска: 0 ↔ −π, 255 ↔ +π |
| `recon_1.png` … `recon_4.png` | Четыре восстановления |
| `metrics.json` | Метрики каждой плоскости, средние и время синтеза |
| `preview/comparison.png` | Цели сверху и восстановления снизу для визуальной проверки |

## Данные и измеренная точность

Исходный пакет содержит 1000 train / 100 validation / 100 test / 1 diagnostic.
Расширение добавляет 1000 train-четвёрок с более разнообразными полутонами и
отдельные 100 stress-validation. Исходные PNG и сплиты сохраняются без изменений.
Информация о генераторах — в [src/dataset/README.md](src/dataset/README.md).

GS-200 на всех 100 исходных validation-четвёрках, после сохранения ДОЭ в 8 бит:

| Пирсон | SSIM global | SSIM skimage |
|---:|---:|---:|
| 0,5657 | 0,3896 | 0,1267 |

Это результат классического baseline, **не точность обученной сети и не предел
достижимого качества**. [Отчёт по примерам](reports/baseline_validation.csv).
Качество сети, время на V100 и эффект расширения измеряются после обучения.
Процедурный набор не гарантирует аналогичный результат на неизвестных объектах.

## Docker для сдачи

Перед сборкой поместите реальные обученные веса в `checkpoints/best_model.pt`.

```bash
docker build -t whitecollars .
mkdir -p output
docker run --rm --gpus all \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" whitecollars \
  --inputs /input/a.png /input/b.png /input/c.png /input/d.png \
  --out /output --device cuda --polish-steps 25
```

Для GPU на хосте нужны NVIDIA-драйвер и NVIDIA Container Toolkit. Python-зависимости
устанавливаются при сборке образа из `uv.lock`; `requirements.txt` содержит их
экспорт для сдачи. Внутри контейнера дополнительные установки не требуются.
Подробные команды CPU, проверки и оценки — в [RUN.md](RUN.md).
Локальная сборка Docker не проверена: в рабочей среде нет доступа к Docker-демону.

## Структура

```text
src/model/       сеть, обучение, экспорт, проверка GPU и сравнение методов
src/doe_eval/    физика Френеля, метрики, PNG, датасет и CLI doe-eval
src/dataset/     исходные данные, генераторы и аудит
src/tests/       тесты физики, метрик, модели, CLI и данных
checkpoints/     обученные веса после запуска train
reports/         результаты проверок и baseline
docs/            схема оптического восстановления
```

[Сверка с ТЗ и принятые допущения](AUDIT.md) ·
[Оптическая схема](docs/optical_reconstruction.svg).
