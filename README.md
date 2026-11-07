# WhiteCollars

Нейросетевой синтез фазовых дифракционных оптических элементов (ДОЭ).

Модель совместно принимает **четыре изображения 128×128**, рассчитывает **одну
фазовую маску** и восстанавливает четыре изображения на расстояниях
24,1 / 25,3 / 26,5 / 27,7 мм. Результат: 8-битный grayscale PNG с фазой −π…+π,
четыре PNG восстановления и отчёт с SSIM и коэффициентом Пирсона.

**[RUN.md](RUN.md) — все команды подготовки, обучения, использования и сдачи.**

Актуальное исправление — **balanced-fidelity-v4**: Пирсон и SSIM вместе с ошибками
энергии, чужих контуров и перепадов. Добавлены дообучение сохранённых весов через
`--init-checkpoint`, уменьшение LR при плато (`--lr-schedule plateau`) и защита
выбора checkpoint от ухудшения любой из двух метрик относительно начала запуска.
Для симуляционных A-четвёрок сохранён порядок расстояний при аугментации.
Восстановления рассчитываются из фазового PNG без ретуши. Чёткость обученной v4
ещё не подтверждена; проверки кода не являются проверкой её качества.
Согласно переданному разъяснению организаторов от 07.11.2026, несколько проходов
и гибрид допустимы при первичном синтезе нейросетью. Здесь исходную фазу предсказывает
`HoloUnrollNet`; Adam при --polish-steps > 0 уточняет именно её. GS используется
отдельно для сравнения, а не вместо нейросети в основном решении.
В `metrics.json` отдельно указаны `network_ms`, `refinement_ms` и общий `inference_ms`.

## Что реализовано

- `HoloUnrollNet`: нейросеть с развёрнутыми физическими итерациями и Fourier U-Net.
- Дифференцируемый single-FFT расчёт Френеля на PyTorch; независимый NumPy-эталон.
- Обучение через физику с учётом квантования фазы в 8 бит. В `visual-fit-v2`
  loss контролирует форму, яркость нужного объекта и разделение плоскостей;
  энергия целевых плоскостей выравнивается.
- Пирсон, глобальный SSIM и оконный SSIM; выбор весов по визуальной ошибке со штрафом худшей плоскости.
- Необязательное уточнение фазы после прямого прохода сети.
- В `stable-carrier-v3` — фазовая инициализация с фиксированным carrier, ограниченные
  поправки, FP32 по умолчанию и контроль реальных обновлений оптимизатора.
- Классический многоплоскостной алгоритм Герчберга–Сэкстона (GS) для сравнения.
- Генерация и проверка процедурных данных, CLI и Dockerfile для сдачи.

## Архитектура и научные источники

`HoloUnrollNet` — название реализованной здесь комбинации физических итераций,
свёрточных и спектральных блоков. Это **не точная копия одной опубликованной сети**
и не готовая архитектура Google. Основной код — [src/model/holonet.py](src/model/holonet.py).
Источники идей и отличия реализации:

| Компонент проекта | Первоисточник | Что используется |
|---|---|---|
| Развёрнутые физические итерации | Monga, Li, Eldar, [Algorithm Unrolling](https://doi.org/10.1109/MSP.2020.3016905), IEEE SPM, 2021 | Итерационный алгоритм превращается в последовательность обучаемых шагов; наши шаги построены вокруг Френеля. |
| `SpectralConv2d` | Li et al., [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895), ICLR 2021 | Обучаемое смешивание низких Fourier-мод. У нас упрощённый спектральный блок, а не полная официальная модель FNO. |
| `ConvNeXtSEBlock` | Liu et al., [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), FAIR / UC Berkeley, CVPR 2022 | Depthwise-свёртка 7×7, pointwise-преобразования, residual. В проекте изменены нормировка и структура блока, добавлен SE. |
| Канальное внимание SE | Hu, Shen, Sun, [Squeeze-and-Excitation Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf), CVPR 2018 | Глобальный pooling и обучаемое перевзвешивание каналов. |
| `FourierUNetRefiner` | Ronneberger, Fischer, Brox, [U-Net](https://lmbweb.informatik.uni-freiburg.de/Publications/2015/RFB15a/), University of Freiburg, MICCAI 2015 | Encoder–decoder со skip-соединениями; здесь другие блоки и спектральный bottleneck. |
| STE-квантование фазы | Bengio, Léonard, Courville, [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/abs/1308.3432), 2013 | Идея straight-through gradient; применена к нашему 8-битному фазовому PNG. |
| Классический GS и фазовое восстановление | Fienup, [Phase retrieval algorithms: a comparison](https://doi.org/10.1364/AO.21.002758), Applied Optics, 1982 | Прямое/обратное распространение с заменой амплитуды. Наш baseline обобщён на четыре плоскости. |
| SSIM | Wang et al., [Image quality assessment: From error visibility to structural similarity](https://ece.uwaterloo.ca/~z70wang/research/ssim/), IEEE TIP, 2004 | Определение SSIM. Конкретные global и skimage 7×7 варианты проекта описаны в RUN.md. |
| Adam / AdamW | Kingma, Ba, [Adam](https://arxiv.org/abs/1412.6980), ICLR 2015; Loshchilov, Hutter, [AdamW](https://arxiv.org/abs/1711.05101), ICLR 2019 | AdamW обучает веса сети; Adam оптимизирует фазу при необязательном уточнении. Это оптимизаторы, а не архитектура сети. |

Профильная работа по нейросетевой голографии: Peng, Choi, Padmanaban, Wetzstein,
[Neural Holography with Camera-in-the-loop Training](https://doi.org/10.1145/3414685.3417802),
Stanford, ACM TOG / SIGGRAPH Asia 2020;
[официальные код и материалы](https://github.com/computational-imaging/neural-holography).
Это контекст для обучения голограмм через дифференцируемое распространение;
наш `HoloUnrollNet` не является их HoloNet, калибровка камерой здесь не реализована.
Предобученные веса этих работ не используются. Ссылки объясняют происхождение
компонентов, а качество конкретной комбинации подтверждается только нашими экспериментами.

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

**Актуальный набор — 4608 train:** прежние 4096 и 512 новых четвёрок из
`ZAKHAR_FULL_TRAIN_train512.zip`. Импорт и команда FP32-обучения описаны в
[RUN.md](RUN.md#актуальный-набор-4608-train-с-дополнительным-архивом).
Validation/test/stress при объединении сохраняются отдельно, пиксели не изменяются.

Для подготовленного архива с 4096 обучающими четвёрками используйте
`--manifest build/data_gray_4096/manifest.json`; команды короткого запуска — в RUN.md.

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
Неудачный запуск v1 показал, что рост SSIM может сопровождаться потерей объектов.
В `visual-fit-v2` исправлены обучающий критерий и баланс энергии, добавлены автоматические
validation-панели. Новая версия требует проверки обучением; команды короткого
эксперимента и применения нового уточнения к старым весам приведены в RUN.md.
Качество сети, время на V100 и эффект расширения измеряются после обучения.
Примеры прямой оптимизации фазы: [панель](reports/visual_000100_visual.png),
[результаты на пяти четвёрках](reports/visual_fit_diagnostic.json).
Это диагностика метода, а не результат обученной сети visual-fit-v2.
Процедурный набор не гарантирует аналогичный результат на неизвестных объектах.

## Физика и ограничения качества

Интерференция учтена суммой **комплексных полей** до вычисления модуля.
В фазе волны усиливаются; при разности фаз π равные поля могут взаимно гаситься.
Это интерференция, а не резонанс ([учебник OpenStax](https://openstax.org/books/university-physics-volume-3/pages/3-1-youngs-double-slit-interference)).
Проверки контролируют противофазу, независимость интенсивности от общей фазы
и норму поля по теореме Парсеваля с масштабом именно нашей формулы Френеля.

Одна маска имеет 16384 фазовых значения, а четыре изображения — 65536 значений
амплитуды. Это объясняет связанность плоскостей, но не даёт численного потолка
SSIM или Пирсона: достижимость зависит от конкретных целей. Увеличение штрафа
не создаёт дополнительные степени свободы. Шум и чужие контуры остаются
измеряемыми ограничениями, а не устранённой проблемой.

Риски оценки: неизвестные типы объектов; ещё не уточнённые организаторами
нормировка и вариант SSIM; компромисс чёткости и времени уточнения. При проверке
нужны все четыре изображения, Пирсон, SSIM и визуальные панели. Наш внутренний
критерий выбора checkpoint не является формулой итогового балла организаторов.

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
