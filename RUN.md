# Обучение и использование WhiteCollars

Все команды выполняются **из корня репозитория**. Основной сценарий — Linux x86_64
и Tesla V100. Полный цикл обучения запускаете вы; проверки его не запускают.

## 1. Установить окружение

Если `uv` ещё не установлен:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Откройте новый shell, если команда `uv` не появилась в PATH. Затем:

```bash
nvidia-smi
uv sync --extra v100 --frozen
```

`uv` использует Python из `.python-version` и зависимости из `uv.lock`.
Профиль V100 фиксирует PyTorch 2.10.0 + CUDA 12.6. Не обновляйте torch отдельно:
CUDA 13 сборки не поддерживают V100 (Volta, sm_70).
[Матрица поддержки PyTorch](https://github.com/pytorch/pytorch/blob/main/RELEASE.md).
На сервере должен быть установлен совместимый NVIDIA-драйвер.

## 2. Проверить установку и программу

```bash
make gpu-check UV_FLAGS="--extra v100"
make check UV_FLAGS="--extra v100"
```

Первая команда проверяет CUDA, FFT complex64, градиент фазы и FP16 AMP.
Вторая запускает pytest, сверку с независимым расчётом и короткое демо GS.
Успешное завершение: `ALL CHECKS PASSED`. Полное обучение при этом не запускается.

Для запуска только на CPU достаточно `uv sync --frozen` и `make check`;
в командах ниже замените `--device cuda` на `--device cpu`.

## 3. Подготовить данные

Исходный набор уже находится в `src/dataset/data/`: 1000 train, 100 validation,
100 test и одна диагностическая четвёрка.

Для более широкого покрытия полутонов создайте расширенный набор:

```bash
uv run --extra v100 --frozen python -m dataset.expand_grayscale \
  --out build/data_gray --extra-train 1000 --stress-count 100
```

**Выполняется один раз в новую папку.** Если `build/data_gray` уже создана, повторять
не нужно; для нового варианта задайте другой `--out`.

Получится 2000 train, исходные validation/test/diagnostic и отдельные 100 новых
stress-validation. Добавлены фигуры разной яркости, серый фон, пятна, полосы,
градиенты, текстуры и плотные композиции. Все исходные PNG и записи сохраняются.
Проверка пишет `build/data_gray/grayscale_report.json`.

Новые stress-изображения не входят в train или автоматический выбор checkpoint.
Расширение — эксперимент по покрытию; повышение точности нужно измерить.

## 4. Запустить обучение на V100

```bash
uv run --extra v100 --frozen project train \
  --manifest build/data_gray/manifest.json \
  --checkpoint checkpoints/best_model.pt \
  --device cuda --epochs 60 --batch-size 4 \
  --lr 0.002 --unroll-steps 6 --base-dim 48 --seed 3701 --polish-steps 25
```

Если нужен исходный набор, замените manifest на `src/dataset/data/manifest.json`.

| Параметр | Значение в примере |
|---|---|
| `--epochs` | 60 эпох; качество проверяется на validation после каждой |
| `--batch-size` | 4; при нехватке GPU-памяти уменьшите до 2 или 1 |
| `--checkpoint` | Куда сохранить лучшие веса |
| `--polish-steps` | 25 шагов уточнения при финальном экспорте; не число шагов обучения |
| `--seed` | Seed инициализации и загрузчика; GPU-вычисления могут быть недетерминированными |

Обучение использует FP16 AMP, физика и метрики — float32/complex64.
В версии `visual-fit-v2` основной критерий — **ошибка формы, яркости нужного объекта и смешения плоскостей**
(меньше лучше); при равенстве предпочтение отдаётся SSIM. Нормировка на максимум
остаётся в финальном восстановлении и официальных метриках. В обучении она больше
не позволяет улучшать главный критерий простым затемнением кадра через яркий пик.
Это внутренний критерий выбора, а не новая метрика организаторов.
Loss сочетает RMS-ошибку формы, ошибку яркости на объекте и штраф сходства
с чужой целью. Пирсон и оба SSIM остаются вспомогательными членами;
добавлен штраф худшей плоскости. Энергия целевых плоскостей выравнивается перед обратным
распространением, чтобы небольшие объекты не подавлялись крупными. Test не участвует в выборе.

Результаты:

```text
checkpoints/best_model.pt          лучшие обученные веса
checkpoints/training_history.json  метрики по эпохам (обновляются каждую эпоху)
checkpoints/previews/              картинки 4 фиксированных validation-примеров
preds/val/                        ДОЭ для validation
preds/diag/                       ДОЭ для диагностической четвёрки
```

Для ручной короткой проверки цикла обучения можно добавить `--smoke --epochs 1`.
Тогда train ограничится 8, validation — 2 примерами; такой запуск не даёт итоговую
модель для сдачи. По умолчанию в примере выше обучается полный train.

### После неудачного запуска v1

Старые checkpoint загружаются с прежним оператором модели; формат весов сохранён.
Обучающая цель и новый баланс энергии автоматически записываются в новые веса.
Для нового эксперимента используйте отдельную папку, сохранив прежний checkpoint:

```bash
uv run --extra v100 --frozen project train \
  --manifest build/data_gray/manifest.json \
  --checkpoint checkpoints/shape_v2/best_model.pt \
  --device cuda --epochs 5 --batch-size 4 --lr 0.0005 \
  --unroll-steps 6 --base-dim 48 --seed 3701 --polish-steps 0
```

Сначала оцените `checkpoints/shape_v2/previews/`, `Visual` и `Shape` в журнале. Пять эпох —
диагностический эксперимент, не обещание окончательного качества. Если изображения
удовлетворительны, полный запуск задаётся отдельно. Данных об обучении версии v2
пока нет; один диагностический пример с оптимизацией фазы не заменяет такой проверки.

Можно также проверить старые веса с новым критерием уточнения фазы без переобучения:

```bash
uv run --extra v100 --frozen project predict \
  --checkpoint checkpoints/best_model.pt --device cuda \
  --sample-dir src/dataset/data/validation/000001 \
  --out outputs/shape_polish --polish-steps 200
```

200 шагов увеличивают время синтеза; это диагностический режим, скорость нужно измерить.

## 5. Использовать модель на четырёх PNG

Подставьте свои пути вместо `a.png` … `d.png`:

```bash
uv run --extra v100 --frozen project predict \
  --checkpoint checkpoints/best_model.pt --device cuda \
  --inputs a.png b.png c.png d.png --out outputs/example --polish-steps 25
```

Файлы должны быть PNG 128×128, 8 бит, grayscale L. Они соответствуют расстояниям
24,1 / 25,3 / 26,5 / 27,7 мм в указанном порядке. RGB и другой размер вызывают ошибку.
Если файлы в папке названы `01.png…04.png` или `1.png…4.png`, используйте:

```bash
uv run --extra v100 --frozen project predict \
  --checkpoint checkpoints/best_model.pt --device cuda \
  --sample-dir src/dataset/data/diagnostic/0000 --out outputs/diagnostic --polish-steps 25
```

В выходной папке будут `doe.png`, `recon_1.png…recon_4.png`, `metrics.json` и
панель `preview/comparison.png` для визуальной проверки целей и восстановлений.
`--polish-steps 0` — только прямой проход сети, без последующей оптимизации фазы.
Для измерения ускорения сравнивайте оба режима. Время `inference_ms` включает
синтез и выбранное уточнение; загрузка весов, PNG и расчёт итоговых метрик в него
не входят. Отсутствующий или несовместимый checkpoint вызывает ошибку.

## 6. Проверить сохранённый ДОЭ

```bash
uv run --extra v100 --frozen doe-eval reconstruct \
  --inputs a.png b.png c.png d.png --doe outputs/example/doe.png \
  --out outputs/recheck --device cuda
```

Расчёт выполняется из PNG на диске. Сохраняются четыре восстановления и
`metrics.json`: SSIM global, SSIM skimage и Пирсон по плоскостям и в среднем.

Оценка уже экспортированной validation:

```bash
uv run --extra v100 --frozen doe-eval evaluate \
  --data src/dataset/data/validation --doe-dir preds/val \
  --out outputs/validation.csv --device cuda
```

## 7. Сравнить сеть, уточнение и классический метод

После обучения:

```bash
uv run --extra v100 --frozen python -m model.benchmark \
  --checkpoint checkpoints/best_model.pt --split validation --limit 0 \
  --iters 200 --polish-steps 25 --device cuda --out outputs/benchmark
```

На одинаковых примерах сравниваются random, GS-200, сеть и сеть с уточнением.
Сохраняются 8-битные ДОЭ, `results.csv` и `summary.json` со средними, минимумами,
10-м процентилем и медианами. Без `--checkpoint` проверяются только random и GS.
`--limit 0` означает весь split; положительное число — первые K примеров.
Замер времени включает первый запуск библиотек, это не прогретый GPU-бенчмарк.

Проверка новых полутоновых типов:

```bash
uv run --extra v100 --frozen python -m model.benchmark \
  --manifest build/data_gray/stress/manifest.json --split validation --limit 0 \
  --checkpoint checkpoints/best_model.pt --device cuda --out outputs/grayscale_stress
```

После выбора параметров выполните итоговую проверку на исходном test:

```bash
uv run --extra v100 --frozen project export \
  --checkpoint checkpoints/best_model.pt --split test --out preds/test \
  --device cuda --batch-size 4 --polish-steps 25

uv run --extra v100 --frozen doe-eval evaluate \
  --data src/dataset/data/test --doe-dir preds/test --out outputs/test.csv --device cuda
```

## 8. Собрать и запустить Docker для сдачи

Перед сборкой в репозитории должны быть **реальные** `checkpoints/best_model.pt`.
Если используете Git LFS, скачайте содержимое весов: текстовый указатель не подходит.
Веса в этой сессии не обучались. Без них контейнер не сможет выполнить прогноз.

```bash
docker build -t whitecollars .
mkdir -p output
docker run --rm --gpus all \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" whitecollars \
  --inputs /input/a.png /input/b.png /input/c.png /input/d.png \
  --out /output --device cuda --polish-steps 25
```

Исходные четыре изображения должны лежать в `input/`; результат появится в `output/`.
Python-зависимости ставятся при сборке автоматически. Для GPU хосту требуются
NVIDIA-драйвер и NVIDIA Container Toolkit. Проверить GPU внутри образа:

```bash
docker run --rm --gpus all --entrypoint python whitecollars -m model.gpu_check
```

CPU-проверка: уберите `--gpus all` и задайте `--device cpu`.
Для расчёта восстановлений через контейнер:

```bash
docker run --rm --gpus all --entrypoint doe-eval \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" whitecollars reconstruct \
  --inputs /input/a.png /input/b.png /input/c.png /input/d.png \
  --doe /output/doe.png --out /output/recheck --device cuda
```

К сдаче требуются код, веса, Dockerfile, requirements.txt и README.
Сборка Docker здесь не подтверждена: доступ к локальному Docker-демону запрещён.

## Протокол и допущения

- Освещение — плоская волна амплитуды 1; `H=exp(jφ)`. Используется single-FFT
  формула Френеля из PROMPT.md, без дополнительного множителя площади пикселя.
- Шаг выходного пикселя `λz/(N·PIXEL)` зависит от расстояния, примерно 10,02–11,51 мкм.
- По умолчанию сравнивается амплитуда, `--compare amplitude`; `intensity` даёт |K|².
- По умолчанию каждая плоскость делится на свой максимум, `--norm max`.
  `--norm minmax` дополнительно вычитает минимум. Эти флаги есть у `doe-eval`;
  обучение и прогноз используют `amplitude/max`.
- Метрики считаются в [0,1] из перечитанного 8-битного ДОЭ, до округления PNG
  восстановлений. Фаза: 0 ↔ −π, 255 ↔ +π.
- Глобальный SSIM и skimage SSIM выводятся отдельно. Для skimage выбран стандартный
  фильтр 7×7, выборочная ковариация, `data_range=1`; Gaussian σ=1,5 не используется.
  Организаторы пока не уточнили окно SSIM и нормировку — это остаётся допущением.
- Для постоянных изображений Пирсон условно равен 1 при точном совпадении и 0 иначе.
- Численную точность сети и качество на неизвестных объектах нельзя обещать заранее.
  Ограничения данных и измеренный GS baseline описаны в [AUDIT.md](AUDIT.md).
