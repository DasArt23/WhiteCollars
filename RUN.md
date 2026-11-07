# Обучение и использование WhiteCollars

Все команды выполняются **из корня репозитория**. Основной сценарий — Linux x86_64
и Tesla V100. Полный цикл обучения запускаете вы; проверки его не запускают.

## Soft600: подготовка и короткая диагностика

Архив `src/dataset/WhiteCollars_Soft600_dataset.zip` включён в репозиторий.
Распаковка с проверкой SHA256, без обучения:

```bash
uv run --extra v100 --frozen python -m dataset.prepare_soft600
```

Основной путь — `src/dataset/data_soft600/manifest.json`: 600 train, 100 validation,
100 test, 1 diagnostic. В train 240 physical_forward и 360 soft_independent.
Контрольные фазы лежат отдельно и не подаются в сеть. Для physical_forward
отключены перестановки каналов и геометрические аугментации. Validation теперь
показывает обе группы отдельно; смешанное среднее не заменяет эти строки.

Добавление только 600 TRAIN к прежним 4608 (512 из архива ZAKHAR уже сохранены):

```bash
uv run --extra v100 --frozen python -m dataset.prepare_soft600 \
  --merge-base build/data_train4608/manifest.json --out build/data_train5208
```

Новый manifest: `build/data_train5208/manifest.json`. Старые validation/test
сохранены; новые находятся отдельно в `build/data_train5208/soft600/manifest.json`.
Результаты выполненных коротких запусков: `reports/soft600_diagnostic/`.

Быстрая проверка обучения на одном TRAIN-примере (не полный train):

```bash
uv run --extra v100 --frozen python scripts/diagnose_training.py \
  --group physical_forward --count 1 --steps 30 --seconds 90 \
  --device cuda --out build/diagnostic_physical_one
```

Для 4 примеров задайте `--count 4`; независимую группу проверяйте отдельным запуском
с `--group soft_independent`. По умолчанию используется новая сеть 6/48 и постоянный
LR=0.001 для изоляции проверки градиентов. Чтобы проверить реальные обученные веса,
добавьте `--checkpoint checkpoints/train4608_v5/best_model.pt`.
В `report.json` сохраняются явно названные SSIM global/local 7×7, LR, нормы
градиентов, изменения весов, shape/min/max/mean промежуточных тензоров и метаданные.
Предельное время проверяется между шагами, поэтому один последний шаг может его
немного превысить. Ни контрольная фаза, ни test не используются в этом обучении.

## Актуальная цель: Pearson + global SSIM, global-fidelity-v5

В v4 основной SSIM в loss и выборе checkpoint был оконным. По уточнению пользователя
в v5 основной **global**: веса Pearson / global / local = 2 / 2 / 0,1.
В ту же метрику переключены проверка регрессии, tie-break и уточнение фазы.
Проверенные ограничения данных и прямые контрольные оптимизации описаны в
[FUNDAMENTALS.md](FUNDAMENTALS.md). 0,8/0,8 не подтверждены.

Если уже идёт v4, остановите его через Ctrl+C и сохраните его веса.
Дообучение с исправленной целью (новый оптимизатор, исходный forward сохранён):

```bash
git pull --ff-only origin zakhar_sandbox
uv run --extra v100 --frozen project train \
  --manifest build/data_train4608/manifest.json \
  --init-checkpoint checkpoints/train4608_v4/best_model.pt \
  --checkpoint checkpoints/train4608_v5/best_model.pt \
  --device cuda --precision fp32 --epochs 5 --batch-size 2 \
  --lr 0.0001 --lr-schedule plateau \
  --unroll-steps 6 --base-dim 48 --seed 3701 --polish-steps 0
```

`Guard` теперь относится к Pearson и **global SSIM**. Оконный SSIM остаётся в журнале.
Полное обучение v5 не запускалось; короткие прогоны Soft600 описаны выше,
результатов новой сети на всей validation ещё нет.
Проверка необходимого физического условия, без оптимизации или обучения:

```bash
uv run --extra v100 --frozen python -m doe_eval.feasibility \
  --manifest build/data_train4608/manifest.json --split train \
  --out build/train_feasibility.json
```

Это диагностический отчёт: он не удаляет данные и не меняет split.

## Прежнее дообучение: balanced-fidelity-v4

При уже обученном `checkpoints/train4608_v3/best_model.pt` сохраните его и используйте
дообучение с новой целью вместо запуска с нуля. Остановите старый запуск через Ctrl+C
перед обновлением кода. Затем:

```bash
git pull --ff-only origin zakhar_sandbox
uv sync --extra v100 --frozen
uv run --extra v100 --frozen project train \
  --manifest build/data_train4608/manifest.json \
  --init-checkpoint checkpoints/train4608_v3/best_model.pt \
  --checkpoint checkpoints/train4608_v4/best_model.pt \
  --device cuda --precision fp32 --epochs 10 --batch-size 2 \
  --lr 0.0001 --lr-schedule plateau \
  --unroll-steps 6 --base-dim 48 --seed 3701 --polish-steps 0
```

Это загрузка весов с новым AdamW, а не восстановление состояния прежнего оптимизатора.
Сохраняются carrier и параметры физического forward исходного checkpoint.
При отсутствии допустимого улучшения validation loss три эпохи LR уменьшается вдвое;
эпоха с провалом проверки Пирсона/SSIM также считается отсутствием прогресса.
Для A-целей из оптической симуляции выключена перестановка плоскостей; уже созданный
manifest повторно собирать не требуется. Test и stress не используются для настройки LR.

Критерий v4: `2*(1-Pearson) + 0.1*(1-globalSSIM) + 2*(1-localSSIM)` плюс
`0.25*shape + 1.5*foreground + 0.5*power + 0.5*multiscale + 0.15*edge + 0.25*foreign`;
добавлен прежний штраф худшей формы `0.15*max(shape)`. `power` сравнивает
распределение энергии, `edge` — ошибку перепадов относительно цели, `foreign` —
энергию чужих объектов на почти чёрных участках своей цели. Это члены loss,
а не подмена amplitude/max при проверке. Значения `Visual` между v2 и v4 несопоставимы.

Новый checkpoint выбирается при Пирсоне и local SSIM не ниже начальной модели
на validation. `Guard: FAIL` означает отказ от сохранения такой эпохи.
`Select` — критерий выбора, `BG` — ошибка на почти чёрном фоне, `Ghost` — доля
энергии в областях чужих объектов. Компоненты сохраняются в training_history.json.
Сравнивайте панели `initial_previews/` и `previews/` после первых двух эпох.
Если остаётся `epoch=0`, дообучение не дало допустимого улучшения: сохранены
исходные веса текущего запуска, которые при --init-checkpoint уже могли быть обученными.
Полное обучение v4 здесь не запускалось; улучшение сети пока не подтверждено.

Без нового обучения можно проверить уточнение фазы старой сети:

```bash
uv run --extra v100 --frozen project predict \
  --checkpoint checkpoints/train4608_v3/best_model.pt \
  --sample-dir src/dataset/data/validation/000067 \
  --out outputs/refined_v4 --device cuda --polish-steps 200
```

Уточнение сохраняет начальный вариант и выбирает кандидат с не меньшими средними
по четвёрке Пирсоном и local SSIM, не большей ошибкой яркости объекта, фона и
перепадов. Если совместного улучшения нет, возвращается исходная фаза. Это
не гарантирует улучшения каждой плоскости по отдельности и не убирает весь шум.
Точечная проверка на присланных DOE PNG без обучения сети:

```bash
uv run --extra v100 --frozen python scripts/diagnose_suppression.py \
  --predictions checkpoints/train4608_v3/previews --steps 200 \
  --out build/final_suppression_v4
```

Скачивание validation-панелей **с локального компьютера**:

```bash
tsh scp --insecure -r \
  fen1x@v100-cc150-server:/home/fen1x/WhiteCollars/checkpoints/train4608_v4/previews \
  ./previews_v4
```

Ниже оставлены подготовка окружения и примеры прежних запусков.

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

Для уже подготовленного архива с **4096 train-четвёрками**:

```bash
mkdir -p build
tar -xzf ~/WhiteCollars_gray_train4096.tar.gz -C build/
```

После распаковки используйте `--manifest build/data_gray_4096/manifest.json`.
Validation и test содержат по 100 отдельных четвёрок; stress также отделён от train.

## 4. Запустить обучение на V100

### Актуальный набор: 4608 train с дополнительным архивом

После подготовки или распаковки `build/data_gray_4096` выполните один раз:

```bash
uv run --extra v100 --frozen python -m dataset.import_exchange \
  --base build/data_gray_4096/manifest.json \
  --archive ZAKHAR_FULL_TRAIN_train512.zip --out build/data_train4608
```

Импорт проверяет параметры оптики, порядок плоскостей, формат PNG, SHA256 и
точные совпадения пикселей с train/validation/test/stress. Исходные файлы не меняются.
Получится 4608 train, 100 validation, 100 test, 1 diagnostic и 100 отдельных stress.
В архиве 256 целей типа A из симуляции и 256 процедурных B, согласно его описанию;
достижимость A нашим оператором без исходных фаз не подтверждена.

Актуальная команда обучения:

```bash
uv run --extra v100 --frozen project train \
  --manifest build/data_train4608/manifest.json \
  --checkpoint checkpoints/train4608_v3/best_model.pt \
  --device cuda --precision fp32 --epochs 60 --batch-size 2 \
  --lr 0.0003 --unroll-steps 6 --base-dim 48 --seed 3701 --polish-steps 0
```

Ниже сохранён пример для прежнего набора из 2000 четвёрок.

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

По умолчанию обучение использует FP32. `--precision amp` включает FP16 AMP;
физика и метрики в обоих режимах — float32/complex64.
В версии `visual-fit-v2` основной критерий — **ошибка формы, яркости нужного объекта и смешения плоскостей**
(меньше лучше); при равенстве предпочтение отдаётся SSIM. Нормировка на максимум
остаётся в финальном восстановлении и официальных метриках. В обучении она больше
не позволяет улучшать главный критерий простым затемнением кадра через яркий пик.
Это внутренний критерий выбора, а не новая метрика организаторов.
Loss сочетает RMS-ошибку формы, ошибку яркости на объекте и штраф сходства
с чужой целью. Пирсон и оба SSIM остаются вспомогательными членами;
добавлен штраф худшей плоскости. Он учитывается также при выборе checkpoint и
результата уточнения: `mean(visual_error) + 0.15 * max(shape_error)` по четырём плоскостям.
В журнале `Worst P` — средний по четвёркам минимум Пирсона, а не общий средний.
Энергия целевых плоскостей выравнивается перед обратным
распространением, чтобы небольшие объекты не подавлялись крупными. Test не участвует в выборе.

Результаты:

```text
checkpoints/best_model.pt          лучшие обученные веса
checkpoints/training_history.json  метрики по эпохам (обновляются каждую эпоху)
checkpoints/previews/              картинки 4 фиксированных validation-примеров
preds/val/                        ДОЭ для validation
preds/diag/                       ДОЭ для диагностической четвёрки
```

Исходная модель участвует в выборе лучших весов. Если обучение её не улучшило,
сохраняется checkpoint с `epoch=0` и явное сообщение в журнале. Это исходные,
не обученные веса; такой результат не подтверждает успех обучения.

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
удовлетворительны, полный запуск задаётся отдельно. В предоставленном запуске v2
плоскости всё ещё смешивались; пять эпох не дали приемлемой чёткости. Эффект
исправлений v3 пока не проверен обучением на V100.

Можно также проверить старые веса с новым критерием уточнения фазы без переобучения:

```bash
uv run --extra v100 --frozen project predict \
  --checkpoint checkpoints/best_model.pt --device cuda \
  --sample-dir src/dataset/data/validation/000001 \
  --out outputs/shape_polish --polish-steps 200
```

200 шагов увеличивают время синтеза; это диагностический режим, скорость нужно измерить.

### Контроль фактического обучения: stable-carrier-v3

Пятиипоховый запуск v2 не дал заметного улучшения формы. В v3 вместо неустойчивого
atan2 двух выходов начальной CNN используется фиксированная случайная фаза и
ограниченная поправка; финальная коррекция фазора также ограничена. Старые веса
сохраняют прежний forward. Режим FP32 выбран по умолчанию, AMP включается явно.

Перед первой эпохой сохраняются `initial_validation.json` и `initial_previews/`.
Сравнивайте лучшие картинки с этими начальными, а не только разные эпохи между собой.
Журнал показывает `Updates`, `AMP skipped` и `Gain` относительно начального Visual.
Если ни одного обновления нет, запуск прекращается с ошибкой. Положительный Gain
означает снижение внутреннего критерия; визуальное качество проверяется отдельно.

Для отдельного эксперимента используйте новую папку весов:

```bash
uv run --extra v100 --frozen project train \
  --manifest build/data_gray_4096/manifest.json \
  --checkpoint checkpoints/stable_v3/best_model.pt \
  --device cuda --precision fp32 --epochs 2 --batch-size 2 \
  --lr 0.0003 --unroll-steps 6 --base-dim 48 --seed 3701 --polish-steps 0
```

Полного обучения v3 в этой сессии не проводилось. Улучшение изображения не гарантируется
изменением численной схемы; начальные и обученные панели нужны для проверки эффекта.

### Архив с 4096 обучающими четвёрками

Архив `WhiteCollars_gray_train4096.tar.gz` содержит папку `data_gray_4096`:
4096 train / 100 validation / 100 test / 1 diagnostic и 100 отдельных stress-validation.

```bash
mkdir -p build
tar -xzf ~/WhiteCollars_gray_train4096.tar.gz -C build/
```

В обучении задайте `--manifest build/data_gray_4096/manifest.json`.
Содержимое архива проверено: 17588 PNG, точных дублей нет. Это те же генератор и seed,
что в исходном расширении; увеличен только train, проверочные сплиты сохранены.

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
