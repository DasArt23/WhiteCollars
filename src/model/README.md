# Каркас нейросети синтеза ДОЭ

```bash
cd ~/photonic_model/src/model
uv run prob.py
```

Скрипт использует четыре PNG из `src/dataset/data/manifest.json` (`diagnostic`, `data_connector`). Чтобы подать свои изображения, передайте четыре пути после `prob.py` в порядке 24,1 / 25,3 / 26,5 / 27,7 мм. Формат входа — одноканальный PNG 128×128.

`DOENet` в `network.py` принимает тензор формы `[B, 4, 128, 128]` и выдаёт фазу в радианах формы `[B, 128, 128]`. Скрипт сохраняет `doe.png`, четыре реконструкции из сохранённой маски и `metrics.json` в отдельную папку.

## Training

Run from the model directory:

```bash
cd ~/photonic_model/src/model
uv run train.py
```

The unchanged data connector provides train and validation batches. The
training program fits an independent phase mask for each training quartet.
These fixed masks supervise the network with a circular phase loss. Only
the neural network weights are updated during supervised training. Validation
uses Pearson and global SSIM on reconstructions from the quantized mask.
The best validation checkpoint is saved as `best.pt`. The training folder
contains `history.json` and a final `progress.png`.

`uv run prob.py` loads the newest `best-*.pt` from `src/model/weights`.
When there are no weights, it trains for 10 epochs on CPU with two quartets per batch, prints
progress, and saves the best validation weights for future runs.
