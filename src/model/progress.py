"""Render training history as a static PNG."""
from PIL import Image, ImageDraw, ImageFont

INK = "#16243a"
MUTED = "#52647d"
GRID = "#dce5f0"


def draw_panel(draw, top, title, color, points, last_epoch, subtitle=None):
    label = ImageFont.load_default(size=16)
    heading = ImageFont.load_default(size=22)
    left, right = 98, 944
    y_top, y_bottom = top + 62, top + 208
    values = [value for _, value in points]
    span = max(max(values) - min(values), 0.001)
    low, high = min(values) - span * 0.12, max(values) + span * 0.12
    if max(values) - min(values) < span:
        low, high = min(values) - span / 2, max(values) + span / 2
    draw.rounded_rectangle((24, top, 976, top + 249), radius=12, fill="white")
    draw.text((45, top + 13), title, fill=color, font=heading)
    if subtitle:
        draw.text((right, top + 19), subtitle, fill=MUTED, font=label, anchor="rt")

    for i in range(5):
        fraction = i / 4
        y = y_bottom - fraction * (y_bottom - y_top)
        draw.line((left, y, right, y), fill=GRID)
        draw.text((left - 10, y), f"{low + fraction * (high - low):.4f}",
                  fill=MUTED, font=label, anchor="rm")
    ticks = min(last_epoch, 8)
    for i in range(ticks + 1):
        epoch = round(i * last_epoch / ticks)
        x = left + epoch / last_epoch * (right - left)
        draw.line((x, y_top, x, y_bottom), fill=GRID)
        draw.text((x, y_bottom + 9), str(epoch), fill=MUTED, font=label, anchor="mt")

    xy = [
        (left + epoch / last_epoch * (right - left),
         y_bottom - (value - low) / (high - low) * (y_bottom - y_top))
        for epoch, value in points
    ]
    if len(xy) > 1:
        draw.line(xy, fill=color, width=3)
    if len(xy) <= 20:
        for x, y in xy:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def save_progress(path, baseline, history):
    canvas = Image.new("RGB", (1000, 875), "#edf2f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 21), "Training progress", fill=INK, font=ImageFont.load_default(size=32))
    draw.text((31, 61), "Epoch 0: validation before training. Each panel has its own Y scale.",
              fill=MUTED, font=ImageFont.load_default(size=17))
    last_epoch = history[-1]["epoch"]
    for key, label, color, top in (
        ("pearson", "Validation Pearson", "#1764bb", 93),
        ("ssim", "Validation global SSIM", "#14795f", 352),
    ):
        points = [(0, baseline[key])] + [(row["epoch"], row["validation"][key]) for row in history]
        draw_panel(draw, top, label, color, points, last_epoch)
    losses = [(row["epoch"], row["optical_loss"]) for row in history]
    draw_panel(draw, 611, "Training optical loss", "#c32635", losses, last_epoch,
               "Penalizes the weaker gain in SSIM and Pearson")
    draw.text((500, 853), "Epoch", fill=INK, font=ImageFont.load_default(size=16), anchor="mt")
    canvas.save(path)

