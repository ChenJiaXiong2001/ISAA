import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\15637\PycharmProjects\ISAA")
DATA = ROOT / "outputs" / "evaluations" / "pool_weighted_100_cosine_20261016"
OUT = DATA / "training_analysis.png"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


epochs = read_csv(DATA / "epochs.csv")
per_class = read_csv(DATA / "per_class_accuracy.csv")
for row in epochs:
    for key in ("epoch", "train_top1", "val_top1", "train_loss", "val_loss"):
        row[key] = float(row[key])
for row in per_class:
    row["accuracy_percent"] = float(row["accuracy_percent"])

font_path = next((p for p in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simsun.ttc")) if p.exists()), None)
font = ImageFont.truetype(str(font_path), 28) if font_path else ImageFont.load_default()
small = ImageFont.truetype(str(font_path), 20) if font_path else ImageFont.load_default()
tiny = ImageFont.truetype(str(font_path), 16) if font_path else ImageFont.load_default()
img = Image.new("RGB", (1800, 650), "white")
draw = ImageDraw.Draw(img)
draw.text((900, 24), "原池化100轮训练与统一评估逐类准确率", fill="#111827", anchor="ma", font=font)

# Left panel: training curve.
lx, rx, ty, by = 110, 980, 110, 540
for tick in range(0, 101, 20):
    y = by - tick / 100 * (by - ty)
    draw.line((lx, y, rx, y), fill="#e5e7eb", width=2)
    draw.text((lx - 12, y), str(tick), fill="#475569", anchor="rm", font=small)
draw.line((lx, ty, lx, by), fill="#111827", width=3)
draw.line((lx, by, rx, by), fill="#111827", width=3)
def point(row, key):
    x = lx + (row["epoch"] - 1) / 99 * (rx - lx)
    y = by - row[key] * (by - ty)
    return x, y
for key, color, width in (("train_top1", "#94a3b8", 3), ("val_top1", "#2563eb", 4)):
    points = [point(r, key) for r in epochs]
    draw.line(points, fill=color, width=width)
for epoch in (36, 56):
    x = lx + (epoch - 1) / 99 * (rx - lx)
    draw.line((x, ty, x, by), fill="#64748b", width=2)
best = max(epochs, key=lambda r: r["val_top1"])
bx, by_best = point(best, "val_top1")
draw.ellipse((bx - 7, by_best - 7, bx + 7, by_best + 7), fill="#2563eb")
draw.text((bx - 18, by_best - 30), f"日志最佳 {100 * best['val_top1']:.4f}% / 第{int(best['epoch'])}轮", fill="#1e3a8a", anchor="rb", font=tiny)
draw.text(((lx + rx) // 2, 592), "训练轮次", fill="#475569", anchor="ma", font=small)
draw.text((62, (ty + by) // 2), "Top-1（%）", fill="#475569", anchor="mm", font=small)
draw.line((lx + 20, 70, lx + 90, 70), fill="#94a3b8", width=3)
draw.text((lx + 100, 70), "训练", fill="#475569", anchor="lm", font=tiny)
draw.line((lx + 190, 70, lx + 260, 70), fill="#2563eb", width=4)
draw.text((lx + 270, 70), "验证", fill="#475569", anchor="lm", font=tiny)

# Right panel: current class extremes.
selected = sorted(per_class, key=lambda r: r["accuracy_percent"])[:5] + sorted(per_class, key=lambda r: r["accuracy_percent"], reverse=True)[:5]
selected.sort(key=lambda r: r["accuracy_percent"])
sx0, sx1, sy0, row_h = 1080, 1740, 140, 42
draw.text((1410, 90), "当前统一评估的逐类准确率极值", fill="#111827", anchor="ma", font=small)
for tick in (70, 80, 90, 100):
    x = sx0 + (tick - 65) / 40 * (sx1 - sx0)
    draw.line((x, sy0 - 10, x, sy0 + row_h * len(selected)), fill="#e5e7eb", width=2)
    draw.text((x, sy0 - 20), str(tick), fill="#475569", anchor="mb", font=tiny)
for i, row in enumerate(selected):
    y = sy0 + i * row_h
    value = row["accuracy_percent"]
    x = sx0 + (value - 65) / 40 * (sx1 - sx0)
    draw.text((sx0 - 14, y + row_h / 2), row["name"], fill="#243b53", anchor="rm", font=tiny)
    draw.line((sx0, y + row_h / 2, x, y + row_h / 2), fill="#d44d54" if value < 90 else "#138a72", width=18)
    draw.text((x + 12, y + row_h / 2), f"{value:.1f}%", fill="#243b53", anchor="lm", font=tiny)

img.save(OUT, quality=95)
print(OUT)
