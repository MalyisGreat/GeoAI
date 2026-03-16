from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    smoothed: list[float] = []
    running = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        smoothed.append(running / len(queue))
    return smoothed


def plot_series(
    draw: ImageDraw.ImageDraw,
    *,
    bounds: tuple[int, int, int, int],
    values: list[float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    left, top, right, bottom = bounds
    draw.rectangle(bounds, outline=(210, 210, 210), width=1)
    if not values:
        draw.text((left + 12, top + 12), f"{label}: no data", fill=(90, 90, 90))
        return
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    width = max(1, right - left - 40)
    height = max(1, bottom - top - 40)
    points = []
    for index, value in enumerate(values):
        x = left + 20 + (index / max(1, len(values) - 1)) * width
        y = top + 20 + (1.0 - ((value - min_value) / (max_value - min_value))) * height
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=color, width=3)
    else:
        x, y = points[0]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    draw.text((left + 12, top + 10), label, fill=(20, 20, 20))
    draw.text((right - 108, top + 10), f"max {max(values):.2f}", fill=(80, 80, 80))
    draw.text((right - 108, bottom - 24), f"min {min(values):.2f}", fill=(80, 80, 80))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render training curves from run JSONL metrics.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    metrics_dir = run_root / "metrics"
    step_rows = load_jsonl(metrics_dir / "train_steps.jsonl")
    train_rows = load_jsonl(metrics_dir / "train.jsonl")
    val_rows = load_jsonl(metrics_dir / "val.jsonl")

    step_loss = smooth([float(row["loss"]) for row in step_rows if "loss" in row], args.smooth_window)
    step_speed = smooth(
        [float(row["step_samples_per_sec"]) for row in step_rows if "step_samples_per_sec" in row],
        args.smooth_window,
    )
    val_median = [float(row["median_km"]) for row in val_rows if "median_km" in row]

    canvas = Image.new("RGB", (1400, 960), color=(248, 247, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 24), f"Training Curves: {run_root.name}", fill=(15, 15, 15))

    plot_series(
        draw,
        bounds=(40, 80, 1360, 340),
        values=step_loss,
        color=(27, 91, 160),
        label="Train loss (smoothed steps)",
    )
    plot_series(
        draw,
        bounds=(40, 380, 1360, 640),
        values=val_median,
        color=(198, 75, 55),
        label="Validation median error km (epochs)",
    )
    plot_series(
        draw,
        bounds=(40, 680, 1360, 940),
        values=step_speed,
        color=(37, 131, 89),
        label="Train throughput samples/sec (smoothed steps)",
    )

    output_path = Path(args.output) if args.output else run_root / "training_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
