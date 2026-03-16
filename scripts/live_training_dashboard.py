from __future__ import annotations

import argparse
import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GeoBot Live Training</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f4f2ea; color: #181818; }
    .wrap { padding: 20px 24px 28px; max-width: 1400px; margin: 0 auto; }
    h1 { margin: 0 0 6px; font-size: 28px; }
    .sub { color: #555; margin-bottom: 18px; }
    .stats { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }
    .card { background: #fffdf8; border: 1px solid #ddd4c7; border-radius: 10px; padding: 12px 14px; }
    .label { color: #6a6256; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }
    .value { font-size: 24px; font-weight: 700; }
    .small { font-size: 13px; color: #5f5a53; }
    canvas { width: 100%; height: 560px; background: #fffdf8; border: 1px solid #ddd4c7; border-radius: 10px; display: block; }
    .legend { display: flex; gap: 18px; margin: 12px 0 16px; font-size: 14px; }
    .legend span::before { content: ""; display: inline-block; width: 12px; height: 12px; border-radius: 999px; margin-right: 8px; vertical-align: -1px; }
    .median::before { background: #1b5ba0; }
    .mean::before { background: #c64b37; }
    .p90::before { background: #258357; }
    pre { background: #fffdf8; border: 1px solid #ddd4c7; border-radius: 10px; padding: 12px; white-space: pre-wrap; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>GeoBot Live Training</h1>
    <div class="sub" id="subtitle">Waiting for metrics...</div>
    <div class="stats">
      <div class="card"><div class="label">Images Seen</div><div class="value" id="imagesSeen">0</div></div>
      <div class="card"><div class="label">Median KM</div><div class="value" id="medianKm">-</div></div>
      <div class="card"><div class="label">Mean KM</div><div class="value" id="meanKm">-</div></div>
      <div class="card"><div class="label">P90 KM</div><div class="value" id="p90Km">-</div></div>
    </div>
    <div class="legend">
      <span class="median">Median KM</span>
      <span class="mean">Mean KM</span>
      <span class="p90">P90 KM</span>
    </div>
    <canvas id="chart" width="1360" height="560"></canvas>
    <div class="stats" style="margin-top: 16px;">
      <div class="card"><div class="label">Latest Train Loss</div><div class="value" id="trainLoss">-</div><div class="small" id="trainMeta">-</div></div>
      <div class="card"><div class="label">Latest Train SPS</div><div class="value" id="trainSps">-</div><div class="small" id="trainEpochMeta">-</div></div>
      <div class="card"><div class="label">Latest Eval Trigger</div><div class="value" id="evalTrigger">-</div><div class="small" id="evalMeta">-</div></div>
      <div class="card"><div class="label">Checkpoints</div><div class="value" id="checkpointCount">0</div><div class="small" id="checkpointMeta">-</div></div>
    </div>
    <pre id="checkpointList">No checkpoints yet.</pre>
  </div>
  <script>
    const chart = document.getElementById("chart");
    const ctx = chart.getContext("2d");

    function drawChart(points) {
      ctx.clearRect(0, 0, chart.width, chart.height);
      ctx.fillStyle = "#fffdf8";
      ctx.fillRect(0, 0, chart.width, chart.height);

      const left = 70, right = chart.width - 24, top = 24, bottom = chart.height - 56;
      ctx.strokeStyle = "#d6cec1";
      ctx.lineWidth = 1;
      ctx.strokeRect(left, top, right - left, bottom - top);

      if (!points.length) {
        ctx.fillStyle = "#6a6256";
        ctx.font = "18px Segoe UI";
        ctx.fillText("Waiting for the first evaluation point...", left + 24, top + 40);
        return;
      }

      const xValues = points.map((p) => p.images_seen);
      const yValues = points.flatMap((p) => [p.median_km, p.mean_km, p.p90_km]);
      const minX = Math.min(...xValues);
      const maxX = Math.max(...xValues);
      const minY = 0;
      const maxY = Math.max(...yValues) * 1.05;

      const xScale = (value) => left + ((value - minX) / Math.max(1, maxX - minX)) * (right - left);
      const yScale = (value) => bottom - ((value - minY) / Math.max(1, maxY - minY)) * (bottom - top);

      ctx.fillStyle = "#6a6256";
      ctx.font = "12px Segoe UI";
      for (let i = 0; i < 5; i++) {
        const ratio = i / 4;
        const y = top + ratio * (bottom - top);
        const value = maxY - ratio * maxY;
        ctx.strokeStyle = "#ece6da";
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.fillText(Math.round(value).toString(), 14, y + 4);
      }
      ctx.fillText("Images trained", right - 90, chart.height - 16);
      ctx.save();
      ctx.translate(18, top + 120);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("KM off", 0, 0);
      ctx.restore();

      const lines = [
        { key: "median_km", color: "#1b5ba0" },
        { key: "mean_km", color: "#c64b37" },
        { key: "p90_km", color: "#258357" },
      ];
      for (const line of lines) {
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        points.forEach((point, index) => {
          const x = xScale(point.images_seen);
          const y = yScale(point[line.key]);
          if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      ctx.fillStyle = "#6a6256";
      points.forEach((point) => {
        const x = xScale(point.images_seen);
        ctx.fillText(Math.round(point.images_seen).toString(), x - 12, bottom + 18);
      });
    }

    async function refresh() {
      const response = await fetch("/api/status", { cache: "no-store" });
      const payload = await response.json();
      const trainLike = payload.latest_train || payload.latest_step;
      document.getElementById("subtitle").textContent = payload.run_root;
      document.getElementById("imagesSeen").textContent = payload.latest_images_seen.toLocaleString();
      document.getElementById("medianKm").textContent = payload.latest_eval ? payload.latest_eval.median_km.toFixed(1) : "-";
      document.getElementById("meanKm").textContent = payload.latest_eval ? payload.latest_eval.mean_km.toFixed(1) : "-";
      document.getElementById("p90Km").textContent = payload.latest_eval ? payload.latest_eval.p90_km.toFixed(1) : "-";
      document.getElementById("trainLoss").textContent = trainLike ? trainLike.loss.toFixed(4) : "-";
      document.getElementById("trainMeta").textContent = payload.latest_train
        ? `images=${Math.round(payload.latest_train.images_seen)} lr=${payload.latest_train.learning_rate.toFixed(6)}`
        : trainLike
          ? `images=${Math.round(trainLike.images_seen)} step=${trainLike.step}`
          : "-";
      document.getElementById("trainSps").textContent = payload.latest_train
        ? payload.latest_train.samples_per_sec.toFixed(1)
        : trainLike
          ? trainLike.step_samples_per_sec.toFixed(1)
          : "-";
      document.getElementById("trainEpochMeta").textContent = payload.latest_train
        ? `epoch=${payload.latest_train.epoch} seconds=${payload.latest_train.epoch_seconds.toFixed(1)}`
        : trainLike
          ? `epoch=${trainLike.epoch} wait=${trainLike.data_wait_seconds.toFixed(3)}s`
          : "-";
      document.getElementById("evalTrigger").textContent = payload.latest_eval ? payload.latest_eval.trigger : "-";
      document.getElementById("evalMeta").textContent = payload.latest_eval ? `epoch=${payload.latest_eval.epoch} images=${Math.round(payload.latest_eval.images_seen)}` : "-";
      document.getElementById("checkpointCount").textContent = payload.checkpoints.length.toString();
      document.getElementById("checkpointMeta").textContent = payload.last_updated || "-";
      document.getElementById("checkpointList").textContent = payload.checkpoints.length
        ? payload.checkpoints.join("\\n")
        : "No checkpoints yet.";
      drawChart(payload.series);
    }

    refresh().catch((error) => console.error(error));
    setInterval(() => refresh().catch((error) => console.error(error)), 1500);
  </script>
</body>
</html>
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_status(run_root: Path) -> dict[str, Any]:
    metrics_dir = run_root / "metrics"
    train_rows = load_jsonl(metrics_dir / "train.jsonl")
    step_rows = load_jsonl(metrics_dir / "train_steps.jsonl")
    val_rows = load_jsonl(metrics_dir / "val.jsonl")
    checkpoints = sorted(path.name for path in (run_root / "checkpoints").glob("*.pt"))
    latest_train = train_rows[-1] if train_rows else None
    latest_step = step_rows[-1] if step_rows else None
    latest_eval = val_rows[-1] if val_rows else None
    last_updated = None
    if latest_eval and latest_eval.get("timestamp_utc"):
        last_updated = latest_eval["timestamp_utc"]
    elif latest_step and latest_step.get("timestamp_utc"):
        last_updated = latest_step["timestamp_utc"]
    elif latest_train and latest_train.get("timestamp_utc"):
        last_updated = latest_train["timestamp_utc"]
    series = [
        {
            "images_seen": float(row.get("images_seen", 0.0)),
            "median_km": float(row.get("median_km", 0.0)),
            "mean_km": float(row.get("mean_km", 0.0)),
            "p90_km": float(row.get("p90_km", 0.0)),
            "trigger": row.get("trigger", "unknown"),
        }
        for row in val_rows
        if "images_seen" in row
    ]
    latest_images_seen = 0
    if latest_step and latest_step.get("images_seen") is not None:
        latest_images_seen = int(float(latest_step["images_seen"]))
    elif latest_train and latest_train.get("images_seen") is not None:
        latest_images_seen = int(float(latest_train["images_seen"]))
    elif latest_eval and latest_eval.get("images_seen") is not None:
        latest_images_seen = int(float(latest_eval["images_seen"]))
    return {
        "run_root": str(run_root),
        "latest_train": latest_train,
        "latest_step": latest_step,
        "latest_eval": latest_eval,
        "latest_images_seen": latest_images_seen,
        "checkpoints": checkpoints[-12:],
        "series": series,
        "last_updated": last_updated,
    }


def make_handler(run_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, body: bytes, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send_bytes(HTML_PAGE.encode("utf-8"), content_type="text/html; charset=utf-8")
                return
            if self.path == "/api/status":
                payload = json.dumps(build_status(run_root), sort_keys=True).encode("utf-8")
                self._send_bytes(payload, content_type="application/json; charset=utf-8")
                return
            self._send_bytes(b"not found", content_type="text/plain; charset=utf-8", status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a live training dashboard for a run directory.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(run_root))
    url = f"http://{args.host}:{args.port}/"
    print(url, flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
