from __future__ import annotations

import asyncio
import csv
import json
import logging
import socket
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from app.config import Settings
from app.models import PaperMMPoint
from app.paper_market_maker import PaperMarketMaker, PaperMarketMakerConfig
from app.polymarket_client import PolymarketClient
from app.quote_service import QuoteService

_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
      <title>Mean-Field Market Maker Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #131a2c;
      --line: #293350;
      --text: #ebf0ff;
      --muted: #98a3c4;
      --green: #24c18a;
      --red: #ff6b6b;
      --blue: #5da8ff;
      --yellow: #ffd166;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.1;
    }
    p {
      margin: 0;
      color: var(--muted);
    }
    .stats {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      margin: 20px 0;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
      min-height: 82px;
    }
    .stat-label {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
    }
    .stat-value {
      margin-top: 10px;
      font-size: 24px;
      font-variant-numeric: tabular-nums;
      overflow-wrap: anywhere;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      margin-bottom: 14px;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 12px;
    }
    .panel-head h2 {
      margin: 0;
      font-size: 18px;
    }
    .legend {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: middle;
      background: currentColor;
    }
    canvas {
      width: 100%;
      height: 280px;
      display: block;
    }
    .status {
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Mean-Field Paper Market Maker</h1>
      <p>Read-only live simulation driven by Polymarket quotes and public crowd-depth proxies. PnL is marked to midpoint and inventory is skew-controlled toward flat.</p>
    </header>
    <section class="stats" id="stats"></section>
    <section class="panel">
      <div class="panel-head">
        <h2>PnL</h2>
        <div class="legend"><span style="color:var(--green)">Mark-to-market PnL</span></div>
      </div>
      <canvas id="pnl-chart"></canvas>
    </section>
    <section class="panel">
      <div class="panel-head">
        <h2>Inventory</h2>
        <div class="legend">
          <span style="color:var(--blue)">YES inventory</span>
          <span style="color:var(--yellow)">NO inventory</span>
          <span style="color:var(--red)">Net inventory</span>
        </div>
      </div>
      <canvas id="inventory-chart"></canvas>
    </section>
    <section class="panel" id="rl-panel">
      <div class="panel-head">
        <h2>RL Exploration</h2>
        <div class="legend">
          <span style="color:var(--yellow)">Epsilon</span>
          <span style="color:var(--blue)">Position bias</span>
        </div>
      </div>
      <canvas id="epsilon-chart"></canvas>
    </section>
    <section class="panel" id="rl-action-panel">
      <div class="panel-head">
        <h2>RL Actions</h2>
        <div class="legend">
          <span style="color:var(--green)">Policy action index</span>
        </div>
      </div>
      <canvas id="action-chart"></canvas>
      <div class="status" id="status">Waiting for data…</div>
    </section>
  </main>
  <script>
    const statsEl = document.getElementById("stats");
    const statusEl = document.getElementById("status");
    const pnlCanvas = document.getElementById("pnl-chart");
    const invCanvas = document.getElementById("inventory-chart");
    const epsilonCanvas = document.getElementById("epsilon-chart");
    const actionCanvas = document.getElementById("action-chart");
    const rlPanel = document.getElementById("rl-panel");
    const rlActionPanel = document.getElementById("rl-action-panel");
    const ACTION_TO_INDEX = {
      lean_no: -2,
      defensive: -1,
      neutral: 0,
      aggressive: 1,
      lean_yes: 2,
    };

    function formatNumber(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
      return Number(value).toFixed(digits);
    }

    function isRLMode(points) {
      return points.some(point => point.policy_name === "q_learning" || point.epsilon !== null);
    }

    function renderStats(latest, meta) {
      if (!latest) {
        statsEl.innerHTML = "";
        return;
      }
      const items = [
        ["Slug", latest.slug],
        ["PnL", formatNumber(latest.mark_to_market_pnl, 4)],
        ["Net Inventory", formatNumber(latest.net_inventory, 2)],
        ["YES Inventory", formatNumber(latest.inventory_yes, 2)],
        ["NO Inventory", formatNumber(latest.inventory_no, 2)],
        ["Seconds To End", formatNumber(latest.seconds_to_window_end, 2)],
        ["Total Fills", String(latest.total_fills)],
        ["Crowd Signal", formatNumber(latest.crowd_signal, 3)],
        ["Action", latest.action_name || "n/a"],
        ["Epsilon", formatNumber(latest.epsilon, 3)],
        ["Inventory Cap", formatNumber(meta.inventory_cap, 2)],
      ];
      statsEl.innerHTML = items.map(([label, value]) => `
        <article class="stat">
          <div class="stat-label">${label}</div>
          <div class="stat-value">${value}</div>
        </article>
      `).join("");
    }

    function resizeCanvas(canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(300, Math.floor(rect.width * dpr));
      const height = Math.max(220, Math.floor(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      return { ctx: canvas.getContext("2d"), width, height, dpr };
    }

    function drawChart(canvas, series, opts = {}) {
      const { ctx, width, height, dpr } = resizeCanvas(canvas);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#131a2c";
      ctx.fillRect(0, 0, width, height);
      if (!series.length || !series.some(item => item.values.length)) {
        ctx.fillStyle = "#98a3c4";
        ctx.font = `${12 * dpr}px sans-serif`;
        ctx.fillText("Waiting for data…", 16 * dpr, 24 * dpr);
        return;
      }

      const pad = { left: 56 * dpr, right: 18 * dpr, top: 14 * dpr, bottom: 32 * dpr };
      const allValues = series.flatMap(item => item.values).filter(value => value !== null && value !== undefined);
      const zeroValues = opts.extraLines || [];
      const minValue = Math.min(...allValues, ...zeroValues, opts.minValue ?? Infinity);
      const maxValue = Math.max(...allValues, ...zeroValues, opts.maxValue ?? -Infinity);
      const span = Math.max(1e-6, maxValue - minValue);
      const innerWidth = width - pad.left - pad.right;
      const innerHeight = height - pad.top - pad.bottom;
      const count = Math.max(...series.map(item => item.values.length));

      const xFor = index => pad.left + (count <= 1 ? innerWidth / 2 : (innerWidth * index) / (count - 1));
      const yFor = value => pad.top + innerHeight - ((value - minValue) / span) * innerHeight;

      ctx.strokeStyle = "#293350";
      ctx.lineWidth = 1 * dpr;
      for (let i = 0; i <= 4; i += 1) {
        const y = pad.top + (innerHeight * i) / 4;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();
      }

      if (opts.extraLines) {
        ctx.setLineDash([6 * dpr, 4 * dpr]);
        opts.extraLines.forEach(line => {
          ctx.strokeStyle = line.color;
          const y = yFor(line.value);
          ctx.beginPath();
          ctx.moveTo(pad.left, y);
          ctx.lineTo(width - pad.right, y);
          ctx.stroke();
        });
        ctx.setLineDash([]);
      }

      series.forEach(item => {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2 * dpr;
        ctx.beginPath();
        item.values.forEach((value, index) => {
          if (value === null || value === undefined) return;
          const x = xFor(index);
          const y = yFor(value);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });

      ctx.fillStyle = "#98a3c4";
      ctx.font = `${11 * dpr}px sans-serif`;
      const labels = opts.yLabels || [maxValue, (maxValue + minValue) / 2, minValue];
      labels.forEach((value, index) => {
        const y = pad.top + (innerHeight * index) / 2;
        const label = typeof value === "number" ? formatNumber(value, 3) : String(value);
        ctx.fillText(label, 8 * dpr, y + 4 * dpr);
      });

      const latestSeries = series.find(item => item.values.length > 0);
      if (latestSeries) {
        const latestValue = latestSeries.values[latestSeries.values.length - 1];
        ctx.fillText(
          opts.title || "",
          pad.left,
          height - 8 * dpr,
        );
        if (latestValue !== null && latestValue !== undefined) {
          ctx.fillText(
            `latest ${formatNumber(latestValue, 4)}`,
            width - 150 * dpr,
            height - 8 * dpr,
          );
        }
      }
    }

    function render(payload) {
      renderStats(payload.latest, payload.meta);
      if (!payload.latest) {
        statusEl.textContent = "Waiting for data…";
        return;
      }
      statusEl.textContent = `Updated ${payload.latest.ts_local} | ${payload.points.length} points`;
      const rlMode = isRLMode(payload.points);
      rlPanel.style.display = rlMode ? "block" : "none";
      rlActionPanel.style.display = rlMode ? "block" : "none";

      drawChart(pnlCanvas, [{
        color: "#24c18a",
        values: payload.points.map(point => point.mark_to_market_pnl),
      }], {
        title: "PnL",
        extraLines: [{ value: 0, color: "#5d698d" }],
      });

      drawChart(invCanvas, [
        { color: "#5da8ff", values: payload.points.map(point => point.inventory_yes) },
        { color: "#ffd166", values: payload.points.map(point => point.inventory_no) },
        { color: "#ff6b6b", values: payload.points.map(point => point.net_inventory) },
      ], {
        title: "Inventory",
        extraLines: [
          { value: 0, color: "#5d698d" },
          { value: payload.meta.inventory_cap, color: "#293350" },
          { value: -payload.meta.inventory_cap, color: "#293350" },
        ],
      });

      if (!rlMode) {
        return;
      }

      drawChart(epsilonCanvas, [
        { color: "#ffd166", values: payload.points.map(point => point.epsilon) },
        { color: "#5da8ff", values: payload.points.map(point => point.position_bias) },
      ], {
        title: "Epsilon / Position Bias",
        minValue: -1,
        maxValue: 1,
        extraLines: [{ value: 0, color: "#5d698d" }],
      });

      drawChart(actionCanvas, [{
        color: "#24c18a",
        values: payload.points.map(point => {
          if (!point.action_name) return null;
          return ACTION_TO_INDEX[point.action_name] ?? null;
        }),
      }], {
        title: "Action Profile",
        minValue: -2,
        maxValue: 2,
        extraLines: [{ value: 0, color: "#5d698d" }],
        yLabels: ["lean_yes", "neutral", "lean_no"],
      });
    }

    async function refresh() {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        if (!response.ok) {
          statusEl.textContent = `HTTP ${response.status}`;
          return;
        }
        const payload = await response.json();
        render(payload);
      } catch (error) {
        statusEl.textContent = String(error);
      }
    }

    window.addEventListener("resize", () => refresh());
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class DashboardStateStore:
    def __init__(self, *, max_points: int, inventory_cap: float) -> None:
        self._points: deque[dict[str, Any]] = deque(maxlen=max_points)
        self._lock = threading.Lock()
        self.inventory_cap = inventory_cap

    def append(self, point: PaperMMPoint) -> None:
        record = point.to_record()
        with self._lock:
            self._points.append(record)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            points = list(self._points)
        return {
            "points": points,
            "latest": points[-1] if points else None,
            "meta": {
                "inventory_cap": self.inventory_cap,
            },
        }


def _pick_available_port(host: str, preferred_port: int, attempts: int = 20) -> int:
    for candidate in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise OSError(f"no free port found near {preferred_port}")


def _handler_factory(store: DashboardStateStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                body = _HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/state":
                payload = json.dumps(store.snapshot(), ensure_ascii=True).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path == "/health":
                payload = json.dumps({"ok": True}, ensure_ascii=True).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


class MarketMakerDashboardServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        store: DashboardStateStore,
    ) -> None:
        actual_port = _pick_available_port(host, port)
        self.host = host
        self.port = actual_port
        self.url = f"http://{host}:{actual_port}"
        self._server = ThreadingHTTPServer((host, actual_port), _handler_factory(store))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


class MarketMakerGraphRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        logger: logging.Logger,
        store: DashboardStateStore,
        mm_config: PaperMarketMakerConfig | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.store = store
        self.paper_mm = PaperMarketMaker(mm_config)

    async def run(self, *, max_iterations: int | None = None) -> None:
        iterations = 0
        async with PolymarketClient(self.settings, logger=self.logger) as client:
            quote_service = QuoteService(self.settings, client, logger=self.logger)
            while True:
                batch = await quote_service.collect_batch(persist=True, source_mode="poll")
                point = self.paper_mm.update(
                    market=batch.market,
                    yes_snapshot=batch.yes_snapshot,
                    no_snapshot=batch.no_snapshot,
                    seconds_to_window_end=batch.features.seconds_to_window_end,
                    mean_field_state=batch.mean_field_state,
                    features=batch.features,
                )
                self.store.append(point)
                self._append_point(point)
                self.logger.info(
                    "updated mean-field paper market maker",
                    extra={
                        "slug": point.slug,
                        "mark_to_market_pnl": round(point.mark_to_market_pnl, 6),
                        "net_inventory": round(point.net_inventory, 6),
                        "gross_inventory": round(point.gross_inventory, 6),
                        "total_fills": point.total_fills,
                        "crowd_signal": None if point.crowd_signal is None else round(point.crowd_signal, 6),
                    },
                )
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                await asyncio.sleep(self.settings.poll_interval_seconds)

    def _append_point(self, point: PaperMMPoint) -> None:
        record = point.to_record()
        date_key = str(record["ts_local"])[:10].replace("-", "")
        data_dir = self.settings.ensure_data_dir()
        jsonl_path = data_dir / f"mm_state_{date_key}.jsonl"
        csv_path = data_dir / f"mm_state_{date_key}.csv"
        with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(record, ensure_ascii=True) + "\n")

        fieldnames = list(record.keys())
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(record)
