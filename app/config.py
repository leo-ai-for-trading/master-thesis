from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


@dataclass(frozen=True, slots=True)
class Settings:
    gamma_base_url: str
    clob_base_url: str
    ws_market_url: str
    base_slug_prefix: str
    window_seconds: int
    poll_interval_seconds: float
    server_time_refresh_seconds: float
    http_timeout_seconds: float
    http_max_retries: int
    http_base_backoff_seconds: float
    use_ws: bool
    data_dir: Path
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parent.parent
        data_dir = Path(os.getenv("DATA_DIR", str(project_root / "data")))
        return cls(
            gamma_base_url=os.getenv("GAMMA_BASE_URL", "https://gamma-api.polymarket.com"),
            clob_base_url=os.getenv("CLOB_BASE_URL", "https://clob.polymarket.com"),
            ws_market_url=os.getenv(
                "WS_MARKET_URL",
                "wss://ws-subscriptions-clob.polymarket.com/ws/market",
            ),
            base_slug_prefix=os.getenv("BASE_SLUG_PREFIX", "btc-updown-5m"),
            window_seconds=int(os.getenv("WINDOW_SECONDS", "300")),
            poll_interval_seconds=float(os.getenv("POLL_INTERVAL_SECONDS", "0.5")),
            server_time_refresh_seconds=float(
                os.getenv("SERVER_TIME_REFRESH_SECONDS", "3.0")
            ),
            http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "10.0")),
            http_max_retries=int(os.getenv("HTTP_MAX_RETRIES", "4")),
            http_base_backoff_seconds=float(os.getenv("HTTP_BASE_BACKOFF_SECONDS", "0.35")),
            use_ws=_env_bool("USE_WS", False),
            data_dir=data_dir,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


_RESERVED_LOG_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=_json_default, ensure_ascii=True)


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("polymarket_research")
    logger.setLevel(level.upper())
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setFormatter(JsonFormatter())

    return logger
