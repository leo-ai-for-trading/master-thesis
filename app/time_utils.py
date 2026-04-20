from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def epoch_seconds_to_utc(epoch_seconds: int | float) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)


def window_start_from_server_ts(server_ts: int | float, window_seconds: int = 300) -> int:
    return int(server_ts // window_seconds) * window_seconds


def window_end_from_start(window_start: int, window_seconds: int = 300) -> int:
    return window_start + window_seconds


def build_market_slug(base_slug_prefix: str, window_start: int) -> str:
    return f"{base_slug_prefix}-{window_start}"


def candidate_window_starts(server_ts: int | float, window_seconds: int = 300) -> list[int]:
    current = window_start_from_server_ts(server_ts, window_seconds=window_seconds)
    return [current, current - window_seconds, current + window_seconds]


def seconds_to_window_end(server_ts: int | float, window_end: int) -> float:
    return max(0.0, float(window_end) - float(server_ts))
