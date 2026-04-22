from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from typing import Any

from app.config import Settings
from app.market_resolver import resolve_current_market, resolve_market_by_slug
from app.models import ResolvedMarket
from app.polymarket_client import PolymarketClient
from app.time_utils import (
    build_market_slug,
    epoch_seconds_to_utc,
    parse_window_start_from_slug,
    seconds_to_window_end,
    utc_now,
    window_end_from_start,
    window_start_from_server_ts,
)


def _to_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _normalize_levels(levels: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for level in levels or []:
        price = _to_text(level.get("price"))
        size = _to_text(level.get("size"))
        if price is None or size is None:
            continue
        normalized.append({"price": price, "size": size})
    return normalized


def normalize_order_book_payload(book: dict[str, Any]) -> dict[str, Any]:
    return {
        "market": _to_text(book.get("market")),
        "asset_id": _to_text(book.get("asset_id")),
        "timestamp": _to_text(book.get("timestamp")),
        "hash": _to_text(book.get("hash")),
        "bids": _normalize_levels(book.get("bids")),
        "asks": _normalize_levels(book.get("asks")),
        "min_order_size": _to_text(book.get("min_order_size")),
        "tick_size": _to_text(book.get("tick_size")),
        "neg_risk": book.get("neg_risk") if isinstance(book.get("neg_risk"), bool) else None,
        "last_trade_price": _to_text(book.get("last_trade_price")),
    }


class OrderBookCollector:
    def __init__(
        self,
        settings: Settings,
        client: PolymarketClient,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.logger = logger or logging.getLogger("polymarket_research.collector")
        self._market: ResolvedMarket | None = None
        self._anchored_start_window: int | None = None
        self._server_time_anchor: float | None = None
        self._anchor_monotonic: float | None = None
        if self.settings.start_slug is not None:
            self._anchored_start_window = parse_window_start_from_slug(
                self.settings.start_slug,
                self.settings.base_slug_prefix,
            )

    async def inspect_current(self) -> dict[str, Any]:
        server_ts = await self._refresh_server_time(force=True)
        current_window_start = window_start_from_server_ts(
            server_ts,
            window_seconds=self.settings.window_seconds,
        )
        current_window_end = window_end_from_start(
            current_window_start,
            window_seconds=self.settings.window_seconds,
        )
        market = await self._ensure_market(force_server_refresh=False)
        return {
            "server_time": epoch_seconds_to_utc(server_ts),
            "current_window_start": epoch_seconds_to_utc(current_window_start),
            "current_window_end": epoch_seconds_to_utc(current_window_end),
            "resolved_slug": market.slug,
            "condition_id": market.condition_id,
            "token_yes": market.token_yes,
            "token_no": market.token_no,
            "yes_label": market.yes_label,
            "no_label": market.no_label,
            "seconds_remaining": seconds_to_window_end(server_ts, market.window_end),
        }

    async def collect_once(self, *, persist: bool = True) -> list[dict[str, Any]]:
        market = await self._ensure_market(force_server_refresh=True)
        return await self._collect_for_market(market, persist=persist)

    async def stream(self, *, max_iterations: int | None = None) -> None:
        iterations = 0
        while True:
            market = await self._ensure_market(force_server_refresh=False)
            await self._collect_for_market(market, persist=True)

            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                return

            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def _collect_for_market(
        self,
        market: ResolvedMarket,
        *,
        persist: bool,
    ) -> list[dict[str, Any]]:
        yes_book, no_book = await asyncio.gather(
            self.client.get_book(market.token_yes),
            self.client.get_book(market.token_no),
        )
        records = [
            normalize_order_book_payload(yes_book),
            normalize_order_book_payload(no_book),
        ]

        if persist:
            self._append_records(
                records,
                prefix="order_books",
                date_key=utc_now().strftime("%Y%m%d"),
            )

        self.logger.info(
            "collected order book snapshot",
            extra={
                "slug": market.slug,
                "condition_id": market.condition_id,
                "rows": len(records),
            },
        )
        return records

    async def _ensure_market(self, *, force_server_refresh: bool) -> ResolvedMarket:
        server_ts = await self._refresh_server_time(force=force_server_refresh)
        if self.settings.start_slug is not None:
            return await self._ensure_anchored_market(server_ts=server_ts)

        current_slug = build_market_slug(
            self.settings.base_slug_prefix,
            window_start_from_server_ts(server_ts, window_seconds=self.settings.window_seconds),
        )
        if self._market is None or server_ts >= self._market.window_end or self._market.slug != current_slug:
            self._market = await resolve_current_market(
                self.client,
                self.settings,
                server_ts=int(server_ts),
            )
            self.logger.info(
                "resolved market",
                extra={
                    "slug": self._market.slug,
                    "condition_id": self._market.condition_id,
                    "token_yes": self._market.token_yes,
                    "token_no": self._market.token_no,
                },
            )
        return self._market

    async def _ensure_anchored_market(self, *, server_ts: float) -> ResolvedMarket:
        if self._anchored_start_window is None or self.settings.start_slug is None:
            raise RuntimeError("anchored market mode requires a configured start slug")

        target_window_start = (
            self._anchored_start_window
            if self._market is None
            else self._market.window_start
        )
        while server_ts >= target_window_start + self.settings.window_seconds:
            target_window_start += self.settings.window_seconds

        target_slug = build_market_slug(
            self.settings.base_slug_prefix,
            target_window_start,
        )
        if self._market is None or self._market.slug != target_slug:
            self._market = await resolve_market_by_slug(
                self.client,
                self.settings,
                slug=target_slug,
                window_start=target_window_start,
            )
            self.logger.info(
                "resolved anchored market",
                extra={
                    "slug": self._market.slug,
                    "condition_id": self._market.condition_id,
                    "token_yes": self._market.token_yes,
                    "token_no": self._market.token_no,
                    "start_slug": self.settings.start_slug,
                },
            )
        return self._market

    async def _refresh_server_time(self, *, force: bool) -> float:
        now_monotonic = time.monotonic()
        if (
            force
            or self._server_time_anchor is None
            or self._anchor_monotonic is None
            or now_monotonic - self._anchor_monotonic >= self.settings.server_time_refresh_seconds
        ):
            server_ts = float(await self.client.get_server_time())
            self._server_time_anchor = server_ts
            self._anchor_monotonic = now_monotonic
            return server_ts

        return self._server_time_anchor + (now_monotonic - self._anchor_monotonic)

    def _append_records(
        self,
        records: list[dict[str, Any]],
        *,
        prefix: str,
        date_key: str,
    ) -> None:
        if not records:
            return

        data_dir = self.settings.ensure_data_dir()
        jsonl_path = data_dir / f"{prefix}_{date_key}.jsonl"
        csv_path = data_dir / f"{prefix}_{date_key}.csv"

        with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            for record in records:
                jsonl_file.write(json.dumps(record, ensure_ascii=True) + "\n")

        fieldnames = list(records[0].keys())
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for record in records:
                writer.writerow(self._csv_safe_record(record))

    @staticmethod
    def _csv_safe_record(record: dict[str, Any]) -> dict[str, Any]:
        safe_record: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (list, dict)):
                safe_record[key] = json.dumps(value, ensure_ascii=True)
            else:
                safe_record[key] = value
        return safe_record
