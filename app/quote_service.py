from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import websockets

from app.config import Settings
from app.feature_engineering import FeatureEngine
from app.market_resolver import resolve_current_market
from app.models import (
    FeatureRow,
    MeanFieldState,
    MarketState,
    QuoteSnapshot,
    ResolvedMarket,
)
from app.polymarket_client import PolymarketClient
from app.time_utils import (
    build_market_slug,
    epoch_seconds_to_utc,
    seconds_to_window_end,
    utc_now,
    window_end_from_start,
    window_start_from_server_ts,
)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _displayed_price(
    midpoint: float | None,
    spread: float | None,
    last_trade_price: float | None,
) -> float | None:
    if spread is not None and spread <= 0.10 and midpoint is not None:
        return midpoint
    if last_trade_price is not None:
        return last_trade_price
    return midpoint


def _best_level(
    levels: list[dict[str, Any]] | None,
    *,
    side: str,
) -> tuple[float | None, float | None]:
    parsed_levels: list[tuple[float, float | None]] = []
    for level in levels or []:
        price = _to_float(level.get("price"))
        size = _to_float(level.get("size"))
        if price is None:
            continue
        parsed_levels.append((price, size))

    if not parsed_levels:
        return None, None

    if side == "bid":
        price, size = max(parsed_levels, key=lambda item: item[0])
    else:
        price, size = min(parsed_levels, key=lambda item: item[0])
    return price, size


def _book_timestamp_to_utc(raw_timestamp: Any) -> datetime | None:
    timestamp = _to_float(raw_timestamp)
    if timestamp is None:
        return None
    if timestamp >= 1_000_000_000_000:
        timestamp /= 1000.0
    return epoch_seconds_to_utc(timestamp)


def normalize_quote_snapshot(
    *,
    market: ResolvedMarket,
    token_id: str,
    outcome: str,
    ts_local: datetime,
    ts_server: datetime,
    book: dict[str, Any],
    last_trade_price: float | None,
    source_mode: str,
) -> QuoteSnapshot:
    best_bid, book_bid_size = _best_level(book.get("bids"), side="bid")
    best_ask, book_ask_size = _best_level(book.get("asks"), side="ask")
    midpoint = None
    spread = None
    book_timestamp = _book_timestamp_to_utc(book.get("timestamp"))

    if best_bid is not None and best_ask is not None:
        midpoint = round((best_bid + best_ask) / 2.0, 10)
        spread = round(best_ask - best_bid, 10)

    return QuoteSnapshot(
        ts_local=ts_local,
        ts_server=ts_server,
        slug=market.slug,
        condition_id=market.condition_id,
        token_id=token_id,
        outcome=outcome,
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        spread=spread,
        last_trade_price_if_present=last_trade_price,
        displayed_price=_displayed_price(midpoint, spread, last_trade_price),
        bid_size_1=book_bid_size,
        ask_size_1=book_ask_size,
        raw_book_hash_if_present=book.get("hash"),
        book_timestamp=book_timestamp,
        source_mode=source_mode,
    )


def _normalize_raw_levels(levels: list[dict[str, Any]] | None) -> list[dict[str, str]]:
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
        "bids": _normalize_raw_levels(book.get("bids")),
        "asks": _normalize_raw_levels(book.get("asks")),
        "min_order_size": _to_text(book.get("min_order_size")),
        "tick_size": _to_text(book.get("tick_size")),
        "neg_risk": book.get("neg_risk") if isinstance(book.get("neg_risk"), bool) else None,
        "last_trade_price": _to_text(book.get("last_trade_price")),
    }


@dataclass(slots=True)
class RawTokenQuote:
    book: dict[str, Any]


@dataclass(slots=True, frozen=True)
class CollectedQuoteBatch:
    market: ResolvedMarket
    yes_snapshot: QuoteSnapshot
    no_snapshot: QuoteSnapshot
    features: FeatureRow
    market_state: MarketState
    mean_field_state: MeanFieldState
    quote_records: list[dict[str, Any]]
    raw_order_book_records: list[dict[str, Any]]


class MarketWebSocketClient:
    def __init__(self, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("polymarket_research.ws")
        self._connection: Any | None = None
        self._asset_ids: list[str] = []

    @property
    def asset_ids(self) -> list[str]:
        return list(self._asset_ids)

    async def connect(self, asset_ids: list[str]) -> None:
        self._connection = await websockets.connect(
            self.settings.ws_market_url,
            ping_interval=20,
            ping_timeout=20,
            max_size=2_000_000,
        )
        self._asset_ids = list(asset_ids)
        payload = {
            "assets_ids": asset_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }
        await self._connection.send(json.dumps(payload))

    async def wait_for_update(self, timeout_seconds: float) -> dict[str, Any] | None:
        if self._connection is None:
            raise RuntimeError("websocket is not connected")
        raw_message = await asyncio.wait_for(self._connection.recv(), timeout=timeout_seconds)
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        payload = json.loads(raw_message)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload:
            return payload[0]
        return None

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        self._asset_ids = []


class QuoteService:
    def __init__(
        self,
        settings: Settings,
        client: PolymarketClient,
        logger: logging.Logger | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.logger = logger or logging.getLogger("polymarket_research.service")
        self.feature_engine = FeatureEngine()
        self._market: ResolvedMarket | None = None
        self._server_time_anchor: float | None = None
        self._anchor_monotonic: float | None = None

    async def collect_once(self, *, persist: bool = True, source_mode: str = "poll") -> list[dict[str, Any]]:
        batch = await self.collect_batch(persist=persist, source_mode=source_mode)
        return batch.raw_order_book_records

    async def collect_batch(
        self,
        *,
        persist: bool = True,
        source_mode: str = "poll",
    ) -> CollectedQuoteBatch:
        market = await self._ensure_market(force_server_refresh=True)
        return await self._collect_for_market(market, persist=persist, source_mode=source_mode)

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
        batch = await self._collect_for_market(market, persist=False, source_mode="poll")
        yes_record = batch.yes_snapshot.to_record()
        no_record = batch.no_snapshot.to_record()
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
            "yes_top_of_book": {
                "best_bid": yes_record["best_bid"],
                "best_ask": yes_record["best_ask"],
                "bid_size_1": yes_record["bid_size_1"],
                "ask_size_1": yes_record["ask_size_1"],
            },
            "no_top_of_book": {
                "best_bid": no_record["best_bid"],
                "best_ask": no_record["best_ask"],
                "bid_size_1": no_record["bid_size_1"],
                "ask_size_1": no_record["ask_size_1"],
            },
            "yes_midpoint": yes_record["midpoint"],
            "no_midpoint": no_record["midpoint"],
            "seconds_remaining": seconds_to_window_end(server_ts, market.window_end),
        }

    async def stream(self, *, max_iterations: int | None = None) -> None:
        iterations = 0
        websocket_client: MarketWebSocketClient | None = None
        websocket_enabled = self.settings.use_ws

        try:
            while True:
                market = await self._ensure_market(force_server_refresh=False)
                if websocket_enabled:
                    websocket_client = await self._maybe_connect_websocket(
                        websocket_client,
                        market,
                    )

                if websocket_client is None:
                    await self._collect_for_market(market, persist=True, source_mode="poll")
                    await asyncio.sleep(self.settings.poll_interval_seconds)
                else:
                    try:
                        message = await websocket_client.wait_for_update(
                            self.settings.poll_interval_seconds
                        )
                        self.logger.info(
                            "websocket update received",
                            extra={
                                "slug": market.slug,
                                "event_type": None if message is None else message.get("event_type"),
                            },
                        )
                        await self._collect_for_market(market, persist=True, source_mode="ws")
                    except asyncio.TimeoutError:
                        await self._collect_for_market(market, persist=True, source_mode="ws")
                    except Exception as exc:  # pragma: no cover - live fallback path
                        self.logger.warning(
                            "websocket failed; falling back to polling",
                            extra={"slug": market.slug, "error": str(exc)},
                        )
                        await websocket_client.close()
                        websocket_client = None

                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
        finally:
            if websocket_client is not None:
                await websocket_client.close()

    async def _maybe_connect_websocket(
        self,
        websocket_client: MarketWebSocketClient | None,
        market: ResolvedMarket,
    ) -> MarketWebSocketClient | None:
        asset_ids = [market.token_yes, market.token_no]
        if websocket_client is not None and websocket_client.asset_ids == asset_ids:
            return websocket_client
        if websocket_client is not None:
            await websocket_client.close()

        websocket_client = MarketWebSocketClient(self.settings, logger=self.logger)
        try:
            await websocket_client.connect(asset_ids)
            self.logger.info(
                "websocket connected",
                extra={"slug": market.slug, "asset_ids": asset_ids},
            )
            return websocket_client
        except Exception as exc:  # pragma: no cover - live fallback path
            self.logger.warning(
                "websocket setup failed; using polling",
                extra={"slug": market.slug, "error": str(exc)},
            )
            await websocket_client.close()
            return None

    async def _collect_for_market(
        self,
        market: ResolvedMarket,
        *,
        persist: bool,
        source_mode: str,
    ) -> CollectedQuoteBatch:
        ts_local = utc_now()
        ts_server = epoch_seconds_to_utc(await self._refresh_server_time(force=False))

        yes_raw, no_raw = await asyncio.gather(
            self._fetch_token_quote(market.token_yes),
            self._fetch_token_quote(market.token_no),
        )
        yes_snapshot = normalize_quote_snapshot(
            market=market,
            token_id=market.token_yes,
            outcome="YES",
            ts_local=ts_local,
            ts_server=ts_server,
            book=yes_raw.book,
            last_trade_price=market.last_trade_price_yes,
            source_mode=source_mode,
        )
        no_snapshot = normalize_quote_snapshot(
            market=market,
            token_id=market.token_no,
            outcome="NO",
            ts_local=ts_local,
            ts_server=ts_server,
            book=no_raw.book,
            last_trade_price=market.last_trade_price_no,
            source_mode=source_mode,
        )

        features, market_state, mean_field_state = self.feature_engine.update(
            market=market,
            yes_snapshot=yes_snapshot,
            no_snapshot=no_snapshot,
            ts_server=ts_server,
        )
        quote_records = self._build_quote_records(
            market=market,
            yes_snapshot=yes_snapshot,
            no_snapshot=no_snapshot,
            features=features,
            market_state=market_state,
            mean_field_state=mean_field_state,
        )
        raw_order_book_records = self._build_raw_order_book_records(
            yes_book=yes_raw.book,
            no_book=no_raw.book,
        )
        if persist:
            self._append_records(quote_records, prefix="quotes")
            self._append_records(
                raw_order_book_records,
                prefix="order_books",
                date_key=ts_local.strftime("%Y%m%d"),
            )
        self.logger.info(
            "collected quote snapshots",
            extra={
                "slug": market.slug,
                "condition_id": market.condition_id,
                "source_mode": source_mode,
                "quote_rows": len(quote_records),
                "order_book_rows": len(raw_order_book_records),
            },
        )
        return CollectedQuoteBatch(
            market=market,
            yes_snapshot=yes_snapshot,
            no_snapshot=no_snapshot,
            features=features,
            market_state=market_state,
            mean_field_state=mean_field_state,
            quote_records=quote_records,
            raw_order_book_records=raw_order_book_records,
        )

    async def _fetch_token_quote(self, token_id: str) -> RawTokenQuote:
        return RawTokenQuote(book=await self.client.get_book(token_id))

    async def _ensure_market(self, *, force_server_refresh: bool) -> ResolvedMarket:
        server_ts = await self._refresh_server_time(force=force_server_refresh)
        current_slug = build_market_slug(
            self.settings.base_slug_prefix,
            window_start_from_server_ts(server_ts, window_seconds=self.settings.window_seconds),
        )
        if self._market is None or server_ts >= self._market.window_end or self._market.slug != current_slug:
            resolved = await resolve_current_market(
                self.client,
                self.settings,
                server_ts=int(server_ts),
            )
            if self._market is None or self._market.slug != resolved.slug:
                self.feature_engine.reset()
            self._market = resolved
            self.logger.info(
                "resolved market",
                extra={
                    "slug": resolved.slug,
                    "condition_id": resolved.condition_id,
                    "token_yes": resolved.token_yes,
                    "token_no": resolved.token_no,
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

    def _build_quote_records(
        self,
        *,
        market: ResolvedMarket,
        yes_snapshot: QuoteSnapshot,
        no_snapshot: QuoteSnapshot,
        features: Any,
        market_state: MarketState,
        mean_field_state: MeanFieldState,
    ) -> list[dict[str, Any]]:
        extra = {
            "window_start": market.window_start,
            "window_end": market.window_end,
            "market_title": market.title,
            "market_question": market.question,
            "yes_label": market.yes_label,
            "no_label": market.no_label,
        }
        extra.update(features.to_record())
        extra.update(market_state.to_record())
        extra.update(mean_field_state.to_record())
        return [
            yes_snapshot.to_record(extra=extra),
            no_snapshot.to_record(extra=extra),
        ]

    def _build_raw_order_book_records(
        self,
        *,
        yes_book: dict[str, Any],
        no_book: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            normalize_order_book_payload(yes_book),
            normalize_order_book_payload(no_book),
        ]

    def _append_records(
        self,
        records: list[dict[str, Any]],
        *,
        prefix: str,
        date_key: str | None = None,
    ) -> None:
        if not records:
            return
        data_dir = self.settings.ensure_data_dir()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            current_date_key = date_key or str(record["ts_local"])[:10].replace("-", "")
            grouped.setdefault(current_date_key, []).append(record)

        for date_key, day_records in grouped.items():
            jsonl_path = data_dir / f"{prefix}_{date_key}.jsonl"
            csv_path = data_dir / f"{prefix}_{date_key}.csv"
            with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
                for record in day_records:
                    jsonl_file.write(json.dumps(record, ensure_ascii=True) + "\n")

            fieldnames = list(day_records[0].keys())
            write_header = not csv_path.exists() or csv_path.stat().st_size == 0
            with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(self._csv_safe_record(record) for record in day_records)

    @staticmethod
    def _csv_safe_record(record: dict[str, Any]) -> dict[str, Any]:
        safe_record: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (list, dict)):
                safe_record[key] = json.dumps(value, ensure_ascii=True)
            else:
                safe_record[key] = value
        return safe_record
