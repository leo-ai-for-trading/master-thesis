from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timezone
import math
from typing import Any


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def dataclass_to_record(instance: Any) -> dict[str, Any]:
    return {
        field.name: _serialize_value(getattr(instance, field.name))
        for field in fields(instance)
    }


@dataclass(slots=True, frozen=True)
class ResolvedMarket:
    slug: str
    window_start: int
    window_end: int
    condition_id: str
    token_yes: str
    token_no: str
    tick_size: float | None
    min_order_size: float | None
    title: str
    question: str
    yes_label: str = "YES"
    no_label: str = "NO"
    last_trade_price_yes: float | None = None
    last_trade_price_no: float | None = None

    def to_record(self) -> dict[str, Any]:
        return dataclass_to_record(self)


@dataclass(slots=True, frozen=True)
class QuoteSnapshot:
    ts_local: datetime
    ts_server: datetime
    slug: str
    condition_id: str
    token_id: str
    outcome: str
    best_bid: float | None
    best_ask: float | None
    midpoint: float | None
    spread: float | None
    last_trade_price_if_present: float | None
    displayed_price: float | None
    bid_size_1: float | None
    ask_size_1: float | None
    raw_book_hash_if_present: str | None
    book_timestamp: datetime | None = None
    source_mode: str = "poll"

    def to_record(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        record = dataclass_to_record(self)
        if extra:
            for key, value in extra.items():
                record[key] = _serialize_value(value)
        return record


@dataclass(slots=True, frozen=True)
class FeatureRow:
    yes_mid_return_1s: float | None
    yes_mid_return_5s: float | None
    yes_mid_return_10s: float | None
    no_mid_return_1s: float | None
    no_mid_return_5s: float | None
    no_mid_return_10s: float | None
    yes_spread: float | None
    no_spread: float | None
    yes_imbalance: float | None
    no_imbalance: float | None
    yes_quote_age_seconds: float | None
    no_quote_age_seconds: float | None
    quote_age_seconds: float | None
    seconds_to_window_end: float

    def to_record(self) -> dict[str, Any]:
        return dataclass_to_record(self)


@dataclass(slots=True, frozen=True)
class MarketState:
    inventory_yes: float
    inventory_no: float
    cash: float
    seconds_to_end: float
    yes_mid: float | None
    no_mid: float | None
    yes_spread: float | None
    no_spread: float | None
    yes_imbalance: float | None
    no_imbalance: float | None
    external_btc_signal: float | None

    def to_record(self) -> dict[str, Any]:
        return dataclass_to_record(self)


@dataclass(slots=True, frozen=True)
class MeanFieldState:
    crowd_yes_depth: float
    crowd_no_depth: float
    crowd_yes_skew: float
    crowd_no_skew: float
    crowd_activity: float

    def to_record(self) -> dict[str, Any]:
        return dataclass_to_record(self)


@dataclass(slots=True, frozen=True)
class PaperMMPoint:
    ts_local: datetime
    ts_server: datetime
    slug: str
    condition_id: str
    inventory_yes: float
    inventory_no: float
    net_inventory: float
    gross_inventory: float
    cash: float
    inventory_value: float
    mark_to_market_pnl: float
    yes_mid: float | None
    no_mid: float | None
    yes_fair_value: float | None
    no_fair_value: float | None
    yes_bid_quote: float | None
    yes_ask_quote: float | None
    no_bid_quote: float | None
    no_ask_quote: float | None
    fill_count_yes_buy: int
    fill_count_yes_sell: int
    fill_count_no_buy: int
    fill_count_no_sell: int
    total_fills: int
    seconds_to_window_end: float
    position_bias: float
    crowd_activity: float | None
    crowd_signal: float | None
    realized_on_roll: float
    action_name: str | None = None
    policy_name: str | None = None
    epsilon: float | None = None

    def to_record(self) -> dict[str, Any]:
        return dataclass_to_record(self)
