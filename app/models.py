from __future__ import annotations

from dataclasses import dataclass


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
