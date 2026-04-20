from __future__ import annotations

from datetime import datetime, timezone

from app.models import ResolvedMarket
from app.quote_service import normalize_quote_snapshot


def _market() -> ResolvedMarket:
    return ResolvedMarket(
        slug="btc-updown-5m-600",
        window_start=600,
        window_end=900,
        condition_id="cid-1",
        token_yes="yes-token",
        token_no="no-token",
        tick_size=0.01,
        min_order_size=5.0,
        title="Bitcoin Up or Down",
        question="Bitcoin Up or Down",
        yes_label="Up",
        no_label="Down",
        last_trade_price_yes=0.52,
        last_trade_price_no=0.48,
    )


def test_quote_normalization_uses_midpoint_when_spread_is_tight() -> None:
    snapshot = normalize_quote_snapshot(
        market=_market(),
        token_id="yes-token",
        outcome="YES",
        ts_local=datetime(2026, 4, 20, 18, 10, tzinfo=timezone.utc),
        ts_server=datetime(2026, 4, 20, 18, 10, tzinfo=timezone.utc),
        book={
            "timestamp": "1776708600000",
            "hash": "abc123",
            "bids": [{"price": "0.49", "size": "22"}],
            "asks": [{"price": "0.53", "size": "11"}],
        },
        midpoint_payload={"mid": "0.51"},
        spread_payload={"spread": "0.04"},
        best_buy_payload={"price": "0.49"},
        best_sell_payload={"price": "0.53"},
        last_trade_price=0.52,
        source_mode="poll",
    )
    assert snapshot.displayed_price == 0.51
    assert snapshot.raw_book_hash_if_present == "abc123"
    assert snapshot.bid_size_1 == 22.0
    assert snapshot.ask_size_1 == 11.0


def test_quote_normalization_uses_last_trade_when_spread_is_wide() -> None:
    snapshot = normalize_quote_snapshot(
        market=_market(),
        token_id="yes-token",
        outcome="YES",
        ts_local=datetime(2026, 4, 20, 18, 10, tzinfo=timezone.utc),
        ts_server=datetime(2026, 4, 20, 18, 10, tzinfo=timezone.utc),
        book={
            "timestamp": "1776708600000",
            "hash": "wide-spread",
            "bids": [{"price": "0.40", "size": "10"}],
            "asks": [{"price": "0.60", "size": "12"}],
        },
        midpoint_payload={"mid": "0.50"},
        spread_payload={"spread": "0.20"},
        best_buy_payload={"price": "0.40"},
        best_sell_payload={"price": "0.60"},
        last_trade_price=0.52,
        source_mode="poll",
    )
    assert snapshot.displayed_price == 0.52


def test_quote_normalization_extracts_best_levels_from_unsorted_book() -> None:
    snapshot = normalize_quote_snapshot(
        market=_market(),
        token_id="no-token",
        outcome="NO",
        ts_local=datetime(2026, 4, 20, 18, 10, tzinfo=timezone.utc),
        ts_server=datetime(2026, 4, 20, 18, 10, tzinfo=timezone.utc),
        book={
            "timestamp": "1776708600500",
            "hash": "book-1",
            "bids": [
                {"price": "0.12", "size": "7"},
                {"price": "0.19", "size": "5"},
                {"price": "0.15", "size": "4"},
            ],
            "asks": [
                {"price": "0.97", "size": "8"},
                {"price": "0.81", "size": "3"},
                {"price": "0.84", "size": "6"},
            ],
        },
        midpoint_payload={},
        spread_payload={},
        best_buy_payload={},
        best_sell_payload={},
        last_trade_price=None,
        source_mode="poll",
    )
    assert snapshot.best_bid == 0.19
    assert snapshot.best_ask == 0.81
    assert snapshot.bid_size_1 == 5.0
    assert snapshot.ask_size_1 == 3.0
    assert snapshot.midpoint == 0.5
    assert round(snapshot.spread or 0.0, 2) == 0.62
