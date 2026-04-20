from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.market_resolver import build_resolved_market, parse_clob_token_ids, resolve_current_market


class FakeClient:
    def __init__(
        self,
        *,
        markets: dict[str, dict],
        clob_markets: dict[str, dict],
        events: dict[str, dict] | None = None,
    ) -> None:
        self.markets = markets
        self.clob_markets = clob_markets
        self.events = events or {}

    async def get_server_time(self) -> int:
        return 600

    async def get_market_by_slug(self, slug: str) -> dict | None:
        return self.markets.get(slug)

    async def get_clob_market(self, condition_id: str) -> dict:
        return self.clob_markets[condition_id]

    async def get_event_by_slug(self, slug: str) -> dict | None:
        return self.events.get(slug)


def _settings() -> Settings:
    return Settings(
        gamma_base_url="https://gamma-api.polymarket.com",
        clob_base_url="https://clob.polymarket.com",
        ws_market_url="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        base_slug_prefix="btc-updown-5m",
        window_seconds=300,
        poll_interval_seconds=0.5,
        server_time_refresh_seconds=3.0,
        http_timeout_seconds=10.0,
        http_max_retries=2,
        http_base_backoff_seconds=0.1,
        use_ws=False,
        data_dir=Path("/tmp/master_thesis_test_data"),
        log_level="INFO",
    )


def _market_payload(slug: str, condition_id: str, token_ids: object) -> dict:
    return {
        "slug": slug,
        "conditionId": condition_id,
        "question": "Bitcoin Up or Down",
        "clobTokenIds": token_ids,
        "outcomes": '["Up", "Down"]',
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "lastTradePrice": 0.61,
    }


def _clob_market(condition_id: str) -> dict:
    return {
        "c": condition_id,
        "t": [
            {"t": "yes-token", "o": "Up"},
            {"t": "no-token", "o": "Down"},
        ],
        "mts": 0.01,
        "mos": 5,
    }


def test_parse_token_ids_from_multiple_formats() -> None:
    assert parse_clob_token_ids('["one", "two"]') == ["one", "two"]
    assert parse_clob_token_ids("one,two") == ["one", "two"]
    assert parse_clob_token_ids(["one", "two"]) == ["one", "two"]


def test_build_resolved_market_uses_clob_fallback_for_token_ids() -> None:
    resolved = build_resolved_market(
        slug="btc-updown-5m-600",
        window_start=600,
        window_seconds=300,
        market_payload=_market_payload("btc-updown-5m-600", "cid-1", ""),
        clob_market_payload=_clob_market("cid-1"),
    )
    assert resolved.token_yes == "yes-token"
    assert resolved.token_no == "no-token"
    assert resolved.last_trade_price_yes == 0.61
    assert resolved.last_trade_price_no == 0.39


def test_resolve_current_market_falls_back_current_previous_then_next() -> None:
    settings = _settings()

    current_slug = "btc-updown-5m-600"
    previous_slug = "btc-updown-5m-300"
    next_slug = "btc-updown-5m-900"

    previous_client = FakeClient(
        markets={previous_slug: _market_payload(previous_slug, "cid-prev", '["a", "b"]')},
        clob_markets={"cid-prev": _clob_market("cid-prev")},
    )
    previous_resolved = asyncio.run(
        resolve_current_market(previous_client, settings, server_ts=600)
    )
    assert previous_resolved.slug == previous_slug

    next_client = FakeClient(
        markets={next_slug: _market_payload(next_slug, "cid-next", ["x", "y"])},
        clob_markets={"cid-next": _clob_market("cid-next")},
    )
    next_resolved = asyncio.run(resolve_current_market(next_client, settings, server_ts=600))
    assert next_resolved.slug == next_slug

    current_client = FakeClient(
        markets={current_slug: _market_payload(current_slug, "cid-current", "p,q")},
        clob_markets={"cid-current": _clob_market("cid-current")},
    )
    current_resolved = asyncio.run(
        resolve_current_market(current_client, settings, server_ts=600)
    )
    assert current_resolved.slug == current_slug
