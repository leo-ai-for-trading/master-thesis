from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.models import ResolvedMarket
from app.polymarket_client import PolymarketClient
from app.time_utils import build_market_slug, candidate_window_starts, window_end_from_start


class MarketResolutionError(RuntimeError):
    """Raised when the current market cannot be resolved."""


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_string_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [
            part.strip().strip('"').strip("'")
            for part in text.split(",")
            if part.strip().strip('"').strip("'")
        ]
    return []


def parse_clob_token_ids(raw_value: Any) -> list[str]:
    return parse_string_list(raw_value)


def _extract_outcomes_from_clob(clob_market_payload: dict[str, Any]) -> list[str]:
    outcomes: list[str] = []
    for entry in clob_market_payload.get("t", []):
        outcome = str(entry.get("o", "")).strip()
        if outcome:
            outcomes.append(outcome)
    return outcomes


def _extract_token_ids_from_clob(clob_market_payload: dict[str, Any]) -> list[str]:
    token_ids: list[str] = []
    for entry in clob_market_payload.get("t", []):
        token_id = str(entry.get("t", "")).strip()
        if token_id:
            token_ids.append(token_id)
    return token_ids


def _infer_no_last_trade(last_trade_yes: float | None) -> float | None:
    if last_trade_yes is None:
        return None
    return round(max(0.0, min(1.0, 1.0 - last_trade_yes)), 10)


def build_resolved_market(
    *,
    slug: str,
    window_start: int,
    window_seconds: int,
    market_payload: dict[str, Any],
    clob_market_payload: dict[str, Any],
    event_payload: dict[str, Any] | None = None,
) -> ResolvedMarket:
    outcomes = parse_string_list(market_payload.get("outcomes"))
    if len(outcomes) < 2:
        outcomes = _extract_outcomes_from_clob(clob_market_payload)
    if len(outcomes) < 2:
        outcomes = ["YES", "NO"]

    token_ids = parse_clob_token_ids(market_payload.get("clobTokenIds"))
    if len(token_ids) < 2:
        token_ids = _extract_token_ids_from_clob(clob_market_payload)
    if len(token_ids) < 2:
        raise MarketResolutionError(f"unable to parse token ids for slug {slug}")

    question = str(
        market_payload.get("question")
        or market_payload.get("title")
        or (event_payload or {}).get("title")
        or slug
    )
    title = str(
        market_payload.get("title")
        or (event_payload or {}).get("title")
        or question
    )

    condition_id = str(
        market_payload.get("conditionId")
        or clob_market_payload.get("c")
        or ""
    )
    if not condition_id:
        raise MarketResolutionError(f"missing condition id for slug {slug}")

    tick_size = _to_float(
        market_payload.get("orderPriceMinTickSize", clob_market_payload.get("mts"))
    )
    min_order_size = _to_float(
        market_payload.get("orderMinSize", clob_market_payload.get("mos"))
    )
    last_trade_yes = _to_float(market_payload.get("lastTradePrice"))

    return ResolvedMarket(
        slug=slug,
        window_start=window_start,
        window_end=window_end_from_start(window_start, window_seconds=window_seconds),
        condition_id=condition_id,
        token_yes=token_ids[0],
        token_no=token_ids[1],
        tick_size=tick_size,
        min_order_size=min_order_size,
        title=title,
        question=question,
        yes_label=outcomes[0],
        no_label=outcomes[1],
        last_trade_price_yes=last_trade_yes,
        last_trade_price_no=_infer_no_last_trade(last_trade_yes),
    )


async def resolve_current_market(
    client: PolymarketClient,
    settings: Settings,
    *,
    server_ts: int | None = None,
) -> ResolvedMarket:
    if server_ts is None:
        server_ts = await client.get_server_time()

    for window_start in candidate_window_starts(
        server_ts,
        window_seconds=settings.window_seconds,
    ):
        slug = build_market_slug(settings.base_slug_prefix, window_start)
        market_payload = await client.get_market_by_slug(slug)
        if market_payload is None:
            continue

        condition_id = str(market_payload.get("conditionId", "")).strip()
        if not condition_id:
            raise MarketResolutionError(f"market payload missing condition id for {slug}")

        clob_market_payload = await client.get_clob_market(condition_id)
        event_payload: dict[str, Any] | None = None
        if not market_payload.get("question") and not market_payload.get("title"):
            event_payload = await client.get_event_by_slug(slug)

        return build_resolved_market(
            slug=slug,
            window_start=window_start,
            window_seconds=settings.window_seconds,
            market_payload=market_payload,
            clob_market_payload=clob_market_payload,
            event_payload=event_payload,
        )

    raise MarketResolutionError(
        f"no market found for current/previous/next slugs around server_ts={server_ts}"
    )


async def resolve_market_by_slug(
    client: PolymarketClient,
    settings: Settings,
    *,
    slug: str,
    window_start: int,
) -> ResolvedMarket:
    market_payload = await client.get_market_by_slug(slug)
    if market_payload is None:
        raise MarketResolutionError(f"market not found for slug {slug}")

    condition_id = str(market_payload.get("conditionId", "")).strip()
    if not condition_id:
        raise MarketResolutionError(f"market payload missing condition id for {slug}")

    clob_market_payload = await client.get_clob_market(condition_id)
    event_payload: dict[str, Any] | None = None
    if not market_payload.get("question") and not market_payload.get("title"):
        event_payload = await client.get_event_by_slug(slug)

    return build_resolved_market(
        slug=slug,
        window_start=window_start,
        window_seconds=settings.window_seconds,
        market_payload=market_payload,
        clob_market_payload=clob_market_payload,
        event_payload=event_payload,
    )
