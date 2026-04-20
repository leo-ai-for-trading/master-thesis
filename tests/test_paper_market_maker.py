from __future__ import annotations

from datetime import datetime, timezone

from app.models import FeatureRow, MeanFieldState, QuoteSnapshot, ResolvedMarket
from app.paper_market_maker import PaperMarketMaker, PaperMarketMakerConfig


def _market(slug: str = "btc-updown-5m-600") -> ResolvedMarket:
    return ResolvedMarket(
        slug=slug,
        window_start=600,
        window_end=900,
        condition_id=f"cond-{slug}",
        token_yes="yes-token",
        token_no="no-token",
        tick_size=0.01,
        min_order_size=5.0,
        title="Bitcoin Up or Down",
        question="Bitcoin Up or Down",
    )


def _snapshot(
    *,
    outcome: str,
    best_bid: float,
    best_ask: float,
    midpoint: float,
    ts_second: int,
    slug: str = "btc-updown-5m-600",
) -> QuoteSnapshot:
    ts = datetime(2026, 4, 20, 18, 10, ts_second, tzinfo=timezone.utc)
    return QuoteSnapshot(
        ts_local=ts,
        ts_server=ts,
        slug=slug,
        condition_id=f"cond-{slug}",
        token_id=f"{outcome.lower()}-token",
        outcome=outcome,
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        spread=round(best_ask - best_bid, 6),
        last_trade_price_if_present=midpoint,
        displayed_price=midpoint,
        bid_size_1=100.0,
        ask_size_1=100.0,
        raw_book_hash_if_present=f"{outcome}-{ts_second}",
        book_timestamp=ts,
    )


def _mean_field_state(
    *,
    crowd_yes_depth: float = 100.0,
    crowd_no_depth: float = 100.0,
    crowd_yes_skew: float = 0.0,
    crowd_no_skew: float = 0.0,
    crowd_activity: float = 100.0,
) -> MeanFieldState:
    return MeanFieldState(
        crowd_yes_depth=crowd_yes_depth,
        crowd_no_depth=crowd_no_depth,
        crowd_yes_skew=crowd_yes_skew,
        crowd_no_skew=crowd_no_skew,
        crowd_activity=crowd_activity,
    )


def _features() -> FeatureRow:
    return FeatureRow(
        yes_mid_return_1s=0.0,
        yes_mid_return_5s=0.0,
        yes_mid_return_10s=0.0,
        no_mid_return_1s=0.0,
        no_mid_return_5s=0.0,
        no_mid_return_10s=0.0,
        yes_spread=0.02,
        no_spread=0.02,
        yes_imbalance=0.0,
        no_imbalance=0.0,
        yes_quote_age_seconds=0.0,
        no_quote_age_seconds=0.0,
        quote_age_seconds=0.0,
        seconds_to_window_end=300.0,
    )


def test_paper_market_maker_buy_fill_updates_inventory_and_pnl() -> None:
    maker = PaperMarketMaker(PaperMarketMakerConfig(order_size=5.0, max_inventory_abs=25.0))
    market = _market()

    first = maker.update(
        market=market,
        yes_snapshot=_snapshot(outcome="YES", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=0),
        no_snapshot=_snapshot(outcome="NO", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=0),
        seconds_to_window_end=300.0,
        mean_field_state=_mean_field_state(),
        features=_features(),
    )
    assert first.inventory_yes == 0.0
    assert first.yes_bid_quote == 0.49

    second = maker.update(
        market=market,
        yes_snapshot=_snapshot(outcome="YES", best_bid=0.50, best_ask=0.49, midpoint=0.50, ts_second=1),
        no_snapshot=_snapshot(outcome="NO", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=1),
        seconds_to_window_end=299.0,
        mean_field_state=_mean_field_state(),
        features=_features(),
    )
    assert second.inventory_yes == 5.0
    assert second.cash == -2.45
    assert round(second.mark_to_market_pnl, 6) == 0.05


def test_positive_inventory_skews_quotes_toward_flattening() -> None:
    maker = PaperMarketMaker(PaperMarketMakerConfig(order_size=5.0, max_inventory_abs=10.0))
    market = _market()

    neutral = maker.update(
        market=market,
        yes_snapshot=_snapshot(outcome="YES", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=0),
        no_snapshot=_snapshot(outcome="NO", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=0),
        seconds_to_window_end=300.0,
        mean_field_state=_mean_field_state(),
        features=_features(),
    )
    maker.update(
        market=market,
        yes_snapshot=_snapshot(outcome="YES", best_bid=0.50, best_ask=0.49, midpoint=0.50, ts_second=1),
        no_snapshot=_snapshot(outcome="NO", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=1),
        seconds_to_window_end=299.0,
        mean_field_state=_mean_field_state(),
        features=_features(),
    )
    skewed = maker.update(
        market=market,
        yes_snapshot=_snapshot(outcome="YES", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=2),
        no_snapshot=_snapshot(outcome="NO", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=2),
        seconds_to_window_end=298.0,
        mean_field_state=_mean_field_state(),
        features=_features(),
    )
    assert skewed.yes_bid_quote is not None and neutral.yes_bid_quote is not None
    assert skewed.no_bid_quote is not None and neutral.no_bid_quote is not None
    assert skewed.yes_bid_quote <= neutral.yes_bid_quote
    assert skewed.no_bid_quote >= neutral.no_bid_quote


def test_rollover_flattens_inventory_at_last_marks() -> None:
    maker = PaperMarketMaker(PaperMarketMakerConfig(order_size=5.0, max_inventory_abs=25.0))
    market = _market()

    maker.update(
        market=market,
        yes_snapshot=_snapshot(outcome="YES", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=0),
        no_snapshot=_snapshot(outcome="NO", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=0),
        seconds_to_window_end=300.0,
        mean_field_state=_mean_field_state(),
        features=_features(),
    )
    maker.update(
        market=market,
        yes_snapshot=_snapshot(outcome="YES", best_bid=0.50, best_ask=0.49, midpoint=0.50, ts_second=1),
        no_snapshot=_snapshot(outcome="NO", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=1),
        seconds_to_window_end=299.0,
        mean_field_state=_mean_field_state(),
        features=_features(),
    )

    rolled = maker.update(
        market=_market(slug="btc-updown-5m-900"),
        yes_snapshot=_snapshot(
            outcome="YES",
            best_bid=0.48,
            best_ask=0.50,
            midpoint=0.49,
            ts_second=2,
            slug="btc-updown-5m-900",
        ),
        no_snapshot=_snapshot(
            outcome="NO",
            best_bid=0.50,
            best_ask=0.52,
            midpoint=0.51,
            ts_second=2,
            slug="btc-updown-5m-900",
        ),
        seconds_to_window_end=300.0,
        mean_field_state=_mean_field_state(),
        features=_features(),
    )
    assert rolled.inventory_yes == 0.0
    assert rolled.inventory_no == 0.0
    assert rolled.realized_on_roll == 2.5


def test_mean_field_signal_pushes_fair_value_toward_yes_when_crowd_is_yes_heavy() -> None:
    maker = PaperMarketMaker(PaperMarketMakerConfig(order_size=5.0, max_inventory_abs=25.0))
    market = _market()

    neutral = maker.update(
        market=market,
        yes_snapshot=_snapshot(outcome="YES", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=0),
        no_snapshot=_snapshot(outcome="NO", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=0),
        seconds_to_window_end=300.0,
        mean_field_state=_mean_field_state(),
        features=_features(),
    )
    signal = maker.update(
        market=market,
        yes_snapshot=_snapshot(outcome="YES", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=1),
        no_snapshot=_snapshot(outcome="NO", best_bid=0.49, best_ask=0.51, midpoint=0.50, ts_second=1),
        seconds_to_window_end=299.0,
        mean_field_state=_mean_field_state(
            crowd_yes_depth=300.0,
            crowd_no_depth=100.0,
            crowd_yes_skew=0.4,
            crowd_no_skew=-0.1,
            crowd_activity=800.0,
        ),
        features=_features(),
    )
    assert neutral.yes_fair_value is not None and signal.yes_fair_value is not None
    assert neutral.no_fair_value is not None and signal.no_fair_value is not None
    assert signal.crowd_signal is not None and signal.crowd_signal > 0.0
    assert signal.yes_fair_value > neutral.yes_fair_value
    assert signal.no_fair_value < neutral.no_fair_value
