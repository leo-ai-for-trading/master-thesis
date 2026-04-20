from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import math

from app.models import FeatureRow, MeanFieldState, PaperMMPoint, QuoteSnapshot, ResolvedMarket


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _mark_price(snapshot: QuoteSnapshot) -> float | None:
    if snapshot.midpoint is not None:
        return snapshot.midpoint
    if snapshot.displayed_price is not None:
        return snapshot.displayed_price
    if snapshot.best_bid is not None and snapshot.best_ask is not None:
        return (snapshot.best_bid + snapshot.best_ask) / 2.0
    return snapshot.best_bid or snapshot.best_ask


def _imbalance(snapshot: QuoteSnapshot) -> float:
    bid = snapshot.bid_size_1 or 0.0
    ask = snapshot.ask_size_1 or 0.0
    if bid + ask <= 0:
        return 0.0
    return (bid - ask) / (bid + ask)


def _finite_or_zero(value: float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    return value


@dataclass(slots=True, frozen=True)
class PaperMarketMakerConfig:
    order_size: float = 5.0
    max_inventory_abs: float = 25.0
    inventory_skew: float = 0.035
    endgame_skew: float = 0.05
    imbalance_sensitivity: float = 0.01
    mean_field_sensitivity: float = 0.015
    activity_sensitivity: float = 0.05
    min_half_edge: float = 0.01


@dataclass(slots=True, frozen=True)
class PolicyAction:
    name: str = "neutral"
    directional_bias: float = 0.0
    spread_multiplier: float = 1.0
    inventory_risk_multiplier: float = 1.0
    mean_field_multiplier: float = 1.0


@dataclass(slots=True)
class QuotePlan:
    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None
    yes_fair_value: float | None
    no_fair_value: float | None
    crowd_signal: float
    crowd_activity: float
    action_name: str


class PaperMarketMaker:
    def __init__(self, config: PaperMarketMakerConfig | None = None) -> None:
        self.config = config or PaperMarketMakerConfig()
        self.inventory_yes = 0.0
        self.inventory_no = 0.0
        self.cash = 0.0
        self.fill_count_yes_buy = 0
        self.fill_count_yes_sell = 0
        self.fill_count_no_buy = 0
        self.fill_count_no_sell = 0
        self._active_quotes: QuotePlan | None = None
        self._last_slug: str | None = None
        self._last_condition_id: str | None = None
        self._last_yes_mark: float | None = None
        self._last_no_mark: float | None = None

    def update(
        self,
        *,
        market: ResolvedMarket,
        yes_snapshot: QuoteSnapshot,
        no_snapshot: QuoteSnapshot,
        seconds_to_window_end: float,
        mean_field_state: MeanFieldState,
        features: FeatureRow | None = None,
        action: PolicyAction | None = None,
    ) -> PaperMMPoint:
        observed_point = self.observe(
            market=market,
            yes_snapshot=yes_snapshot,
            no_snapshot=no_snapshot,
            seconds_to_window_end=seconds_to_window_end,
            mean_field_state=mean_field_state,
            features=features,
        )
        return self.set_action(
            observed_point=observed_point,
            market=market,
            yes_snapshot=yes_snapshot,
            no_snapshot=no_snapshot,
            seconds_to_window_end=seconds_to_window_end,
            mean_field_state=mean_field_state,
            features=features,
            action=action,
        )

    def observe(
        self,
        *,
        market: ResolvedMarket,
        yes_snapshot: QuoteSnapshot,
        no_snapshot: QuoteSnapshot,
        seconds_to_window_end: float,
        mean_field_state: MeanFieldState,
        features: FeatureRow | None = None,
    ) -> PaperMMPoint:
        del mean_field_state, features
        realized_on_roll = 0.0
        if self._last_slug is not None and market.slug != self._last_slug:
            realized_on_roll = self._flatten_at_last_marks()
            self._active_quotes = None

        trade_size = max(self.config.order_size, market.min_order_size or 1.0)
        if self._active_quotes is not None and market.slug == self._last_slug:
            self._apply_fills(
                quote_plan=self._active_quotes,
                yes_snapshot=yes_snapshot,
                no_snapshot=no_snapshot,
                trade_size=trade_size,
            )

        yes_mark = _mark_price(yes_snapshot)
        no_mark = _mark_price(no_snapshot)
        active_quotes = self._active_quotes
        position_bias = self._position_bias()

        inventory_value = (
            self.inventory_yes * (yes_mark or 0.0)
            + self.inventory_no * (no_mark or 0.0)
        )
        mark_to_market_pnl = self.cash + inventory_value
        point = PaperMMPoint(
            ts_local=yes_snapshot.ts_local,
            ts_server=yes_snapshot.ts_server,
            slug=market.slug,
            condition_id=market.condition_id,
            inventory_yes=self.inventory_yes,
            inventory_no=self.inventory_no,
            net_inventory=self.inventory_yes - self.inventory_no,
            gross_inventory=abs(self.inventory_yes) + abs(self.inventory_no),
            cash=self.cash,
            inventory_value=inventory_value,
            mark_to_market_pnl=mark_to_market_pnl,
            yes_mid=yes_mark,
            no_mid=no_mark,
            yes_fair_value=None if active_quotes is None else active_quotes.yes_fair_value,
            no_fair_value=None if active_quotes is None else active_quotes.no_fair_value,
            yes_bid_quote=None if active_quotes is None else active_quotes.yes_bid,
            yes_ask_quote=None if active_quotes is None else active_quotes.yes_ask,
            no_bid_quote=None if active_quotes is None else active_quotes.no_bid,
            no_ask_quote=None if active_quotes is None else active_quotes.no_ask,
            fill_count_yes_buy=self.fill_count_yes_buy,
            fill_count_yes_sell=self.fill_count_yes_sell,
            fill_count_no_buy=self.fill_count_no_buy,
            fill_count_no_sell=self.fill_count_no_sell,
            total_fills=(
                self.fill_count_yes_buy
                + self.fill_count_yes_sell
                + self.fill_count_no_buy
                + self.fill_count_no_sell
            ),
            seconds_to_window_end=seconds_to_window_end,
            position_bias=position_bias,
            crowd_activity=None if active_quotes is None else active_quotes.crowd_activity,
            crowd_signal=None if active_quotes is None else active_quotes.crowd_signal,
            action_name=None if active_quotes is None else active_quotes.action_name,
            realized_on_roll=realized_on_roll,
        )

        self._last_slug = market.slug
        self._last_condition_id = market.condition_id
        self._last_yes_mark = yes_mark
        self._last_no_mark = no_mark
        return point

    def set_action(
        self,
        *,
        observed_point: PaperMMPoint,
        market: ResolvedMarket,
        yes_snapshot: QuoteSnapshot,
        no_snapshot: QuoteSnapshot,
        seconds_to_window_end: float,
        mean_field_state: MeanFieldState,
        features: FeatureRow | None = None,
        action: PolicyAction | None = None,
    ) -> PaperMMPoint:
        quote_plan = self._build_quote_plan(
            market=market,
            yes_snapshot=yes_snapshot,
            no_snapshot=no_snapshot,
            seconds_to_window_end=seconds_to_window_end,
            position_bias=self._position_bias(),
            mean_field_state=mean_field_state,
            features=features,
            action=action,
        )
        self._active_quotes = quote_plan
        return replace(
            observed_point,
            yes_fair_value=quote_plan.yes_fair_value,
            no_fair_value=quote_plan.no_fair_value,
            yes_bid_quote=quote_plan.yes_bid,
            yes_ask_quote=quote_plan.yes_ask,
            no_bid_quote=quote_plan.no_bid,
            no_ask_quote=quote_plan.no_ask,
            crowd_activity=quote_plan.crowd_activity,
            crowd_signal=quote_plan.crowd_signal,
            action_name=quote_plan.action_name,
        )

    def _flatten_at_last_marks(self) -> float:
        yes_mark = self._last_yes_mark or 0.0
        no_mark = self._last_no_mark or 0.0
        realized = self.inventory_yes * yes_mark + self.inventory_no * no_mark
        self.cash += realized
        self.inventory_yes = 0.0
        self.inventory_no = 0.0
        return realized

    def _apply_fills(
        self,
        *,
        quote_plan: QuotePlan,
        yes_snapshot: QuoteSnapshot,
        no_snapshot: QuoteSnapshot,
        trade_size: float,
    ) -> None:
        if quote_plan.yes_bid is not None and yes_snapshot.best_ask is not None:
            if yes_snapshot.best_ask <= quote_plan.yes_bid:
                self.inventory_yes += trade_size
                self.cash -= trade_size * quote_plan.yes_bid
                self.fill_count_yes_buy += 1
        if quote_plan.yes_ask is not None and yes_snapshot.best_bid is not None:
            if yes_snapshot.best_bid >= quote_plan.yes_ask:
                self.inventory_yes -= trade_size
                self.cash += trade_size * quote_plan.yes_ask
                self.fill_count_yes_sell += 1
        if quote_plan.no_bid is not None and no_snapshot.best_ask is not None:
            if no_snapshot.best_ask <= quote_plan.no_bid:
                self.inventory_no += trade_size
                self.cash -= trade_size * quote_plan.no_bid
                self.fill_count_no_buy += 1
        if quote_plan.no_ask is not None and no_snapshot.best_bid is not None:
            if no_snapshot.best_bid >= quote_plan.no_ask:
                self.inventory_no -= trade_size
                self.cash += trade_size * quote_plan.no_ask
                self.fill_count_no_sell += 1

    def _build_quote_plan(
        self,
        *,
        market: ResolvedMarket,
        yes_snapshot: QuoteSnapshot,
        no_snapshot: QuoteSnapshot,
        seconds_to_window_end: float,
        position_bias: float,
        mean_field_state: MeanFieldState,
        features: FeatureRow | None,
        action: PolicyAction | None,
    ) -> QuotePlan:
        action = action or PolicyAction()
        tick = market.tick_size or 0.01
        window_seconds = max(1.0, float(market.window_end - market.window_start))
        time_pressure = 1.0 - min(seconds_to_window_end / window_seconds, 1.0)
        crowd_signal = self._crowd_signal(mean_field_state)
        crowd_activity = max(0.0, _finite_or_zero(mean_field_state.crowd_activity))
        activity_scale = 1.0 + min(2.0, crowd_activity / 1_000.0) * self.config.activity_sensitivity
        inventory_pressure = position_bias * (
            self.config.inventory_skew + self.config.endgame_skew * time_pressure
        ) * activity_scale * action.inventory_risk_multiplier
        short_horizon_signal = _finite_or_zero(None if features is None else features.yes_mid_return_1s)
        mean_field_component = self.config.mean_field_sensitivity * action.mean_field_multiplier * crowd_signal

        yes_bid, yes_ask = self._build_side_quotes(
            snapshot=yes_snapshot,
            tick=tick,
            shift=(
                -inventory_pressure
                + self.config.imbalance_sensitivity * _imbalance(yes_snapshot)
                + mean_field_component
                + 0.5 * short_horizon_signal
                + action.directional_bias
            ),
            spread_multiplier=action.spread_multiplier,
        )
        no_bid, no_ask = self._build_side_quotes(
            snapshot=no_snapshot,
            tick=tick,
            shift=(
                inventory_pressure
                + self.config.imbalance_sensitivity * _imbalance(no_snapshot)
                - mean_field_component
                - 0.5 * short_horizon_signal
                - action.directional_bias
            ),
            spread_multiplier=action.spread_multiplier,
        )
        yes_mark = _mark_price(yes_snapshot)
        no_mark = _mark_price(no_snapshot)
        yes_fair_value = None if yes_mark is None else round(
            yes_mark
            - inventory_pressure
            + mean_field_component
            + action.directional_bias,
            6,
        )
        no_fair_value = None if no_mark is None else round(
            no_mark
            + inventory_pressure
            - mean_field_component
            - action.directional_bias,
            6,
        )

        net_inventory = self.inventory_yes - self.inventory_no
        if net_inventory >= self.config.max_inventory_abs:
            yes_bid = None
            no_ask = None
        if net_inventory <= -self.config.max_inventory_abs:
            yes_ask = None
            no_bid = None
        return QuotePlan(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            yes_fair_value=yes_fair_value,
            no_fair_value=no_fair_value,
            crowd_signal=crowd_signal,
            crowd_activity=crowd_activity,
            action_name=action.name,
        )

    def _build_side_quotes(
        self,
        *,
        snapshot: QuoteSnapshot,
        tick: float,
        shift: float,
        spread_multiplier: float,
    ) -> tuple[float | None, float | None]:
        mark_price = _mark_price(snapshot)
        if mark_price is None:
            return None, None

        half_edge = max(
            self.config.min_half_edge,
            tick,
            (snapshot.spread or (2.0 * tick)) / 2.0,
        )
        half_edge *= max(0.5, spread_multiplier)
        bid = mark_price + shift - half_edge
        ask = mark_price + shift + half_edge
        if snapshot.best_bid is not None:
            bid = max(bid, snapshot.best_bid)
        if snapshot.best_ask is not None:
            ask = min(ask, snapshot.best_ask)

        bid = _clamp(bid, tick, 1.0 - tick)
        ask = _clamp(ask, tick, 1.0 - tick)
        if ask <= bid:
            ask = _clamp(max(mark_price + tick, bid + tick), tick, 1.0 - tick)
            bid = _clamp(min(mark_price - tick, ask - tick), tick, 1.0 - tick)

        if ask <= bid:
            return None, None
        return round(bid, 6), round(ask, 6)

    def _position_bias(self) -> float:
        if self.config.max_inventory_abs <= 0:
            return 0.0
        return _clamp(
            (self.inventory_yes - self.inventory_no) / self.config.max_inventory_abs,
            -1.0,
            1.0,
        )

    def _crowd_signal(self, mean_field_state: MeanFieldState) -> float:
        yes_depth = max(0.0, _finite_or_zero(mean_field_state.crowd_yes_depth))
        no_depth = max(0.0, _finite_or_zero(mean_field_state.crowd_no_depth))
        depth_denominator = yes_depth + no_depth
        depth_signal = 0.0
        if depth_denominator > 0:
            depth_signal = (yes_depth - no_depth) / depth_denominator

        skew_signal = 0.5 * (
            _finite_or_zero(mean_field_state.crowd_yes_skew)
            - _finite_or_zero(mean_field_state.crowd_no_skew)
        )
        return depth_signal + skew_signal
