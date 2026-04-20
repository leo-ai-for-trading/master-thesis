from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
import math

from app.models import FeatureRow, MarketState, MeanFieldState, QuoteSnapshot, ResolvedMarket


class FeatureEngine:
    def __init__(self, max_history_seconds: int = 30) -> None:
        self.max_history_seconds = max_history_seconds
        self._history: dict[str, deque[tuple[datetime, float]]] = {
            "YES": deque(),
            "NO": deque(),
        }

    def reset(self) -> None:
        for history in self._history.values():
            history.clear()

    def update(
        self,
        *,
        market: ResolvedMarket,
        yes_snapshot: QuoteSnapshot,
        no_snapshot: QuoteSnapshot,
        ts_server: datetime,
    ) -> tuple[FeatureRow, MarketState, MeanFieldState]:
        yes_imbalance = self._imbalance(yes_snapshot)
        no_imbalance = self._imbalance(no_snapshot)
        yes_quote_age = self._quote_age_seconds(yes_snapshot, ts_server)
        no_quote_age = self._quote_age_seconds(no_snapshot, ts_server)

        features = FeatureRow(
            yes_mid_return_1s=self._compute_return("YES", yes_snapshot.midpoint, ts_server, 1),
            yes_mid_return_5s=self._compute_return("YES", yes_snapshot.midpoint, ts_server, 5),
            yes_mid_return_10s=self._compute_return("YES", yes_snapshot.midpoint, ts_server, 10),
            no_mid_return_1s=self._compute_return("NO", no_snapshot.midpoint, ts_server, 1),
            no_mid_return_5s=self._compute_return("NO", no_snapshot.midpoint, ts_server, 5),
            no_mid_return_10s=self._compute_return("NO", no_snapshot.midpoint, ts_server, 10),
            yes_spread=yes_snapshot.spread,
            no_spread=no_snapshot.spread,
            yes_imbalance=yes_imbalance,
            no_imbalance=no_imbalance,
            yes_quote_age_seconds=yes_quote_age,
            no_quote_age_seconds=no_quote_age,
            quote_age_seconds=self._aggregate_quote_age(yes_quote_age, no_quote_age),
            seconds_to_window_end=max(
                0.0,
                float(market.window_end) - ts_server.timestamp(),
            ),
        )

        self._append_midpoint("YES", ts_server, yes_snapshot.midpoint)
        self._append_midpoint("NO", ts_server, no_snapshot.midpoint)
        self._trim(ts_server)

        market_state = MarketState(
            inventory_yes=0.0,
            inventory_no=0.0,
            cash=0.0,
            seconds_to_end=features.seconds_to_window_end,
            yes_mid=yes_snapshot.midpoint,
            no_mid=no_snapshot.midpoint,
            yes_spread=yes_snapshot.spread,
            no_spread=no_snapshot.spread,
            yes_imbalance=yes_imbalance,
            no_imbalance=no_imbalance,
            external_btc_signal=None,
        )

        crowd_yes_depth = self._depth_proxy(yes_snapshot)
        crowd_no_depth = self._depth_proxy(no_snapshot)
        activity_values = [value for value in [crowd_yes_depth, crowd_no_depth] if value > 0.0]
        mean_field_state = MeanFieldState(
            crowd_yes_depth=crowd_yes_depth if crowd_yes_depth > 0.0 else math.nan,
            crowd_no_depth=crowd_no_depth if crowd_no_depth > 0.0 else math.nan,
            crowd_yes_skew=yes_imbalance if yes_imbalance is not None else math.nan,
            crowd_no_skew=no_imbalance if no_imbalance is not None else math.nan,
            crowd_activity=(
                sum(activity_values) / len(activity_values)
                if activity_values
                else math.nan
            ),
        )
        return features, market_state, mean_field_state

    def _append_midpoint(
        self,
        outcome: str,
        ts_server: datetime,
        midpoint: float | None,
    ) -> None:
        if midpoint is None:
            return
        self._history[outcome].append((ts_server, midpoint))

    def _trim(self, ts_server: datetime) -> None:
        cutoff = ts_server - timedelta(seconds=self.max_history_seconds)
        for history in self._history.values():
            while history and history[0][0] < cutoff:
                history.popleft()

    def _compute_return(
        self,
        outcome: str,
        current_midpoint: float | None,
        ts_server: datetime,
        horizon_seconds: int,
    ) -> float | None:
        if current_midpoint is None or current_midpoint <= 0:
            return None

        target_time = ts_server - timedelta(seconds=horizon_seconds)
        reference_midpoint: float | None = None
        for observed_ts, observed_midpoint in reversed(self._history[outcome]):
            if observed_ts <= target_time:
                reference_midpoint = observed_midpoint
                break
        if reference_midpoint is None or reference_midpoint <= 0:
            return None
        return current_midpoint / reference_midpoint - 1.0

    @staticmethod
    def _imbalance(snapshot: QuoteSnapshot) -> float | None:
        bid = snapshot.bid_size_1
        ask = snapshot.ask_size_1
        if bid is None or ask is None:
            return None
        denominator = bid + ask
        if denominator <= 0:
            return None
        return (bid - ask) / denominator

    @staticmethod
    def _quote_age_seconds(snapshot: QuoteSnapshot, ts_server: datetime) -> float | None:
        if snapshot.book_timestamp is None:
            return None
        delta = ts_server - snapshot.book_timestamp
        return max(0.0, delta.total_seconds())

    @staticmethod
    def _aggregate_quote_age(
        yes_quote_age: float | None,
        no_quote_age: float | None,
    ) -> float | None:
        available = [value for value in [yes_quote_age, no_quote_age] if value is not None]
        if not available:
            return None
        return max(available)

    @staticmethod
    def _depth_proxy(snapshot: QuoteSnapshot) -> float:
        return float((snapshot.bid_size_1 or 0.0) + (snapshot.ask_size_1 or 0.0))
