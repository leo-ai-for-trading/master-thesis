from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import PaperMMPoint
from app.paper_market_maker import PolicyAction
from app.quote_service import CollectedQuoteBatch, QuoteService
from app.polymarket_client import PolymarketClient
from app.paper_market_maker import PaperMarketMaker, PaperMarketMakerConfig


RL_ACTIONS: tuple[PolicyAction, ...] = (
    PolicyAction(name="lean_no", directional_bias=-0.015, spread_multiplier=1.05, mean_field_multiplier=1.1),
    PolicyAction(name="neutral", directional_bias=0.0, spread_multiplier=1.0, mean_field_multiplier=1.0),
    PolicyAction(name="lean_yes", directional_bias=0.015, spread_multiplier=1.05, mean_field_multiplier=1.1),
    PolicyAction(name="aggressive", directional_bias=0.0, spread_multiplier=0.85, inventory_risk_multiplier=0.8, mean_field_multiplier=1.15),
    PolicyAction(name="defensive", directional_bias=0.0, spread_multiplier=1.3, inventory_risk_multiplier=1.35, mean_field_multiplier=0.9),
)


@dataclass(slots=True, frozen=True)
class QLearningConfig:
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon: float = 0.20
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.995
    inventory_penalty: float = 0.03
    fill_penalty: float = 0.002
    terminal_inventory_penalty: float = 0.08
    random_seed: int = 7
    replay_enabled: bool = True
    replay_epochs: int = 3
    replay_sample_size: int = 2048
    replay_min_rows: int = 256
    replay_max_rows: int = 50_000


@dataclass(slots=True)
class PendingTransition:
    state_features: dict[str, Any]
    state_key: tuple[int, ...]
    action_index: int
    action_name: str
    point: PaperMMPoint
    epsilon: float


@dataclass(slots=True, frozen=True)
class HistoricalTransition:
    state_key: tuple[int, ...]
    action_index: int
    reward: float
    next_state_key: tuple[int, ...]
    done: bool


def build_state_features(point: PaperMMPoint, batch: CollectedQuoteBatch) -> dict[str, Any]:
    mean_field = batch.mean_field_state
    yes_depth = _finite(mean_field.crowd_yes_depth)
    no_depth = _finite(mean_field.crowd_no_depth)
    depth_denominator = yes_depth + no_depth
    depth_signal = 0.0 if depth_denominator <= 0 else (yes_depth - no_depth) / depth_denominator
    crowd_signal = depth_signal + 0.5 * (
        _finite(mean_field.crowd_yes_skew) - _finite(mean_field.crowd_no_skew)
    )
    return {
        "slug": point.slug,
        "condition_id": point.condition_id,
        "yes_mid": point.yes_mid,
        "no_mid": point.no_mid,
        "yes_spread": batch.features.yes_spread,
        "no_spread": batch.features.no_spread,
        "yes_imbalance": batch.features.yes_imbalance,
        "no_imbalance": batch.features.no_imbalance,
        "yes_mid_return_1s": batch.features.yes_mid_return_1s,
        "no_mid_return_1s": batch.features.no_mid_return_1s,
        "seconds_to_window_end": point.seconds_to_window_end,
        "net_inventory": point.net_inventory,
        "gross_inventory": point.gross_inventory,
        "position_bias": point.position_bias,
        "crowd_signal": crowd_signal,
        "crowd_activity": _finite(mean_field.crowd_activity),
        "mark_to_market_pnl": point.mark_to_market_pnl,
        "fill_count": point.total_fills,
    }


def _finite(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def discretize_state(
    state_features: dict[str, Any],
    *,
    max_inventory_abs: float,
) -> tuple[int, ...]:
    yes_mid = max(0.0, min(0.999, _finite(state_features.get("yes_mid"), 0.5)))
    spread = max(
        _finite(state_features.get("yes_spread")),
        _finite(state_features.get("no_spread")),
    )
    crowd_signal = max(-2.0, min(2.0, _finite(state_features.get("crowd_signal"))))
    net_inventory = _finite(state_features.get("net_inventory"))
    seconds_to_end = max(0.0, _finite(state_features.get("seconds_to_window_end")))
    short_return = _finite(state_features.get("yes_mid_return_1s")) - _finite(
        state_features.get("no_mid_return_1s")
    )
    crowd_activity = max(0.0, _finite(state_features.get("crowd_activity")))

    inventory_scale = max(1.0, max_inventory_abs / 4.0)
    inventory_bin = int(max(-4, min(4, round(net_inventory / inventory_scale))))
    seconds_bin = 0 if seconds_to_end > 180 else 1 if seconds_to_end > 60 else 2
    spread_bin = 0 if spread <= 0.02 else 1 if spread <= 0.05 else 2
    return_bin = -1 if short_return < -0.002 else 1 if short_return > 0.002 else 0
    activity_bin = 0 if crowd_activity < 500 else 1 if crowd_activity < 1200 else 2

    return (
        int(yes_mid * 20),
        spread_bin,
        int(round(crowd_signal * 2)),
        inventory_bin,
        seconds_bin,
        return_bin,
        activity_bin,
    )


class QLearningAgent:
    def __init__(self, config: QLearningConfig) -> None:
        self.config = config
        self.epsilon = config.epsilon
        self.random = random.Random(config.random_seed)
        self.q_table: dict[tuple[int, ...], list[float]] = {}

    def select_action(self, state_key: tuple[int, ...]) -> tuple[int, float]:
        values = self.q_table.setdefault(state_key, [0.0] * len(RL_ACTIONS))
        epsilon_used = self.epsilon
        if self.random.random() < self.epsilon:
            action_index = self.random.randrange(len(RL_ACTIONS))
        else:
            action_index = max(range(len(values)), key=lambda idx: values[idx])
        return action_index, epsilon_used

    def update(
        self,
        state_key: tuple[int, ...],
        action_index: int,
        reward: float,
        next_state_key: tuple[int, ...],
        done: bool,
        *,
        decay_epsilon: bool = True,
    ) -> None:
        state_values = self.q_table.setdefault(state_key, [0.0] * len(RL_ACTIONS))
        next_values = self.q_table.setdefault(next_state_key, [0.0] * len(RL_ACTIONS))
        bootstrap = 0.0 if done else self.config.gamma * max(next_values)
        td_target = reward + bootstrap
        td_error = td_target - state_values[action_index]
        state_values[action_index] += self.config.alpha * td_error
        if decay_epsilon:
            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def save_q_table(self, path: Path) -> None:
        serializable = {
            "|".join(str(part) for part in key): values
            for key, values in self.q_table.items()
        }
        path.write_text(json.dumps(serializable, ensure_ascii=True, indent=2), encoding="utf-8")


def compute_reward(
    previous_point: PaperMMPoint,
    current_point: PaperMMPoint,
    *,
    config: QLearningConfig,
    max_inventory_abs: float,
) -> tuple[float, dict[str, float]]:
    pnl_delta = current_point.mark_to_market_pnl - previous_point.mark_to_market_pnl
    inventory_penalty = config.inventory_penalty * abs(current_point.net_inventory) / max(
        1.0,
        max_inventory_abs,
    )
    fills_delta = float(current_point.total_fills - previous_point.total_fills)
    fill_penalty = config.fill_penalty * fills_delta
    terminal_penalty = 0.0
    if current_point.slug != previous_point.slug:
        terminal_penalty = (
            config.terminal_inventory_penalty
            * abs(previous_point.net_inventory)
            / max(1.0, max_inventory_abs)
        )
    reward = pnl_delta - inventory_penalty - fill_penalty - terminal_penalty
    return reward, {
        "pnl_delta": pnl_delta,
        "inventory_penalty": inventory_penalty,
        "fill_penalty": fill_penalty,
        "terminal_penalty": terminal_penalty,
        "fills_delta": fills_delta,
    }


class RLDataLogger:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def append_observation(
        self,
        *,
        point: PaperMMPoint,
        state_features: dict[str, Any],
        state_key: tuple[int, ...],
        action_index: int,
    ) -> None:
        record = point.to_record()
        record.update(
            {
                f"state_{key}": value
                for key, value in state_features.items()
            }
        )
        record["state_key"] = "|".join(str(part) for part in state_key)
        record["action_index"] = action_index
        self._append_csv(record, prefix="rl_observations")

    def append_transition(
        self,
        *,
        ts_local: datetime,
        transition_record: dict[str, Any],
    ) -> None:
        self._append_csv(transition_record, prefix="rl_transitions", ts_local=ts_local)

    def append_replay_event(
        self,
        *,
        ts_local: datetime,
        replay_record: dict[str, Any],
    ) -> None:
        self._append_csv(replay_record, prefix="rl_replay_events", ts_local=ts_local)

    def _append_csv(
        self,
        record: dict[str, Any],
        *,
        prefix: str,
        ts_local: datetime | None = None,
    ) -> None:
        timestamp = ts_local or record["ts_local"]
        date_key = str(timestamp)[:10].replace("-", "")
        data_dir = self.settings.ensure_data_dir()
        path = data_dir / f"{prefix}_{date_key}.csv"
        fieldnames = list(record.keys())
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(record)


def build_transition_record(
    *,
    pending: PendingTransition,
    next_point: PaperMMPoint,
    next_state_features: dict[str, Any],
    next_state_key: tuple[int, ...],
    reward: float,
    reward_terms: dict[str, float],
    done: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ts_local": pending.point.ts_local.isoformat(),
        "ts_server": pending.point.ts_server.isoformat(),
        "slug": pending.point.slug,
        "condition_id": pending.point.condition_id,
        "action_name": pending.action_name,
        "action_index": pending.action_index,
        "epsilon": pending.epsilon,
        "reward": reward,
        "done": done,
        "next_slug": next_point.slug,
        "next_condition_id": next_point.condition_id,
        "next_ts_local": next_point.ts_local.isoformat(),
        "next_ts_server": next_point.ts_server.isoformat(),
        "state_key": "|".join(str(part) for part in pending.state_key),
        "next_state_key": "|".join(str(part) for part in next_state_key),
    }
    for key, value in pending.state_features.items():
        record[f"state_{key}"] = value
    for key, value in next_state_features.items():
        record[f"next_state_{key}"] = value
    record.update(reward_terms)
    return record


def _parse_state_key(raw_value: Any) -> tuple[int, ...]:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return ()
    return tuple(int(part) for part in raw_text.split("|") if part)


def _parse_bool(raw_value: Any) -> bool:
    return str(raw_value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_historical_transitions(*, data_dir: Path, max_rows: int) -> list[HistoricalTransition]:
    if max_rows <= 0:
        return []

    transitions: list[HistoricalTransition] = []
    for path in sorted(data_dir.glob("rl_transitions_*.csv"), reverse=True):
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        for row in reversed(rows):
            try:
                state_key = _parse_state_key(row.get("state_key"))
                next_state_key = _parse_state_key(row.get("next_state_key"))
                action_index = int(row["action_index"])
                if not state_key or not next_state_key:
                    continue
                if action_index < 0 or action_index >= len(RL_ACTIONS):
                    continue
                transitions.append(
                    HistoricalTransition(
                        state_key=state_key,
                        action_index=action_index,
                        reward=float(row["reward"]),
                        next_state_key=next_state_key,
                        done=_parse_bool(row.get("done")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
            if len(transitions) >= max_rows:
                break
        if len(transitions) >= max_rows:
            break
    transitions.reverse()
    return transitions


def replay_train(
    agent: QLearningAgent,
    *,
    transitions: list[HistoricalTransition],
) -> dict[str, int]:
    if not transitions:
        return {
            "replay_available_rows": 0,
            "replay_sample_size": 0,
            "replay_epochs": 0,
            "replay_updates_applied": 0,
        }

    sample_size = min(agent.config.replay_sample_size, len(transitions))
    updates_applied = 0
    epochs = max(0, agent.config.replay_epochs)
    for _ in range(epochs):
        if sample_size == len(transitions):
            sample = list(transitions)
            agent.random.shuffle(sample)
        else:
            sample = agent.random.sample(transitions, sample_size)
        for transition in sample:
            agent.update(
                transition.state_key,
                transition.action_index,
                transition.reward,
                transition.next_state_key,
                transition.done,
                decay_epsilon=False,
            )
            updates_applied += 1

    return {
        "replay_available_rows": len(transitions),
        "replay_sample_size": sample_size,
        "replay_epochs": epochs,
        "replay_updates_applied": updates_applied,
    }


class RLTrainingRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        logger: logging.Logger,
        store: Any | None = None,
        mm_config: PaperMarketMakerConfig | None = None,
        rl_config: QLearningConfig | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.store = store
        self.paper_mm = PaperMarketMaker(mm_config)
        self.agent = QLearningAgent(rl_config or QLearningConfig())
        self.data_logger = RLDataLogger(settings)
        self.pending: PendingTransition | None = None

    async def run(self, *, max_iterations: int | None = None) -> None:
        iterations = 0
        async with PolymarketClient(self.settings, logger=self.logger) as client:
            quote_service = QuoteService(self.settings, client, logger=self.logger)
            while True:
                batch = await quote_service.collect_batch(persist=True, source_mode="poll")
                observed_point = self.paper_mm.observe(
                    market=batch.market,
                    yes_snapshot=batch.yes_snapshot,
                    no_snapshot=batch.no_snapshot,
                    seconds_to_window_end=batch.features.seconds_to_window_end,
                    mean_field_state=batch.mean_field_state,
                    features=batch.features,
                )
                state_features = build_state_features(observed_point, batch)
                state_key = discretize_state(
                    state_features,
                    max_inventory_abs=self.paper_mm.config.max_inventory_abs,
                )

                if self.pending is not None:
                    done = observed_point.slug != self.pending.point.slug
                    reward, reward_terms = compute_reward(
                        self.pending.point,
                        observed_point,
                        config=self.agent.config,
                        max_inventory_abs=self.paper_mm.config.max_inventory_abs,
                    )
                    self.agent.update(
                        self.pending.state_key,
                        self.pending.action_index,
                        reward,
                        state_key,
                        done,
                    )
                    transition = build_transition_record(
                        pending=self.pending,
                        next_point=observed_point,
                        next_state_features=state_features,
                        next_state_key=state_key,
                        reward=reward,
                        reward_terms=reward_terms,
                        done=done,
                    )
                    self.data_logger.append_transition(
                        ts_local=self.pending.point.ts_local,
                        transition_record=transition,
                    )
                    if done:
                        self._run_replay_training(
                            ts_local=observed_point.ts_local,
                            completed_slug=self.pending.point.slug,
                            next_slug=observed_point.slug,
                        )

                action_index, epsilon_used = self.agent.select_action(state_key)
                action = RL_ACTIONS[action_index]
                planned_point = self.paper_mm.set_action(
                    observed_point=observed_point,
                    market=batch.market,
                    yes_snapshot=batch.yes_snapshot,
                    no_snapshot=batch.no_snapshot,
                    seconds_to_window_end=batch.features.seconds_to_window_end,
                    mean_field_state=batch.mean_field_state,
                    features=batch.features,
                    action=action,
                )
                planned_point = replace(
                    planned_point,
                    action_name=action.name,
                    policy_name="q_learning",
                    epsilon=epsilon_used,
                )

                if self.store is not None:
                    self.store.append(planned_point)
                self._append_point(planned_point)
                self.data_logger.append_observation(
                    point=planned_point,
                    state_features=state_features,
                    state_key=state_key,
                    action_index=action_index,
                )
                self.pending = PendingTransition(
                    state_features=state_features,
                    state_key=state_key,
                    action_index=action_index,
                    action_name=action.name,
                    point=planned_point,
                    epsilon=epsilon_used,
                )
                self._save_q_table()
                self.logger.info(
                    "updated rl paper market maker",
                    extra={
                        "slug": planned_point.slug,
                        "action_name": action.name,
                        "epsilon": round(self.agent.epsilon, 6),
                        "mark_to_market_pnl": round(planned_point.mark_to_market_pnl, 6),
                        "net_inventory": round(planned_point.net_inventory, 6),
                        "total_fills": planned_point.total_fills,
                    },
                )
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                await asyncio.sleep(self.settings.poll_interval_seconds)

    def _append_point(self, point: PaperMMPoint) -> None:
        record = point.to_record()
        date_key = str(record["ts_local"])[:10].replace("-", "")
        data_dir = self.settings.ensure_data_dir()
        path = data_dir / f"rl_mm_state_{date_key}.csv"
        fieldnames = list(record.keys())
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    def _save_q_table(self) -> None:
        data_dir = self.settings.ensure_data_dir()
        self.agent.save_q_table(data_dir / "rl_q_table_latest.json")

    def _run_replay_training(
        self,
        *,
        ts_local: datetime,
        completed_slug: str,
        next_slug: str,
    ) -> None:
        if not self.agent.config.replay_enabled:
            return

        data_dir = self.settings.ensure_data_dir()
        transitions = load_historical_transitions(
            data_dir=data_dir,
            max_rows=self.agent.config.replay_max_rows,
        )
        if len(transitions) < self.agent.config.replay_min_rows:
            self.logger.info(
                "skipped rl replay training",
                extra={
                    "completed_slug": completed_slug,
                    "next_slug": next_slug,
                    "replay_available_rows": len(transitions),
                    "replay_min_rows": self.agent.config.replay_min_rows,
                },
            )
            return

        replay_record: dict[str, Any] = {
            "ts_local": ts_local.isoformat(),
            "completed_slug": completed_slug,
            "next_slug": next_slug,
            "epsilon": self.agent.epsilon,
        }
        replay_record.update(
            replay_train(
                self.agent,
                transitions=transitions,
            )
        )
        replay_record["q_table_states"] = len(self.agent.q_table)
        self.data_logger.append_replay_event(
            ts_local=ts_local,
            replay_record=replay_record,
        )
        self.logger.info("completed rl replay training", extra=replay_record)
