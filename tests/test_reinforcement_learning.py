from __future__ import annotations

import csv
from datetime import datetime, timezone

from app.models import PaperMMPoint
from app.reinforcement_learning import (
    HistoricalTransition,
    QLearningAgent,
    QLearningConfig,
    compute_reward,
    discretize_state,
    load_historical_transitions,
    replay_train,
)


def _point(
    *,
    ts_second: int,
    slug: str = "btc-updown-5m-600",
    pnl: float = 0.0,
    net_inventory: float = 0.0,
    total_fills: int = 0,
) -> PaperMMPoint:
    ts = datetime(2026, 4, 20, 18, 10, ts_second, tzinfo=timezone.utc)
    return PaperMMPoint(
        ts_local=ts,
        ts_server=ts,
        slug=slug,
        condition_id=f"cond-{slug}",
        inventory_yes=max(0.0, net_inventory),
        inventory_no=max(0.0, -net_inventory),
        net_inventory=net_inventory,
        gross_inventory=abs(net_inventory),
        cash=0.0,
        inventory_value=pnl,
        mark_to_market_pnl=pnl,
        yes_mid=0.5,
        no_mid=0.5,
        yes_fair_value=0.5,
        no_fair_value=0.5,
        yes_bid_quote=0.49,
        yes_ask_quote=0.51,
        no_bid_quote=0.49,
        no_ask_quote=0.51,
        fill_count_yes_buy=total_fills,
        fill_count_yes_sell=0,
        fill_count_no_buy=0,
        fill_count_no_sell=0,
        total_fills=total_fills,
        seconds_to_window_end=100.0,
        position_bias=0.0,
        crowd_activity=500.0,
        crowd_signal=0.1,
        realized_on_roll=0.0,
    )


def test_discretize_state_is_stable_for_same_inputs() -> None:
    features = {
        "yes_mid": 0.63,
        "yes_spread": 0.02,
        "no_spread": 0.03,
        "crowd_signal": -0.35,
        "net_inventory": 6.0,
        "seconds_to_window_end": 45.0,
        "yes_mid_return_1s": 0.01,
        "no_mid_return_1s": -0.01,
        "crowd_activity": 900.0,
    }
    left = discretize_state(features, max_inventory_abs=25.0)
    right = discretize_state(features, max_inventory_abs=25.0)
    assert left == right


def test_q_learning_update_increases_value_after_positive_reward() -> None:
    agent = QLearningAgent(QLearningConfig(alpha=0.5, gamma=0.9, epsilon=0.0))
    state_key = (1, 0, 0, 0, 0, 0, 0)
    next_state_key = (2, 0, 0, 0, 0, 0, 0)
    before = agent.q_table.setdefault(state_key, [0.0] * 5)[2]
    agent.update(state_key, 2, reward=1.0, next_state_key=next_state_key, done=False)
    after = agent.q_table[state_key][2]
    assert after > before


def test_reward_penalizes_inventory_and_fills() -> None:
    config = QLearningConfig(inventory_penalty=0.1, fill_penalty=0.05, terminal_inventory_penalty=0.2)
    previous = _point(ts_second=0, pnl=0.0, net_inventory=0.0, total_fills=0)
    current = _point(ts_second=1, pnl=1.0, net_inventory=5.0, total_fills=2)
    reward, terms = compute_reward(
        previous,
        current,
        config=config,
        max_inventory_abs=25.0,
    )
    assert terms["pnl_delta"] == 1.0
    assert terms["inventory_penalty"] > 0.0
    assert terms["fill_penalty"] > 0.0
    assert reward < 1.0


def test_terminal_reward_applies_penalty_on_rollover() -> None:
    config = QLearningConfig(terminal_inventory_penalty=0.2)
    previous = _point(ts_second=0, slug="btc-updown-5m-600", pnl=0.4, net_inventory=10.0)
    current = _point(ts_second=1, slug="btc-updown-5m-900", pnl=0.4, net_inventory=0.0)
    reward, terms = compute_reward(
        previous,
        current,
        config=config,
        max_inventory_abs=25.0,
    )
    assert terms["terminal_penalty"] > 0.0
    assert reward < 0.0


def test_load_historical_transitions_parses_saved_csv_rows(tmp_path) -> None:
    path = tmp_path / "rl_transitions_20260420.csv"
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["state_key", "action_index", "reward", "next_state_key", "done"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "state_key": "1|0|0|0|0|0|0",
                "action_index": "2",
                "reward": "0.5",
                "next_state_key": "2|0|0|0|0|0|0",
                "done": "false",
            }
        )
        writer.writerow(
            {
                "state_key": "2|1|0|-1|2|0|1",
                "action_index": "1",
                "reward": "-0.25",
                "next_state_key": "3|1|0|0|2|0|1",
                "done": "true",
            }
        )

    transitions = load_historical_transitions(data_dir=tmp_path, max_rows=10)

    assert len(transitions) == 2
    assert transitions[0].state_key == (1, 0, 0, 0, 0, 0, 0)
    assert transitions[0].done is False
    assert transitions[1].state_key == (2, 1, 0, -1, 2, 0, 1)
    assert transitions[1].done is True


def test_replay_train_updates_q_values_without_decaying_epsilon() -> None:
    agent = QLearningAgent(
        QLearningConfig(
            alpha=0.5,
            gamma=0.0,
            epsilon=0.4,
            replay_epochs=2,
            replay_sample_size=8,
        )
    )
    transitions = [
        HistoricalTransition(
            state_key=(1, 0, 0, 0, 0, 0, 0),
            action_index=2,
            reward=1.0,
            next_state_key=(2, 0, 0, 0, 0, 0, 0),
            done=False,
        )
    ]

    summary = replay_train(agent, transitions=transitions)

    assert summary["replay_updates_applied"] == 2
    assert agent.q_table[(1, 0, 0, 0, 0, 0, 0)][2] > 0.0
    assert agent.epsilon == 0.4
