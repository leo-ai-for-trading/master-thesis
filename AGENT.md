# AGENTS.md

## Project Overview

This repository is for a **toy research prototype** of a **reinforcement-learning market-making system** for **Polymarket binary markets**.

The project is for a **supervisor demo**, not for production trading.

The system must:
- ingest or load Polymarket-style order book and trade data,
- build a replay-based simulator,
- train and evaluate a market-making RL agent,
- react to buy/sell pressure by skewing quotes,
- maintain inventory as close to zero as possible.

This project must remain:
- simulation-only,
- paper-trading only,
- easy to explain,
- mathematically interpretable,
- modular enough to extend later.

---

## Main Objective

Build a Python codebase for a replay-based market-making environment where an agent:

1. observes market state from order book and trade flow,
2. chooses a discrete quoting action,
3. receives simulated fills,
4. updates inventory and cash,
5. receives reward based on PnL and inventory control,
6. learns a quoting policy using DQN or Double DQN.

The first version must support:
- replay mode,
- baseline strategies,
- DQN training,
- evaluation and plotting.

---

## Non-Goals

Do **not** build:
- live production trading,
- real order submission,
- exchange authentication and order management,
- high-performance infrastructure,
- multi-agent equilibrium models in version 1,
- full mean-field game solvers in version 1,
- complex latent-state models unless clearly isolated as optional future work.

Do **not** optimize prematurely.

Do **not** hide missing logic behind vague placeholders.

---

## Core Modeling Requirements

### RL framing

Treat the problem as a sequential decision problem with:

- **state**: market and inventory features,
- **action**: discrete quoting decision,
- **reward**: PnL minus inventory/adverse-selection penalties,
- **transition**: replay-based market evolution and simulated fills.

### Market-making objective

The agent should:
- quote passively,
- adapt quote skew to market pressure,
- widen/narrow quotes depending on action,
- avoid accumulating directional inventory,
- stay approximately inventory-neutral through both reward design and hard constraints.

### Inventory philosophy

Inventory must remain **flattish**.

This is mandatory.

Implement both:
- **soft inventory penalties** in the reward,
- **hard inventory constraints** via action masking or side disabling.

---

## Technical Stack

Use:
- Python
- PyTorch
- Gymnasium
- numpy
- pandas
- matplotlib

Allowed if useful:
- pydantic
- dataclasses
- pathlib
- typing
- yaml
- pyarrow / parquet support
- websocket/http libraries for data adapters only

Avoid unnecessary frameworks.

Prefer readable, modular code.

---

## Repository Expectations

If the repository is empty or loosely structured, prefer a layout like:

```text
project_root/
  README.md
  requirements.txt
  configs/
    default.yaml
  data/
    raw/
    processed/
  notebooks/
  scripts/
    collect_polymarket_data.py
    build_replay_dataset.py
    train_dqn.py
    evaluate_agent.py
  src/
    polymarket_mm/
      __init__.py
      config.py
      data/
        adapters.py
        schemas.py
        storage.py
        preprocessing.py
      features/
        market_features.py
      env/
        market_making_env.py
        fill_simulator.py
        action_space.py
        reward.py
      agents/
        replay_buffer.py
        q_network.py
        dqn_agent.py
      baselines/
        symmetric_mm.py
        skew_mm.py
      training/
        trainer.py
        evaluation.py
        metrics.py
      utils/
        logging_utils.py
        plotting.py
        seeds.py
  tests/
    test_action_space.py
    test_reward.py
    test_features.py
    test_env_smoke.py
