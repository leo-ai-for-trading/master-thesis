# Master Thesis Polymarket Quote Collector

Read-only Python 3.11+ research package for continuously collecting live Polymarket BTC 5-minute Up/Down quotes. It uses raw HTTP via `httpx` and an optional public market WebSocket trigger path. No wallet authentication is required and no orders are placed.

## What It Does

- Resolves the current BTC 5-minute market from CLOB server time.
- Falls back across current, previous, and next slugs.
- Fetches YES and NO token books and derives best bid, best ask, midpoint, and spread from those books.
- Logs raw Polymarket `/book` responses with full `bids` and `asks` arrays to `data/order_books_YYYYMMDD.jsonl`.
- Appends raw order book rows to `data/order_books_YYYYMMDD.csv`.
- Logs one JSON line per snapshot to `data/quotes_YYYYMMDD.jsonl`.
- Appends flattened rows to `data/quotes_YYYYMMDD.csv`.
- Computes rolling research features and placeholder mean-field state.

## Setup

```bash
cd /Users/attiliopittelli/Desktop/Coding/master_thesis
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Commands

```bash
python -m app.main inspect-current
python -m app.main once
python -m app.main stream
python -m app.main stream --poll-interval 0.25
python -m app.main stream --use-ws
python -m app.main simulate-mm
python -m app.main train-rl-mm
python -m app.main graph-mm
python -m app.main graph-mm --open-browser
python -m app.main graph-rl-mm --open-browser
```

The default collection path is polling because it is easier to validate. When `--use-ws` or `USE_WS=true` is set, the service attempts to subscribe to the public market WebSocket and uses incoming updates as a trigger. If that setup fails, it automatically falls back to polling.

`graph-mm` starts a local dashboard that runs a paper market-maker on top of the live quote stream. It remains read-only: no wallet, no auth, no orders. The dashboard plots:

- mark-to-mid PnL
- YES inventory
- NO inventory
- net inventory

`simulate-mm` runs the same environment headlessly and writes `mm_state_YYYYMMDD.jsonl` and `mm_state_YYYYMMDD.csv`.

The paper market-maker uses passive quote levels derived from the public top of book, the live short-horizon quote returns, and the public mean-field proxies already produced by the collector:

- crowd YES depth vs crowd NO depth
- crowd YES skew vs crowd NO skew
- crowd activity

Those proxies shift the simulated fair value, while the inventory term keeps the strategy from leaning too far in one direction, so the net inventory line should stay relatively flat unless the market starts running one way.

## Reinforcement Learning

`train-rl-mm` runs an online Q-learning agent on top of the same paper environment. The action space is a small set of discrete quote profiles:

- lean NO
- neutral
- lean YES
- aggressive
- defensive

The RL loop logs training data continuously while live quotes are being fetched. At each 5-minute market rollover, it also replays the saved transition dataset from `data/rl_transitions_*.csv` and applies extra Q-learning updates on that historical experience, so the agent keeps learning from the backlog it has already collected.

The main CSV outputs are:

- `data/rl_observations_YYYYMMDD.csv`
- `data/rl_transitions_YYYYMMDD.csv`
- `data/rl_mm_state_YYYYMMDD.csv`
- `data/rl_replay_events_YYYYMMDD.csv`

These contain state features, selected action, reward terms, next-state features, and realized paper PnL / inventory so you can reuse them later for offline training, analysis, or thesis tables.

`graph-rl-mm` runs the same RL loop but keeps the local PnL / inventory dashboard open while the CSV dataset is being built.

Replay tuning is available from the CLI if you want to make the offline pass larger or smaller:

```bash
python -m app.main train-rl-mm --replay --replay-epochs 5 --replay-sample-size 4000
python -m app.main train-rl-mm --no-replay
```

## Environment Overrides

```bash
export BASE_SLUG_PREFIX=btc-updown-5m
export WINDOW_SECONDS=300
export POLL_INTERVAL_SECONDS=0.5
export SERVER_TIME_REFRESH_SECONDS=3.0
export USE_WS=false
```

## Sample Output

`inspect-current` prints a compact live summary:

```text
Server time UTC: 2026-04-20 18:12:04+00:00
Current window start UTC: 2026-04-20 18:10:00+00:00
Current window end UTC: 2026-04-20 18:15:00+00:00
Resolved slug: btc-updown-5m-1776708600
Condition id: 0x...
YES token (Up): 2594...
NO token (Down): 4092...
Top of book YES: bid=0.28 ask=0.29 bid_size_1=146.79 ask_size_1=260.0
Top of book NO: bid=0.71 ask=0.72 bid_size_1=260.0 ask_size_1=146.79
Midpoint YES: 0.285
Midpoint NO: 0.715
Seconds remaining: 176.421
```

`once` prints two raw Polymarket order book payloads, one for YES and one for NO. A JSONL line in `order_books_YYYYMMDD.jsonl` looks like:

```json
{"market":"0x...","asset_id":"2594...","timestamp":"1776708724432","hash":"17ab56...","bids":[{"price":"0.28","size":"146.79"}],"asks":[{"price":"0.29","size":"260.0"}],"min_order_size":"1","tick_size":"0.01","neg_risk":false,"last_trade_price":"0.51"}
```

The normalized research view is still written separately to `quotes_YYYYMMDD.jsonl` and `quotes_YYYYMMDD.csv` for the feature engine, paper market-maker, and RL pipeline.

## Research Notes

- This package is strictly for read-only data collection and research.
- It uses timezone-aware UTC datetimes internally.
- `displayed_price` follows Polymarket’s midpoint-versus-last-trade rule.
- The NO-side last trade is inferred as `1 - YES lastTradePrice` when Gamma only exposes a single market-level last-trade value for the binary market.
