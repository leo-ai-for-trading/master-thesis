# Master Thesis Order Book Collector

Minimal Python 3.11+ collector for continuously fetching live Polymarket BTC 5-minute Up/Down order books and saving them to `csv` and `jsonl`.

## What It Does

- Resolves the current BTC 5-minute market from Polymarket server time.
- Falls back across current, previous, and next slugs when needed.
- Calls the public `GET https://clob.polymarket.com/book?token_id=...` endpoint for the active YES and NO tokens.
- Writes the raw `/book` response shape to `data/order_books_YYYYMMDD.jsonl`.
- Writes the same shape to `data/order_books_YYYYMMDD.csv`, with `bids` and `asks` serialized as JSON strings.
- Can anchor collection on a specific event slug and then roll forward by 5 minutes after each expiry.

## Setup

```bash
cd /Users/attiliopittelli/Desktop/Coding/master-thesis
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Commands

```bash
python -m app.main inspect-current
python -m app.main once
python -m app.main stream
python -m app.main stream --poll-interval 0.25
python -m app.main stream --start-slug btc-updown-5m-1776880200
```

## Environment Overrides

```bash
export BASE_SLUG_PREFIX=btc-updown-5m
export START_SLUG=btc-updown-5m-1776880200
export WINDOW_SECONDS=300
export POLL_INTERVAL_SECONDS=0.5
export SERVER_TIME_REFRESH_SECONDS=3.0
```

## Output

Files are written to `data/` using the current UTC date:

- `data/order_books_YYYYMMDD.jsonl`
- `data/order_books_YYYYMMDD.csv`

Each JSONL record matches the `/book` response schema:

```json
{
  "market": "0x1234567890123456789012345678901234567890",
  "asset_id": "0xabc123def456...",
  "timestamp": "1234567890",
  "hash": "a1b2c3d4e5f6...",
  "bids": [
    {
      "price": "0.45",
      "size": "100"
    },
    {
      "price": "0.44",
      "size": "200"
    }
  ],
  "asks": [
    {
      "price": "0.46",
      "size": "150"
    },
    {
      "price": "0.47",
      "size": "250"
    }
  ],
  "min_order_size": "1",
  "tick_size": "0.01",
  "neg_risk": false,
  "last_trade_price": "0.45"
}
```
