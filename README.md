# auto-trader

Crypto trading **research platform** for Coinbase markets. Paper trading only.

Three strategy generations have been built and evaluated here. None demonstrated
a durable, statistically significant edge net of realistic fees — that negative
result is documented in `results/` and is the project's main finding to date.

| Generation | Approach | Verdict |
|---|---|---|
| 1 | Three-tier local LLM council (Ollama) | Abandoned; removed from tree (see git history) |
| 2 | LightGBM on 52 microstructure features, 4h bars | In-sample Sharpe 6.6 collapsed to ~0 edge in expanding-window OOS |
| 3 | LightGBM on 1-min OHLCV features, 15-min horizon | 934 paper trades in 10 days ≈ break-even *before* fees |

## Current focus

Durable data collection first; model claims only after 30+ days of fresh data
and pre-registered validation criteria (expanding-window OOS, net of
starter-tier fees, Sharpe CI excluding zero).

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

## Data collection (runs 24/7 under launchd)

`scripts/live_collector.py` polls Coinbase REST for tickers + trades on 12
pairs and appends daily-sharded parquet to `data/raw/` — same schema as the
legacy Jetson/InfluxDB archive (Nov 2025 → May 2026), which
`src/data/influx_client.py:LocalParquetClient` reads transparently alongside
the new shards.

```bash
cp deploy/launchd/com.autotrader.*.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.autotrader.collector.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.autotrader.heartbeat.plist
tail -f logs/collector.log
```

The heartbeat job checks twice daily that fresh parquet is being written and
raises a macOS notification if the feed is stale (`scripts/heartbeat.py`).

## Paper trading loop (off by default — data first)

```bash
.venv/bin/python -m src.main
```

`src/main.py` runs the gen-3 loop: 1-min candle ingestion into SQLite →
periodic LightGBM retrain with purged walk-forward CV (won't trade below a
minimum CV AUC) → signal generation → `PaperExecutor`, which models
starter-tier taker fees (1.2%) and slippage so results are honest, and
restores open positions from the DB after a restart. Hard risk limits
(position cap, max positions, daily-drawdown halt) are enforced in the loop.

## Research pipeline

- `scripts/model_search_v2.py` — feature/config search with purged
  walk-forward CV, spread + fee modeling, Sharpe confidence intervals, and
  expanding-window OOS validation (`--stream A|B|C`)
- `src/ml/` — feature engineering (microstructure + OHLCV), trainers,
  backtest engine, live/paper runners
- `results/` — all search reports and OOS validations (committed; these are
  the evidence)

## Configuration

`config.yaml` — pairs, schedule, ML thresholds, risk limits, simulated fee
tier. Secrets live in `.env` (see `.env.example`); nothing secret is committed.

## Hard-won operational rules

1. Commit early and often — this repo once lost most of its working tree with
   zero commits to restore from.
2. Long-running processes go under launchd with `KeepAlive`, logs, and a
   heartbeat — silent death cost this project 7 weeks of data.
3. Model claims require expanding-window OOS at realistic fee tiers; a good
   holdout number selected during search is not evidence.
