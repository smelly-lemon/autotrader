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
| 4 | Exhaustive pre-registered strategy search (`research/`) | Two survivors net of maker fees → now in a paper forward test |

## Current focus

A live two-sleeve **paper forward test** (`scripts/paper_trader.py`, under
launchd) of the only two strategies that survived the exhaustive search:

- **Crash sleeve** — buy 3σ 24h crashes on 66 high-amplitude USD alts, hold
  72h, max 5 positions × 10% equity, contention ranked by a weekly-retrained
  LightGBM triage model. Backtest: +199 bps/trade net (borderline
  significance) — `research/swing_screen_wide_results.json`.
- **Trend sleeve** — Donchian 55d/20d on 8 majors, exit on 20d low or 120d
  cap, max 4 positions × 12.5% equity, implemented as self-healing
  target-state reconciliation. Backtest: +1933 bps/trade net over 10y —
  `research/strategy_battery_results.json`.

$100 starting equity, maker fees 0.6%/side + 0.05% slippage (verified account
tier, post-only assumption), dedicated ledger `paper_trades.db`, every signal
(taken or skipped) logged to the `decisions` table for forward-vs-backtest
comparison. Live status: `logs/paper_status.json`.

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

## Paper trading loop (running 24/7 under launchd)

```bash
# manual run / smoke test
.venv/bin/python -u scripts/paper_trader.py --once
# service
cp deploy/launchd/com.autotrader.papertrader.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.autotrader.papertrader.plist
tail -f logs/paper_trader.log
```

Hourly cycle: refresh 1h candle caches (67 symbols) → crash-sleeve exits due
at entry+72h → new 3σ signals ranked and entered. Daily cycle (00:06 UTC):
refresh daily candles → replay the Donchian rule over full history and trade
toward the replayed target state (restarts, downtime, and first-start
initialization all self-heal). Fills use live ticker prices through
`PaperExecutor` (maker fee + slippage folded into recorded P&L). The legacy
gen-3 loop (`src/main.py`) is retired but kept for reference.

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
