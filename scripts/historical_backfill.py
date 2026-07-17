#!/usr/bin/env python3
"""Backfill historical OHLCV data from Coinbase API.

Fetches historical candles going back as far as Coinbase allows,
storing them in parquet format compatible with the existing pipeline.

This supplements the Jetson-collected tick data with broader history,
which is especially valuable for validating models across more regimes.

Usage:
    python scripts/historical_backfill.py --data-dir data/raw --timeframe 4h --days 365
    python scripts/historical_backfill.py --data-dir data/raw --pairs BTC-USD,ETH-USD
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.influx_client import PRODUCT_IDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill_pair(
    exchange,
    pair: str,
    timeframe: str,
    start_dt: datetime,
    stop_dt: datetime,
    output_dir: Path,
):
    """Fetch OHLCV candles for one pair, paginating backwards."""
    symbol = pair.replace("-", "/")
    safe = pair.replace("-", "_")
    out_path = output_dir / f"ohlcv_{safe}_{timeframe}.parquet"

    if out_path.exists():
        logger.info("  [EXISTS] %s — loading and extending", out_path.name)
        existing = pd.read_parquet(out_path)
        if not existing.empty:
            oldest = existing.index.min()
            if oldest.tzinfo is None:
                oldest = oldest.tz_localize("UTC")
            if oldest <= start_dt:
                logger.info("  [SKIP] Already have data from %s", oldest.date())
                return
    else:
        existing = pd.DataFrame()

    all_candles = []
    cursor_ms = int(start_dt.timestamp() * 1000)
    stop_ms = int(stop_dt.timestamp() * 1000)
    total_fetched = 0

    while cursor_ms < stop_ms:
        try:
            candles = await exchange.fetch_ohlcv(
                symbol, timeframe=timeframe,
                since=cursor_ms, limit=300,
            )
            if not candles:
                break

            all_candles.extend(candles)
            total_fetched += len(candles)
            last_ts = candles[-1][0]
            cursor_ms = last_ts + 1

            if total_fetched % 1000 == 0:
                logger.info("    %s: %d candles fetched so far...", pair, total_fetched)

            # Rate limiting
            await asyncio.sleep(0.35)

        except Exception as e:
            logger.warning("    %s: error at cursor=%d: %s", pair, cursor_ms, str(e)[:80])
            await asyncio.sleep(2)
            continue

    if not all_candles:
        logger.info("  %s: no candles fetched", pair)
        return

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]

    if not existing.empty:
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")]

    df.sort_index(inplace=True)
    df.to_parquet(out_path, engine="pyarrow")
    logger.info("  [DONE] %s: %d candles, %s -> %s, %.1f KB",
                pair, len(df), df.index.min().date(), df.index.max().date(),
                out_path.stat().st_size / 1024)


async def main_async(args):
    import ccxt.async_support as ccxt
    exchange = ccxt.coinbase({"enableRateLimit": True})

    output_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = args.pairs.split(",") if args.pairs else list(PRODUCT_IDS)
    stop_dt = datetime.now(timezone.utc)
    start_dt = stop_dt - timedelta(days=args.days)

    logger.info("Backfilling %d pairs, %s, %d days (%s -> %s)",
                len(pairs), args.timeframe, args.days,
                start_dt.date(), stop_dt.date())

    for i, pair in enumerate(pairs):
        logger.info("[%d/%d] %s", i + 1, len(pairs), pair)
        await backfill_pair(exchange, pair, args.timeframe, start_dt, stop_dt, output_dir)

    await exchange.close()
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Backfill historical OHLCV from Coinbase")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--timeframe", default="4h",
                        help="Candle timeframe (1m, 5m, 1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=365,
                        help="How many days of history to fetch")
    parser.add_argument("--pairs", default=None,
                        help="Comma-separated pairs (default: all)")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
