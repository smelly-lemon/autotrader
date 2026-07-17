#!/usr/bin/env python3
"""Backfill 1-minute OHLCV candles from Coinbase into the SQLite candles table.

Coinbase serves full 1m history via REST, so unlike tick data, candle gaps
are always repairable. Idempotent (INSERT OR REPLACE) — safe to re-run.

Usage:
    python scripts/backfill_candles.py                        # all 12 pairs since 2025-11-05
    python scripts/backfill_candles.py --pairs BTC/USD --since 2026-05-17
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.influx_client import DATA_START, PRODUCT_IDS
from src.data.store import TradeStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PAGE_LIMIT = 300
MAX_RETRIES = 5


async def backfill_pair(exchange, store: TradeStore, symbol: str, since_ms: int) -> int:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    total = 0
    pages = 0
    retries = 0

    while since_ms < now_ms:
        try:
            raw = await exchange.fetch_ohlcv(symbol, "1m", since=since_ms, limit=PAGE_LIMIT)
        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                logger.error("%s: giving up at %s after %d retries: %s",
                             symbol, pd.Timestamp(since_ms, unit="ms", tz="UTC"), MAX_RETRIES, e)
                break
            await asyncio.sleep(5 * retries)
            continue

        retries = 0
        if not raw:
            # No data for this window (delisted period etc.) — skip forward.
            since_ms += PAGE_LIMIT * 60_000
            continue

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        store.upsert_candles(symbol, "1m", df)

        total += len(df)
        pages += 1
        last_ms = int(df.index[-1].timestamp() * 1000)
        if last_ms <= since_ms:  # no forward progress — bail out of the loop
            break
        since_ms = last_ms + 60_000

        if pages % 100 == 0:
            logger.info("  %s: %d pages, %d candles, at %s",
                        symbol, pages, total, str(df.index[-1])[:16])

    return total


async def main_async(args) -> None:
    import ccxt.async_support as ccxt

    pairs = ([p.replace("-", "/") for p in args.pairs.split(",")]
             if args.pairs else [p.replace("-", "/") for p in PRODUCT_IDS])
    since_ms = int(pd.Timestamp(args.since, tz="UTC").timestamp() * 1000)

    store = TradeStore()
    exchange = ccxt.coinbase({"enableRateLimit": True})
    logger.info("Backfilling 1m candles for %d pairs since %s", len(pairs), args.since)

    try:
        for symbol in pairs:
            t0 = datetime.now(timezone.utc)
            n = await backfill_pair(exchange, store, symbol, since_ms)
            count = store.get_candle_count(symbol, "1m")
            logger.info("%s done: +%d fetched, %d total in DB (%.1f min)",
                        symbol, n, count, (datetime.now(timezone.utc) - t0).total_seconds() / 60)
    finally:
        await exchange.close()
        store.close()

    logger.info("BACKFILL COMPLETE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=None,
                        help="Comma-separated pairs (slash or dash form); default: all 12")
    parser.add_argument("--since", default=DATA_START[:10],
                        help="Start date YYYY-MM-DD (default: %(default)s)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
