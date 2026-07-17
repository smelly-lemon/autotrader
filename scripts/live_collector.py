#!/usr/bin/env python3
"""Live Coinbase data collector using ccxt.

Fetches OHLCV + ticker data for all trading pairs at regular intervals,
appending to local parquet files in the same format as jetson_export.py.
This replaces the Jetson InfluxDB collector with a simpler approach
that stores data directly in parquet format.

Usage:
    python scripts/live_collector.py --data-dir data/raw --interval 60
    python scripts/live_collector.py --data-dir data/raw --interval 60 --pairs BTC-USD,ETH-USD
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.influx_client import PRODUCT_IDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class LiveCollector:
    """Collects ticker and trade data from Coinbase, appends to parquet files."""

    def __init__(
        self,
        data_dir: str | Path = "data/raw",
        pairs: list[str] | None = None,
        collect_interval: int = 60,
        flush_interval: int = 300,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pairs = pairs or list(PRODUCT_IDS)
        self.collect_interval = collect_interval
        self.flush_interval = flush_interval
        self._exchange = None
        self._running = False

        # Buffers
        self._ticker_buf: dict[str, list[dict]] = {p: [] for p in self.pairs}
        self._matches_buf: dict[str, list[dict]] = {p: [] for p in self.pairs}
        self._last_flush = time.time()
        self._last_trade_id: dict[str, str] = {}

    async def _get_exchange(self):
        if self._exchange is None:
            import ccxt.async_support as ccxt
            self._exchange = ccxt.coinbase({"enableRateLimit": True})
        return self._exchange

    async def run(self):
        self._running = True
        logger.info("LiveCollector starting")
        logger.info("  Pairs: %s", self.pairs)
        logger.info("  Collect interval: %ds, flush interval: %ds",
                     self.collect_interval, self.flush_interval)
        logger.info("  Data dir: %s", self.data_dir)

        cycle = 0
        while self._running:
            cycle += 1
            t0 = time.time()
            try:
                await self._collect_cycle()
            except Exception:
                logger.exception("Collection cycle %d failed", cycle)

            # Flush periodically
            if time.time() - self._last_flush > self.flush_interval:
                self._flush_buffers()
                self._last_flush = time.time()

            elapsed = time.time() - t0
            sleep_time = max(0, self.collect_interval - elapsed)
            if cycle % 10 == 0:
                total_ticker = sum(len(v) for v in self._ticker_buf.values())
                total_matches = sum(len(v) for v in self._matches_buf.values())
                logger.info("  Cycle %d: collected %d ticker, %d match records (%.1fs)",
                            cycle, total_ticker, total_matches, elapsed)
            await asyncio.sleep(sleep_time)

    async def _collect_cycle(self):
        exchange = await self._get_exchange()

        for pair in self.pairs:
            symbol = pair.replace("-", "/")
            try:
                # Fetch ticker
                ticker = await exchange.fetch_ticker(symbol)
                now = datetime.now(timezone.utc)
                self._ticker_buf[pair].append({
                    "_time": now,
                    "price": ticker.get("last", 0),
                    "best_bid": ticker.get("bid", 0),
                    "best_ask": ticker.get("ask", 0),
                    "volume_24h": ticker.get("baseVolume", 0),
                    "product_id": pair,
                })

                # Fetch recent trades
                try:
                    trades = await exchange.fetch_trades(symbol, limit=50)
                    for t in trades:
                        tid = str(t.get("id", ""))
                        if tid and tid == self._last_trade_id.get(pair):
                            continue
                        self._matches_buf[pair].append({
                            "_time": pd.Timestamp(t["timestamp"], unit="ms", tz="UTC"),
                            "price": t["price"],
                            "size": t["amount"],
                            "side": "BUY" if t["side"] == "buy" else "SELL",
                            "trade_id": tid,
                            "product_id": pair,
                        })
                    if trades:
                        self._last_trade_id[pair] = str(trades[-1].get("id", ""))
                except Exception:
                    pass  # trades endpoint may be rate-limited

            except Exception as e:
                logger.debug("Failed to fetch %s: %s", pair, str(e)[:60])

    def _flush_buffers(self):
        """Append buffered data to parquet files."""
        for pair in self.pairs:
            safe = pair.replace("-", "_")

            if self._ticker_buf[pair]:
                df = pd.DataFrame(self._ticker_buf[pair])
                df.set_index("_time", inplace=True)
                self._append_parquet(f"ticker_{safe}.parquet", df)
                self._ticker_buf[pair] = []

            if self._matches_buf[pair]:
                df = pd.DataFrame(self._matches_buf[pair])
                df.set_index("_time", inplace=True)
                self._append_parquet(f"matches_{safe}.parquet", df)
                self._matches_buf[pair] = []

        logger.info("  Flushed buffers to parquet")

    def _append_parquet(self, filename: str, new_data: pd.DataFrame):
        """Append new data to existing parquet file."""
        path = self.data_dir / filename
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        else:
            combined = new_data

        combined.to_parquet(path, engine="pyarrow")

    def stop(self):
        self._running = False
        self._flush_buffers()

    async def close(self):
        self.stop()
        if self._exchange:
            await self._exchange.close()


async def async_main(args):
    pairs = args.pairs.split(",") if args.pairs else None
    collector = LiveCollector(
        data_dir=args.data_dir,
        pairs=pairs,
        collect_interval=args.interval,
        flush_interval=args.flush_interval,
    )

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, collector.stop)

    try:
        await collector.run()
    finally:
        await collector.close()


def main():
    parser = argparse.ArgumentParser(description="Live Coinbase data collector")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--interval", type=int, default=60,
                        help="Collection interval in seconds")
    parser.add_argument("--flush-interval", type=int, default=300,
                        help="Flush to disk interval in seconds")
    parser.add_argument("--pairs", default=None,
                        help="Comma-separated pairs (default: all)")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
