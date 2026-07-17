from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import pandas as pd

from src.config import AppConfig

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Fetches OHLCV candles and order book data from Coinbase via ccxt."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._exchange: ccxt.Exchange | None = None

    async def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            exchange_cls = getattr(ccxt, self.config.exchange.name)
            params: dict = {"enableRateLimit": self.config.exchange.rate_limit}
            if self.config.coinbase_api_key:
                params["apiKey"] = self.config.coinbase_api_key
                params["secret"] = self.config.coinbase_api_secret
            self._exchange = exchange_cls(params)
        return self._exchange

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 200,
    ) -> pd.DataFrame:
        exchange = await self._get_exchange()
        raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        exchange = await self._get_exchange()
        book = await exchange.fetch_order_book(symbol, limit=limit)
        return {
            "bids": book["bids"][:limit],
            "asks": book["asks"][:limit],
            "spread": book["asks"][0][0] - book["bids"][0][0] if book["asks"] and book["bids"] else 0,
            "mid_price": (book["asks"][0][0] + book["bids"][0][0]) / 2 if book["asks"] and book["bids"] else 0,
        }

    async def fetch_ticker(self, symbol: str) -> dict:
        exchange = await self._get_exchange()
        return await exchange.fetch_ticker(symbol)

    async def fetch_all_pairs(self, timeframe: str = "5m") -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for all configured trading pairs concurrently."""
        tasks = {pair: self.fetch_ohlcv(pair, timeframe) for pair in self.config.trading.pairs}
        results = {}
        for pair, coro in tasks.items():
            try:
                results[pair] = await coro
            except Exception:
                logger.exception("Failed to fetch data for %s", pair)
        return results

    async def close(self):
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
