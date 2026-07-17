from __future__ import annotations

import logging
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import pandas as pd

from src.config import AppConfig

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Fetches OHLCV candles and ticker data from Coinbase via ccxt."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._exchange: ccxt.Exchange | None = None

    async def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            exchange_cls = getattr(ccxt, self.config.exchange.name)
            params: dict = {"enableRateLimit": self.config.exchange.rate_limit}
            if self.config.coinbase_api_key_name:
                params["apiKey"] = self.config.coinbase_api_key_name
                params["secret"] = self.config.coinbase_api_private_key
            self._exchange = exchange_cls(params)
        return self._exchange

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 300,
    ) -> pd.DataFrame:
        exchange = await self._get_exchange()
        raw = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    async def fetch_ohlcv_since(
        self, symbol: str, since_ms: int, timeframe: str = "1m", limit: int = 300,
    ) -> pd.DataFrame:
        """Fetch candles starting from a specific timestamp (ms)."""
        exchange = await self._get_exchange()
        raw = await exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=limit,
        )
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    async def fetch_ticker(self, symbol: str) -> dict:
        exchange = await self._get_exchange()
        return await exchange.fetch_ticker(symbol)

    async def close(self):
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
