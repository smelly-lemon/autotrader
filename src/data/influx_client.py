"""Client for querying historical tick data from hesiod's InfluxDB.

Supports two modes:
- HesiodInfluxClient: queries InfluxDB over SSH tunnel (slow for bulk)
- LocalParquetClient: reads from exported parquet files (fast)

Usage:
    # From InfluxDB (requires SSH tunnel)
    from src.data.influx_client import HesiodInfluxClient
    client = HesiodInfluxClient()

    # From local parquet files (fast, no tunnel needed)
    from src.data.influx_client import LocalParquetClient
    client = LocalParquetClient("data/raw")

    # Both share the same interface:
    df = client.get_ticker_data("BTC-USD", start="2026-01-01", stop="2026-01-02")
    client.close()
"""
from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CHUNK_DAYS = 5  # max days per query to avoid OOM on the Jetson

INFLUX_URL = "http://localhost:8086"
INFLUX_ORG = "Sybl"
INFLUX_BUCKET = "coinbase_data"
# Set INFLUX_TOKEN in .env — never hardcode credentials here.
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "")

PRODUCT_IDS = [
    "BTC-USD", "ETH-USD", "BTC-USDC", "ETH-USDC",
    "ETH-BTC", "SOL-BTC", "SOL-USD", "XRP-USD",
    "DOGE-USD", "SHIB-USD", "LINK-USD", "UNI-USD",
]

# Continuous data starts here (skip the early gap period)
DATA_START = "2025-11-05T00:00:00Z"


class HesiodInfluxClient:
    """Query interface for hesiod's InfluxDB coinbase_data bucket."""

    def __init__(
        self,
        url: str = INFLUX_URL,
        token: str = INFLUX_TOKEN,
        org: str = INFLUX_ORG,
        bucket: str = INFLUX_BUCKET,
        timeout_ms: int = 300_000,
    ):
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.warnings import MissingPivotFunction
        warnings.simplefilter("ignore", MissingPivotFunction)

        self._client = InfluxDBClient(url=url, token=token, org=org, timeout=timeout_ms)
        self._query_api = self._client.query_api()
        self._bucket = bucket

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _query_df(self, flux: str) -> pd.DataFrame:
        df = self._query_api.query_data_frame(query=flux)
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
        return df.drop(columns=["result", "table"], errors="ignore")

    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        if "T" not in ts:
            ts = f"{ts}T00:00:00Z"
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @staticmethod
    def _fmt_ts(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _chunked_query(
        self,
        measurement: str,
        product_id: str,
        start: str,
        stop: str | None,
        chunk_days: int = CHUNK_DAYS,
    ) -> pd.DataFrame:
        """Fetch data in time-chunked windows to avoid OOM on the Jetson."""
        start_dt = self._parse_ts(start)
        stop_dt = self._parse_ts(stop) if stop else datetime.now(timezone.utc)

        chunks = []
        cursor = start_dt
        chunk_i = 0
        total_chunks = max(1, int((stop_dt - start_dt).days / chunk_days) + 1)

        while cursor < stop_dt:
            chunk_end = min(cursor + timedelta(days=chunk_days), stop_dt)
            chunk_i += 1
            s = self._fmt_ts(cursor)
            e = self._fmt_ts(chunk_end)

            logger.debug("    chunk %d/%d: %s -> %s", chunk_i, total_chunks, s[:10], e[:10])

            flux = f'''
            from(bucket: "{self._bucket}")
                |> range(start: {s}, stop: {e})
                |> filter(fn: (r) => r._measurement == "{measurement}" and r.product_id == "{product_id}")
                |> pivot(rowKey:["_time", "product_id"], columnKey: ["_field"], valueColumn: "_value")
                |> drop(columns: ["_start", "_stop", "_measurement"])
                |> sort(columns: ["_time"])
            '''
            try:
                df = self._query_df(flux)
                if not df.empty:
                    chunks.append(df)
            except Exception:
                logger.warning("    chunk %d failed for %s %s, retrying...", chunk_i, measurement, product_id)
                import time
                time.sleep(2)
                try:
                    df = self._query_df(flux)
                    if not df.empty:
                        chunks.append(df)
                except Exception:
                    logger.error("    chunk %d retry failed, skipping", chunk_i)

            cursor = chunk_end

        if not chunks:
            return pd.DataFrame()

        result = pd.concat(chunks, ignore_index=True)
        if "_time" in result.columns:
            result["_time"] = pd.to_datetime(result["_time"], utc=True)
            result.set_index("_time", inplace=True)
            result.sort_index(inplace=True)
        return result

    def get_ticker_data(
        self,
        product_id: str,
        start: str = DATA_START,
        stop: str | None = None,
    ) -> pd.DataFrame:
        """Get pivoted ticker data: price, best_bid, best_ask, volume_24h, etc."""
        return self._chunked_query("ticker", product_id, start, stop)

    def get_matches(
        self,
        product_id: str,
        start: str = DATA_START,
        stop: str | None = None,
    ) -> pd.DataFrame:
        """Get executed trades: price, size, side, trade_id."""
        return self._chunked_query("matches", product_id, start, stop)

    def get_level2(
        self,
        product_id: str,
        start: str = DATA_START,
        stop: str | None = None,
    ) -> pd.DataFrame:
        """Get L2 order book updates: price, size, side."""
        stop_clause = f', stop: {stop}T00:00:00Z' if stop and "T" not in stop else (f', stop: {stop}' if stop else '')
        start_ts = f'{start}T00:00:00Z' if "T" not in start else start

        flux = f'''
        from(bucket: "{self._bucket}")
            |> range(start: {start_ts}{stop_clause})
            |> filter(fn: (r) => r._measurement == "level2" and r.product_id == "{product_id}")
            |> pivot(rowKey:["_time", "product_id"], columnKey: ["_field"], valueColumn: "_value")
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> sort(columns: ["_time"])
        '''
        df = self._query_df(flux)
        if "_time" in df.columns:
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df.set_index("_time", inplace=True)
        return df

    def build_ohlcv(
        self,
        product_id: str,
        interval: str = "1m",
        start: str = DATA_START,
        stop: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate tick-level matches into OHLCV candles at any interval.

        This builds candles from actual trade data (matches), giving you
        any arbitrary timeframe that the exchange doesn't natively offer.
        """
        stop_clause = f', stop: {stop}T00:00:00Z' if stop and "T" not in stop else (f', stop: {stop}' if stop else '')
        start_ts = f'{start}T00:00:00Z' if "T" not in start else start

        # Build OHLCV from trade matches
        flux = f'''
        base = from(bucket: "{self._bucket}")
            |> range(start: {start_ts}{stop_clause})
            |> filter(fn: (r) => r._measurement == "matches" and r.product_id == "{product_id}")

        open = base
            |> filter(fn: (r) => r._field == "price")
            |> aggregateWindow(every: {interval}, fn: first, createEmpty: false)
            |> set(key: "_field", value: "open")

        high = base
            |> filter(fn: (r) => r._field == "price")
            |> aggregateWindow(every: {interval}, fn: max, createEmpty: false)
            |> set(key: "_field", value: "high")

        low = base
            |> filter(fn: (r) => r._field == "price")
            |> aggregateWindow(every: {interval}, fn: min, createEmpty: false)
            |> set(key: "_field", value: "low")

        close = base
            |> filter(fn: (r) => r._field == "price")
            |> aggregateWindow(every: {interval}, fn: last, createEmpty: false)
            |> set(key: "_field", value: "close")

        volume = base
            |> filter(fn: (r) => r._field == "size")
            |> aggregateWindow(every: {interval}, fn: sum, createEmpty: false)
            |> set(key: "_field", value: "volume")

        trade_count = base
            |> filter(fn: (r) => r._field == "price")
            |> aggregateWindow(every: {interval}, fn: count, createEmpty: false)
            |> set(key: "_field", value: "trade_count")

        union(tables: [open, high, low, close, volume, trade_count])
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> drop(columns: ["_start", "_stop", "_measurement", "product_id"])
            |> sort(columns: ["_time"])
        '''
        df = self._query_df(flux)
        if "_time" in df.columns:
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df.set_index("_time", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_spread_data(
        self,
        product_id: str,
        interval: str = "1m",
        start: str = DATA_START,
        stop: str | None = None,
    ) -> pd.DataFrame:
        """Get bid-ask spread statistics aggregated to any interval."""
        stop_clause = f', stop: {stop}T00:00:00Z' if stop and "T" not in stop else (f', stop: {stop}' if stop else '')
        start_ts = f'{start}T00:00:00Z' if "T" not in start else start

        flux = f'''
        from(bucket: "{self._bucket}")
            |> range(start: {start_ts}{stop_clause})
            |> filter(fn: (r) => r._measurement == "ticker" and r.product_id == "{product_id}")
            |> filter(fn: (r) => r._field == "best_bid" or r._field == "best_ask" or r._field == "price")
            |> aggregateWindow(every: {interval}, fn: mean, createEmpty: false)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> map(fn: (r) => ({{r with spread: r.best_ask - r.best_bid}}))
            |> drop(columns: ["_start", "_stop", "_measurement", "product_id"])
            |> sort(columns: ["_time"])
        '''
        df = self._query_df(flux)
        if "_time" in df.columns:
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df.set_index("_time", inplace=True)
        return df


class LocalParquetClient:
    """Drop-in replacement for HesiodInfluxClient that reads from local parquet files.

    Expects parquet files exported by scripts/jetson_export.py:
        data_dir/ticker_BTC_USD.parquet
        data_dir/matches_BTC_USD.parquet
        etc.
    """

    def __init__(self, data_dir: str | Path = "data/raw"):
        self._dir = Path(data_dir)
        self._cache: dict[str, pd.DataFrame] = {}
        if not self._dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self._dir}")
        logger.info("LocalParquetClient: reading from %s", self._dir)

    def close(self):
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _load(self, measurement: str, product_id: str) -> pd.DataFrame:
        safe = product_id.replace("-", "_")
        key = f"{measurement}_{safe}"
        if key in self._cache:
            return self._cache[key]

        path = self._dir / f"{key}.parquet"
        parts_dir = self._dir / f".parts_{key}"
        # Daily shards written by scripts/live_collector.py (key.YYYYMMDD.parquet)
        shard_files = sorted(self._dir.glob(f"{key}.*.parquet"))

        frames: list[pd.DataFrame] = []
        if path.exists():
            logger.info("  Loading %s (%.1f MB)...", path.name, path.stat().st_size / 1e6)
            frames.append(pd.read_parquet(path))
        elif parts_dir.exists():
            part_files = sorted(parts_dir.glob("part_*.parquet"))
            if part_files:
                logger.info("  Loading %d part files from %s...", len(part_files), parts_dir.name)
                combined = pd.concat([pd.read_parquet(p) for p in part_files])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.sort_index(inplace=True)
                # Cache the combined result as a single parquet for next time
                combined.to_parquet(path, engine="pyarrow")
                logger.info("  Combined -> %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
                frames.append(combined)

        if shard_files:
            logger.info("  Loading %d daily shard files for %s...", len(shard_files), key)
            frames.extend(pd.read_parquet(p) for p in shard_files)

        if not frames:
            logger.debug("No parquet file: %s", path)
            return pd.DataFrame()

        df = pd.concat(frames)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

        # Handle raw (unpivoted) format from jetson_export.py:
        # columns: _time, _field, _value, product_id
        if "_field" in df.columns and "_value" in df.columns:
            logger.info("  Pivoting raw data (%d rows)...", len(df))
            if "_time" in df.columns:
                df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df = df.pivot_table(
                index="_time", columns="_field", values="_value", aggfunc="first"
            )
            df.columns.name = None
            df["product_id"] = product_id
            df.sort_index(inplace=True)
        else:
            # Already pivoted format
            if not isinstance(df.index, pd.DatetimeIndex):
                if "_time" in df.columns:
                    df["_time"] = pd.to_datetime(df["_time"], utc=True)
                    df.set_index("_time", inplace=True)
                elif df.index.dtype == "object":
                    df.index = pd.to_datetime(df.index, utc=True)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        self._cache[key] = df
        return df

    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        if "T" not in ts:
            ts = f"{ts}T00:00:00Z"
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _slice(self, df: pd.DataFrame, start: str, stop: str | None) -> pd.DataFrame:
        if df.empty:
            return df
        start_dt = self._parse_ts(start)
        if stop:
            stop_dt = self._parse_ts(stop)
            return df.loc[start_dt:stop_dt]
        return df.loc[start_dt:]

    def get_ticker_data(
        self,
        product_id: str,
        start: str = DATA_START,
        stop: str | None = None,
    ) -> pd.DataFrame:
        df = self._load("ticker", product_id)
        return self._slice(df, start, stop)

    def get_matches(
        self,
        product_id: str,
        start: str = DATA_START,
        stop: str | None = None,
    ) -> pd.DataFrame:
        df = self._load("matches", product_id)
        return self._slice(df, start, stop)
