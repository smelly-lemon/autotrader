#!/usr/bin/env python3
"""Bulk-export InfluxDB data to parquet — runs ON the Jetson.

Strategy for Jetson's 7.4GB RAM:
- Ticker: aggregateWindow(1m) + pivot (tiny)
- Matches: pivot + 1-day chunks, writes part files to avoid OOM

Supports resume: skips files that already have data.

Usage (on hesiod):
    source ~/Desktop/crypto/jetson_hft_bot/venv/bin/activate
    python3 export_data.py
"""
import gc
import json
import logging
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction

warnings.simplefilter("ignore", MissingPivotFunction)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

INFLUX_URL = "http://localhost:8086"
INFLUX_ORG = "Sybl"
INFLUX_BUCKET = "coinbase_data"
SECRETS_FILE = os.path.expanduser("~/Desktop/crypto/jetson_hft_bot/secrets.json")

OUTPUT_DIR = os.path.expanduser("~/export")

PRODUCT_IDS = [
    "BTC-USD", "ETH-USD", "BTC-USDC", "ETH-USDC",
    "ETH-BTC", "SOL-BTC", "SOL-USD", "XRP-USD",
    "DOGE-USD", "SHIB-USD", "LINK-USD", "UNI-USD",
]

DATA_START = datetime(2025, 11, 5, tzinfo=timezone.utc)


def load_token():
    with open(SECRETS_FILE) as f:
        return json.load(f)["if-db-token"]


def query_df(query_api, flux):
    df = query_api.query_data_frame(query=flux)
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True)
    return df.drop(columns=["result", "table"], errors="ignore")


def export_ticker(query_api, product_id, start_dt, stop_dt, out_path):
    """Ticker: 1-min aggregated bars. Very fast."""
    if out_path.exists() and out_path.stat().st_size > 100:
        logger.info("  [SKIP] %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
        return

    chunk_days = 30
    cursor = start_dt
    chunks = []
    total_rows = 0
    total_chunks = max(1, (stop_dt - start_dt).days // chunk_days + 1)
    ci = 0

    while cursor < stop_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), stop_dt)
        ci += 1
        s = cursor.strftime("%Y-%m-%dT%H:%M:%SZ")
        e = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        t0 = time.time()

        flux = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: {s}, stop: {e})
            |> filter(fn: (r) => r._measurement == "ticker" and r.product_id == "{product_id}")
            |> aggregateWindow(every: 1m, fn: last, createEmpty: false)
            |> pivot(rowKey:["_time", "product_id"], columnKey: ["_field"], valueColumn: "_value")
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> sort(columns: ["_time"])
        '''
        try:
            df = query_df(query_api, flux)
            if not df.empty:
                chunks.append(df)
                total_rows += len(df)
                logger.info("    %d/%d %s->%s %d rows %.0fs (total: %d)",
                            ci, total_chunks, s[:10], e[:10], len(df), time.time() - t0, total_rows)
        except Exception as ex:
            logger.error("    %d/%d FAIL: %s", ci, total_chunks, str(ex)[:150])

        cursor = chunk_end

    if chunks:
        df = pd.concat(chunks, ignore_index=True)
        if "_time" in df.columns:
            df["_time"] = pd.to_datetime(df["_time"], utc=True)
            df.sort_values("_time", inplace=True)
            df.set_index("_time", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(out_path, engine="pyarrow")
        logger.info("  [DONE] %s: %d rows, %.1f MB", out_path.name, len(df), out_path.stat().st_size / 1e6)


def export_matches(query_api, product_id, start_dt, stop_dt, out_dir, final_name):
    """Matches: 1-day chunks, writes individual part files to avoid OOM.

    Part files are combined at the end into the final parquet.
    """
    final_path = out_dir / final_name
    if final_path.exists() and final_path.stat().st_size > 100:
        logger.info("  [SKIP] %s (%.1f MB)", final_path.name, final_path.stat().st_size / 1e6)
        return

    parts_dir = out_dir / f".parts_{final_name.replace('.parquet', '')}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    cursor = start_dt
    total_rows = 0
    total_days = (stop_dt - start_dt).days
    ci = 0

    while cursor < stop_dt:
        chunk_end = min(cursor + timedelta(days=1), stop_dt)
        ci += 1
        day_str = cursor.strftime("%Y-%m-%d")
        part_path = parts_dir / f"part_{day_str}.parquet"

        # Skip if this day's part already exists (resume support)
        if part_path.exists() and part_path.stat().st_size > 100:
            try:
                nrows = len(pd.read_parquet(part_path))
                total_rows += nrows
                logger.info("    %d/%d %s [cached: %d rows] (total: %d)",
                            ci, total_days, day_str, nrows, total_rows)
                cursor = chunk_end
                continue
            except Exception:
                pass

        s = cursor.strftime("%Y-%m-%dT%H:%M:%SZ")
        e = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        t0 = time.time()

        flux = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: {s}, stop: {e})
            |> filter(fn: (r) => r._measurement == "matches" and r.product_id == "{product_id}")
            |> pivot(rowKey:["_time", "product_id"], columnKey: ["_field"], valueColumn: "_value")
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> sort(columns: ["_time"])
        '''
        try:
            df = query_df(query_api, flux)
            if not df.empty:
                if "_time" in df.columns:
                    df["_time"] = pd.to_datetime(df["_time"], utc=True)
                    df.sort_values("_time", inplace=True)
                    df.set_index("_time", inplace=True)
                df.to_parquet(part_path, engine="pyarrow")
                total_rows += len(df)
                elapsed = time.time() - t0
                logger.info("    %d/%d %s %d rows %.0fs (total: %d)",
                            ci, total_days, day_str, len(df), elapsed, total_rows)
                del df
            else:
                logger.info("    %d/%d %s empty", ci, total_days, day_str)
        except Exception as ex:
            logger.error("    %d/%d %s FAIL: %s", ci, total_days, day_str, str(ex)[:150])

        cursor = chunk_end
        gc.collect()

    # Leave part files for the Mac to combine (avoids OOM on Jetson)
    part_files = sorted(parts_dir.glob("part_*.parquet"))
    total_size = sum(p.stat().st_size for p in part_files)
    logger.info("  [PARTS] %d part files, %.1f MB total (combine on Mac)",
                len(part_files), total_size / 1e6)


def export_level2(query_api, product_id, start_dt, stop_dt, out_dir, final_name):
    """Level 2 orderbook snapshots: 1-day chunks, part files like matches."""
    final_path = out_dir / final_name
    if final_path.exists() and final_path.stat().st_size > 100:
        logger.info("  [SKIP] %s (%.1f MB)", final_path.name, final_path.stat().st_size / 1e6)
        return

    parts_dir = out_dir / f".parts_{final_name.replace('.parquet', '')}"
    parts_dir.mkdir(parents=True, exist_ok=True)

    cursor = start_dt
    total_rows = 0
    total_days = (stop_dt - start_dt).days
    ci = 0

    while cursor < stop_dt:
        chunk_end = min(cursor + timedelta(days=1), stop_dt)
        ci += 1
        day_str = cursor.strftime("%Y-%m-%d")
        part_path = parts_dir / f"part_{day_str}.parquet"

        if part_path.exists() and part_path.stat().st_size > 100:
            try:
                nrows = len(pd.read_parquet(part_path))
                total_rows += nrows
                cursor = chunk_end
                continue
            except Exception:
                pass

        s = cursor.strftime("%Y-%m-%dT%H:%M:%SZ")
        e = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        t0 = time.time()

        flux = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: {s}, stop: {e})
            |> filter(fn: (r) => r._measurement == "level2" and r.product_id == "{product_id}")
            |> pivot(rowKey:["_time", "product_id"], columnKey: ["_field"], valueColumn: "_value")
            |> drop(columns: ["_start", "_stop", "_measurement"])
            |> sort(columns: ["_time"])
        '''
        try:
            df = query_df(query_api, flux)
            if not df.empty:
                if "_time" in df.columns:
                    df["_time"] = pd.to_datetime(df["_time"], utc=True)
                    df.sort_values("_time", inplace=True)
                    df.set_index("_time", inplace=True)
                df.to_parquet(part_path, engine="pyarrow")
                total_rows += len(df)
                elapsed = time.time() - t0
                logger.info("    %d/%d %s %d rows %.0fs (total: %d)",
                            ci, total_days, day_str, len(df), elapsed, total_rows)
                del df
            else:
                logger.info("    %d/%d %s empty", ci, total_days, day_str)
        except Exception as ex:
            logger.error("    %d/%d %s FAIL: %s", ci, total_days, day_str, str(ex)[:150])

        cursor = chunk_end
        gc.collect()

    part_files = sorted(parts_dir.glob("part_*.parquet"))
    total_size = sum(p.stat().st_size for p in part_files)
    logger.info("  [PARTS] L2 %d part files, %.1f MB total", len(part_files), total_size / 1e6)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-l2", action="store_true",
                        help="Also export level2 orderbook data")
    parser.add_argument("--l2-only", action="store_true",
                        help="Only export level2 data (skip ticker/matches)")
    args = parser.parse_args()

    token = load_token()
    client = InfluxDBClient(url=INFLUX_URL, token=token, org=INFLUX_ORG, timeout=600_000)
    query_api = client.query_api()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = DATA_START
    stop_dt = datetime.now(timezone.utc)
    t_start = time.time()

    for i, pair in enumerate(PRODUCT_IDS):
        safe = pair.replace("-", "_")
        logger.info("=" * 50)
        logger.info("[%d/%d] %s", i + 1, len(PRODUCT_IDS), pair)
        logger.info("=" * 50)

        if not args.l2_only:
            export_ticker(query_api, pair, start_dt, stop_dt, out_dir / f"ticker_{safe}.parquet")
            gc.collect()

            export_matches(query_api, pair, start_dt, stop_dt, out_dir, f"matches_{safe}.parquet")
            gc.collect()

        if args.include_l2 or args.l2_only:
            logger.info("  Exporting L2 orderbook...")
            export_level2(query_api, pair, start_dt, stop_dt, out_dir, f"level2_{safe}.parquet")
            gc.collect()

    elapsed = time.time() - t_start
    logger.info("=" * 50)
    logger.info("ALL DONE in %.1f minutes (%.1f hours)", elapsed / 60, elapsed / 3600)

    total_size = sum(f.stat().st_size for f in out_dir.glob("*.parquet"))
    total_files = len(list(out_dir.glob("*.parquet")))
    logger.info("%d files, %.1f MB total", total_files, total_size / 1e6)

    client.close()


if __name__ == "__main__":
    main()
