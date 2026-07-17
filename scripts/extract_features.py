"""Robust chunked feature extraction — saves progress incrementally.

Extracts features in weekly chunks per pair and saves intermediate results
to parquet. If interrupted, resumes from where it left off.

Usage:
    python scripts/extract_features.py --dataset lead-lag --days 90
    python scripts/extract_features.py --dataset swing --days 195
"""
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.data.influx_client import HesiodInfluxClient
from src.ml.features import (
    FeatureBuilder, BTC_PAIR, STABLECOIN_PAIR, ALT_PAIRS,
    build_bar_features_from_ticker, build_bar_features_from_matches,
    compute_vpin, compute_return_features, compute_temporal_features,
    compute_targets, compute_cross_pair_features,
)
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_DAYS = 7


def extract_pair_chunked(
    client: HesiodInfluxClient,
    pair: str,
    start_dt: datetime,
    stop_dt: datetime,
    interval: str,
    out_dir: Path,
) -> pd.DataFrame | None:
    """Extract features for one pair in weekly chunks, caching each chunk."""
    cache_path = out_dir / f"{pair.replace('-', '_')}_{interval}.parquet"

    if cache_path.exists():
        logger.info("  [CACHED] %s — loading from %s", pair, cache_path)
        return pd.read_parquet(cache_path)

    builder = FeatureBuilder(client)
    cursor = start_dt
    chunks = []
    chunk_i = 0
    total_chunks = max(1, int((stop_dt - start_dt).days / CHUNK_DAYS) + 1)

    while cursor < stop_dt:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), stop_dt)
        chunk_i += 1
        s = cursor.strftime("%Y-%m-%dT%H:%M:%SZ")
        e = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("  [%s] chunk %d/%d: %s -> %s", pair, chunk_i, total_chunks, s[:10], e[:10])

        t0 = time.time()
        try:
            df = builder.build_pair_features(pair, s, e, interval, include_targets=True)
            if not df.empty:
                chunks.append(df)
                logger.info("    -> %d rows in %.1fs", len(df), time.time() - t0)
        except Exception:
            logger.exception("    chunk %d FAILED for %s", chunk_i, pair)

        cursor = chunk_end

    if not chunks:
        logger.warning("  No data for %s", pair)
        return None

    combined = pd.concat(chunks)
    combined.sort_index(inplace=True)

    # Remove duplicates from chunk overlaps
    combined = combined[~combined.index.duplicated(keep="last")]

    combined.to_parquet(cache_path)
    logger.info("  [SAVED] %s: %d rows -> %s", pair, len(combined), cache_path)
    return combined


def build_lead_lag_from_cached(
    client: HesiodInfluxClient,
    start_dt: datetime,
    stop_dt: datetime,
    interval: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Build the full lead-lag dataset using per-pair cached parquets."""

    # BTC features
    btc_df = extract_pair_chunked(client, BTC_PAIR, start_dt, stop_dt, interval, out_dir)
    if btc_df is None or btc_df.empty:
        logger.error("No BTC data")
        return pd.DataFrame()

    btc_price = btc_df["price"]

    # Stablecoin price
    stablecoin_price = None
    try:
        sc_cache = out_dir / f"{STABLECOIN_PAIR.replace('-', '_')}_stablecoin.parquet"
        if sc_cache.exists():
            sc_df = pd.read_parquet(sc_cache)
            stablecoin_price = sc_df["price"]
        else:
            logger.info("Fetching stablecoin pair for spread calculation...")
            sc_ticker = client.get_ticker_data(STABLECOIN_PAIR,
                                                start=start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                stop=stop_dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            if not sc_ticker.empty and "price" in sc_ticker.columns:
                sc_ticker["price"] = pd.to_numeric(sc_ticker["price"], errors="coerce")
                stablecoin_price = sc_ticker.resample(interval)["price"].last().dropna()
                pd.DataFrame({"price": stablecoin_price}).to_parquet(sc_cache)
    except Exception:
        logger.warning("Stablecoin pair unavailable")

    btc_ret_5 = np.log(btc_price).diff(5)
    btc_moved = btc_ret_5.abs() > 0.003

    all_alt_frames = []

    for alt_pair in ALT_PAIRS:
        alt_df = extract_pair_chunked(client, alt_pair, start_dt, stop_dt, interval, out_dir)
        if alt_df is None or alt_df.empty:
            continue

        alt_price = alt_df["price"]

        cross = compute_cross_pair_features(btc_price, alt_price, stablecoin_price)

        btc_cols = {
            "spread_bps_mean": "btc_spread_bps_mean",
            "volatility_15": "btc_volatility_15",
            "volatility_60": "btc_volatility_60",
            "tick_count": "btc_tick_count",
        }
        btc_context = btc_df[[c for c in btc_cols.keys() if c in btc_df.columns]].rename(columns=btc_cols)

        combined = alt_df.join(cross).join(btc_context)
        combined["btc_moved"] = btc_moved.reindex(combined.index, fill_value=False)
        combined["product_id"] = alt_pair
        all_alt_frames.append(combined)

    if not all_alt_frames:
        return pd.DataFrame()

    dataset = pd.concat(all_alt_frames)
    dataset.sort_index(inplace=True)
    return dataset


def build_swing_from_cached(
    client: HesiodInfluxClient,
    start_dt: datetime,
    stop_dt: datetime,
    interval: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Build swing dataset from cached per-pair parquets."""
    from src.data.influx_client import PRODUCT_IDS

    all_frames = []
    price_dict = {}

    for pair in PRODUCT_IDS:
        df = extract_pair_chunked(client, pair, start_dt, stop_dt, interval, out_dir)
        if df is None or df.empty:
            continue
        price_dict[pair] = df["price"]
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    dataset = pd.concat(all_frames)

    if len(price_dict) >= 4:
        logger.info("Computing cross-pair correlation matrix...")
        price_df = pd.DataFrame(price_dict)
        ret_df = np.log(price_df).diff()
        rolling_corr_mean = ret_df.rolling(30, min_periods=10).corr().groupby(level=0).mean().mean(axis=1)
        rolling_corr_mean.name = "cross_corr_mean"

        for pair in PRODUCT_IDS:
            mask = dataset["product_id"] == pair
            idx = dataset.loc[mask].index
            dataset.loc[mask, "cross_corr_mean"] = rolling_corr_mean.reindex(idx).values

    dataset.sort_index(inplace=True)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lead-lag", "swing", "both"], default="both")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--out", default="data/features")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stop_dt = datetime.now(timezone.utc)
    start_dt = stop_dt - timedelta(days=args.days)

    client = HesiodInfluxClient()

    try:
        if args.dataset in ("lead-lag", "both"):
            logger.info("=" * 60)
            logger.info("Extracting LEAD-LAG features (%d days, 1min bars)", args.days)
            logger.info("=" * 60)

            ll_cache_dir = out_dir / "lead_lag_cache"
            ll_cache_dir.mkdir(exist_ok=True)

            dataset = build_lead_lag_from_cached(client, start_dt, stop_dt, "1min", ll_cache_dir)
            if not dataset.empty:
                path = out_dir / "lead_lag_features.parquet"
                dataset.to_parquet(path, engine="pyarrow")
                logger.info("SAVED lead-lag dataset: %s (%d rows, %d cols)",
                            path, len(dataset), len(dataset.columns))
            else:
                logger.error("No lead-lag data extracted")

        if args.dataset in ("swing", "both"):
            logger.info("=" * 60)
            logger.info("Extracting SWING features (%d days, 4h bars)", args.days)
            logger.info("=" * 60)

            sw_cache_dir = out_dir / "swing_cache"
            sw_cache_dir.mkdir(exist_ok=True)

            dataset = build_swing_from_cached(client, start_dt, stop_dt, "4h", sw_cache_dir)
            if not dataset.empty:
                path = out_dir / "swing_features.parquet"
                dataset.to_parquet(path, engine="pyarrow")
                logger.info("SAVED swing dataset: %s (%d rows, %d cols)",
                            path, len(dataset), len(dataset.columns))
            else:
                logger.error("No swing data extracted")

    finally:
        client.close()

    logger.info("DONE.")


if __name__ == "__main__":
    main()
