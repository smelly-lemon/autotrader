"""Feature engineering pipeline for ML trading models.

Extracts cross-pair lead-lag features, VPIN, spread statistics, and
microstructure metrics from the Hesiod InfluxDB tick data.

Usage:
    python -m src.ml.features --start 2025-11-05 --stop 2026-05-16 --interval 1m --out data/features
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.influx_client import HesiodInfluxClient, PRODUCT_IDS, DATA_START

logger = logging.getLogger(__name__)

BTC_PAIR = "BTC-USD"
STABLECOIN_PAIR = "BTC-USDC"
ALT_PAIRS = [p for p in PRODUCT_IDS if p not in (BTC_PAIR, STABLECOIN_PAIR)]

# VPIN parameters
VPIN_BUCKET_SIZE = 50  # trades per volume bucket
VPIN_WINDOW = 50  # number of buckets for rolling VPIN


def compute_vpin(trades_df: pd.DataFrame, bucket_size: int = VPIN_BUCKET_SIZE,
                 window: int = VPIN_WINDOW) -> pd.Series:
    """Compute Volume-Synchronized Probability of Informed Trading.

    Groups trades into fixed-volume buckets and measures buy/sell imbalance
    across a rolling window. Returns a Series indexed by the timestamp of
    the last trade in each bucket.
    """
    if trades_df.empty or "size" not in trades_df.columns:
        return pd.Series(dtype=float)

    df = trades_df[["size", "side"]].copy()
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
    df["signed_vol"] = df["size"].where(df["side"] == "BUY", -df["size"])
    df["cumvol"] = df["size"].cumsum()
    df["bucket"] = (df["cumvol"] / bucket_size).astype(int)

    grouped = df.groupby("bucket")
    bucket_imbalance = grouped["signed_vol"].sum().abs()
    bucket_volume = grouped["size"].sum()
    bucket_time = grouped.apply(lambda g: g.index[-1])

    ratio = bucket_imbalance / bucket_volume.replace(0, np.nan)
    vpin = ratio.rolling(window, min_periods=max(1, window // 2)).mean()
    idx = pd.DatetimeIndex(bucket_time.values)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    vpin.index = idx
    return vpin.dropna()


def build_bar_features_from_ticker(
    ticker_df: pd.DataFrame,
    interval: str = "1min",
) -> pd.DataFrame:
    """Aggregate raw ticker ticks into bars with spread/microstructure stats."""
    if ticker_df.empty:
        return pd.DataFrame()

    for col in ["price", "best_bid", "best_ask", "volume_24h"]:
        if col in ticker_df.columns:
            ticker_df[col] = pd.to_numeric(ticker_df[col], errors="coerce")

    if "best_bid" in ticker_df.columns and "best_ask" in ticker_df.columns:
        ticker_df["spread"] = ticker_df["best_ask"] - ticker_df["best_bid"]
        ticker_df["spread_bps"] = (
            ticker_df["spread"] / ticker_df["price"].replace(0, np.nan) * 10000
        )

    agg = {}
    agg["price"] = ("price", "last")
    agg["price_open"] = ("price", "first")
    agg["price_high"] = ("price", "max")
    agg["price_low"] = ("price", "min")
    agg["tick_count"] = ("price", "count")

    if "best_bid" in ticker_df.columns:
        agg["best_bid"] = ("best_bid", "last")
    if "best_ask" in ticker_df.columns:
        agg["best_ask"] = ("best_ask", "last")
    if "spread_bps" in ticker_df.columns:
        agg["spread_bps_mean"] = ("spread_bps", "mean")
        agg["spread_bps_std"] = ("spread_bps", "std")
    if "volume_24h" in ticker_df.columns:
        agg["volume_24h"] = ("volume_24h", "last")

    bars = ticker_df.resample(interval).agg(**agg).dropna(subset=["price"])
    return bars


def build_bar_features_from_matches(
    matches_df: pd.DataFrame,
    interval: str = "1min",
) -> pd.DataFrame:
    """Aggregate trade-level data into bars with microstructure features."""
    if matches_df.empty:
        return pd.DataFrame()

    df = matches_df.copy()
    for col in ["price", "size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["notional"] = df["price"] * df["size"]
    df["buy_size"] = df["size"].where(df["side"] == "BUY", 0.0)
    df["sell_size"] = df["size"].where(df["side"] != "BUY", 0.0)

    bars = df.resample(interval).agg(
        trade_count=("price", "count"),
        trade_volume=("size", "sum"),
        trade_notional=("notional", "sum"),
        buy_volume=("buy_size", "sum"),
        sell_volume=("sell_size", "sum"),
        vwap=("notional", "sum"),
        avg_trade_size=("size", "mean"),
        max_trade_size=("size", "max"),
    )

    bars["vwap"] = bars["vwap"] / bars["trade_volume"].replace(0, np.nan)
    total_vol = bars["buy_volume"] + bars["sell_volume"]
    bars["buy_sell_imbalance"] = (
        (bars["buy_volume"] - bars["sell_volume"]) / total_vol.replace(0, np.nan)
    )
    bars = bars.dropna(subset=["trade_count"])
    bars = bars[bars["trade_count"] > 0]
    return bars


def compute_return_features(price_series: pd.Series, lags: list[int] | None = None) -> pd.DataFrame:
    """Compute log returns and rolling statistics at multiple lags."""
    lags = lags or [1, 5, 15, 60]
    log_price = np.log(price_series.replace(0, np.nan))
    features = pd.DataFrame(index=price_series.index)

    for lag in lags:
        features[f"ret_{lag}"] = log_price.diff(lag)

    features["volatility_15"] = features["ret_1"].rolling(15, min_periods=5).std()
    features["volatility_60"] = features["ret_1"].rolling(60, min_periods=15).std()
    features["momentum_ratio"] = (
        features["ret_5"] / features["volatility_15"].replace(0, np.nan)
    )
    return features


def compute_cross_pair_features(
    btc_price: pd.Series,
    alt_price: pd.Series,
    stablecoin_price: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute lead-lag and relative value features between BTC and an alt."""
    features = pd.DataFrame(index=btc_price.index)

    btc_ret_1 = np.log(btc_price).diff(1)
    btc_ret_5 = np.log(btc_price).diff(5)
    btc_ret_15 = np.log(btc_price).diff(15)
    alt_ret_1 = np.log(alt_price).diff(1)
    alt_ret_5 = np.log(alt_price).diff(5)
    alt_ret_15 = np.log(alt_price).diff(15)

    features["btc_ret_1"] = btc_ret_1
    features["btc_ret_5"] = btc_ret_5
    features["btc_ret_15"] = btc_ret_15

    features["alt_ret_1"] = alt_ret_1
    features["alt_ret_5"] = alt_ret_5

    # How much of BTC's move the alt has absorbed
    features["absorption_ratio_5"] = alt_ret_5 / btc_ret_5.replace(0, np.nan)
    features["absorption_ratio_15"] = alt_ret_15 / btc_ret_15.replace(0, np.nan)

    # Rolling beta to BTC (60-bar window)
    cov = btc_ret_1.rolling(60, min_periods=20).cov(alt_ret_1)
    var = btc_ret_1.rolling(60, min_periods=20).var()
    features["rolling_beta_60"] = cov / var.replace(0, np.nan)

    # Rolling correlation
    features["rolling_corr_60"] = btc_ret_1.rolling(60, min_periods=20).corr(alt_ret_1)

    # BTC-USD vs BTC-USDC spread (stablecoin premium)
    if stablecoin_price is not None:
        aligned = stablecoin_price.reindex(btc_price.index, method="ffill")
        features["stablecoin_spread_bps"] = (
            (btc_price - aligned) / btc_price.replace(0, np.nan) * 10000
        )

    return features


def compute_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour-of-day and day-of-week features (empirically significant in crypto)."""
    features = pd.DataFrame(index=index)
    features["hour"] = index.hour
    features["day_of_week"] = index.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["dow_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
    features["dow_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
    return features


def compute_targets(price_series: pd.Series, horizons: list[int] | None = None) -> pd.DataFrame:
    """Forward-looking return targets for supervised training."""
    horizons = horizons or [1, 6, 15, 60, 240]
    log_price = np.log(price_series.replace(0, np.nan))
    targets = pd.DataFrame(index=price_series.index)

    for h in horizons:
        fwd = log_price.shift(-h) - log_price
        targets[f"fwd_ret_{h}"] = fwd
        targets[f"direction_{h}"] = (fwd > 0).astype(int)
        targets[f"fwd_abs_ret_{h}"] = fwd.abs()

    return targets


class FeatureBuilder:
    """Orchestrates feature extraction for a given pair and period.

    Accepts either HesiodInfluxClient (queries over tunnel) or
    LocalParquetClient (reads from exported parquet files).
    """

    def __init__(self, client=None):
        if client is None:
            client = HesiodInfluxClient()
        self._client = client
        self._owns_client = client is not None

    def close(self):
        if self._owns_client:
            self._client.close()

    def build_pair_features(
        self,
        product_id: str,
        start: str,
        stop: str,
        interval: str = "1min",
        include_targets: bool = True,
    ) -> pd.DataFrame:
        """Build full feature matrix for a single pair."""
        logger.info("Building features for %s [%s -> %s] @ %s", product_id, start, stop, interval)

        # Fetch ticker data
        logger.info("  Fetching ticker data...")
        ticker_df = self._client.get_ticker_data(product_id, start=start, stop=stop)
        ticker_bars = build_bar_features_from_ticker(ticker_df, interval)
        if ticker_bars.empty:
            logger.warning("  No ticker data for %s", product_id)
            return pd.DataFrame()

        # Fetch matches
        logger.info("  Fetching matches data...")
        matches_df = self._client.get_matches(product_id, start=start, stop=stop)
        match_bars = build_bar_features_from_matches(matches_df, interval)

        # Compute VPIN from matches
        vpin_series = pd.Series(dtype=float)
        if not matches_df.empty:
            logger.info("  Computing VPIN...")
            vpin_series = compute_vpin(matches_df)

        # Returns & volatility from ticker price
        logger.info("  Computing return features...")
        ret_features = compute_return_features(ticker_bars["price"])

        # Temporal features
        temporal = compute_temporal_features(ticker_bars.index)

        # Combine
        combined = ticker_bars.join(match_bars, how="left", rsuffix="_match")
        combined = combined.join(ret_features)
        combined = combined.join(temporal)

        # Resample VPIN to bar frequency
        if not vpin_series.empty:
            vpin_resampled = vpin_series.resample(interval).last().rename("vpin")
            combined = combined.join(vpin_resampled)

        combined["product_id"] = product_id

        if include_targets:
            targets = compute_targets(ticker_bars["price"])
            combined = combined.join(targets)

        return combined

    def build_lead_lag_dataset(
        self,
        start: str,
        stop: str,
        interval: str = "1min",
    ) -> pd.DataFrame:
        """Build the full lead-lag feature dataset across all pairs.

        Fetches BTC as the leading indicator, then builds per-alt features
        including cross-pair metrics. This is the primary dataset for
        Strategy 1 (BTC lead-lag altcoin trading).
        """
        logger.info("=" * 60)
        logger.info("Building lead-lag dataset [%s -> %s]", start, stop)
        logger.info("=" * 60)

        # BTC features first (the leading indicator)
        btc_features = self.build_pair_features(BTC_PAIR, start, stop, interval, include_targets=False)
        if btc_features.empty:
            logger.error("No BTC data — cannot build lead-lag dataset")
            return pd.DataFrame()

        btc_price = btc_features["price"]

        # Stablecoin pair for spread feature
        stablecoin_price = None
        try:
            logger.info("Fetching stablecoin pair %s...", STABLECOIN_PAIR)
            sc_ticker = self._client.get_ticker_data(STABLECOIN_PAIR, start=start, stop=stop)
            if not sc_ticker.empty:
                sc_ticker["price"] = pd.to_numeric(sc_ticker["price"], errors="coerce")
                sc_bars = sc_ticker.resample(interval)["price"].last().dropna()
                stablecoin_price = sc_bars
        except Exception:
            logger.warning("Could not fetch stablecoin pair")

        # Flag bars where BTC moved significantly (>0.3% in 5 bars)
        btc_ret_5 = np.log(btc_price).diff(5)
        btc_moved = btc_ret_5.abs() > 0.003

        all_alt_frames = []

        for alt_pair in ALT_PAIRS:
            logger.info("Processing alt pair: %s", alt_pair)
            alt_features = self.build_pair_features(alt_pair, start, stop, interval)
            if alt_features.empty:
                continue

            alt_price = alt_features["price"]

            # Cross-pair features
            cross = compute_cross_pair_features(btc_price, alt_price, stablecoin_price)

            # BTC microstructure features (prefix to avoid collision)
            btc_cols = {
                "spread_bps_mean": "btc_spread_bps_mean",
                "volatility_15": "btc_volatility_15",
                "volatility_60": "btc_volatility_60",
                "tick_count": "btc_tick_count",
            }
            btc_context = btc_features[[c for c in btc_cols.keys() if c in btc_features.columns]].rename(
                columns=btc_cols
            )

            combined = alt_features.join(cross).join(btc_context)
            combined["btc_moved"] = btc_moved.reindex(combined.index, fill_value=False)
            combined["product_id"] = alt_pair

            all_alt_frames.append(combined)

        if not all_alt_frames:
            return pd.DataFrame()

        dataset = pd.concat(all_alt_frames)
        dataset.sort_index(inplace=True)

        logger.info(
            "Lead-lag dataset built: %d rows, %d columns, %d pairs",
            len(dataset), len(dataset.columns), len(all_alt_frames),
        )
        return dataset

    def build_swing_dataset(
        self,
        start: str,
        stop: str,
        interval: str = "4h",
    ) -> pd.DataFrame:
        """Build feature dataset for Strategy 2 (regime-aware swing trading).

        Aggregates at 4h intervals with richer microstructure and cross-pair
        correlation features.
        """
        logger.info("=" * 60)
        logger.info("Building swing dataset [%s -> %s] @ %s", start, stop, interval)
        logger.info("=" * 60)

        all_frames = []
        price_dict: dict[str, pd.Series] = {}

        for pair in PRODUCT_IDS:
            logger.info("Processing pair: %s", pair)
            features = self.build_pair_features(pair, start, stop, interval)
            if features.empty:
                continue
            price_dict[pair] = features["price"]
            all_frames.append(features)

        if not all_frames:
            return pd.DataFrame()

        dataset = pd.concat(all_frames)

        # Cross-pair correlation regime features
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
        logger.info("Swing dataset built: %d rows, %d columns", len(dataset), len(dataset.columns))
        return dataset


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build ML feature datasets")
    parser.add_argument("--start", default="2025-11-05", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--stop", default=None, help="Stop date (YYYY-MM-DD), default=now")
    parser.add_argument("--interval", default="1min", help="Bar interval (1min, 5min, 4h, etc.)")
    parser.add_argument("--out", default="data/features", help="Output directory")
    parser.add_argument("--dataset", choices=["lead-lag", "swing", "both"], default="both")
    parser.add_argument("--days", type=int, default=None, help="Only process last N days (for testing)")
    parser.add_argument("--from-local", type=str, default=None,
                        help="Read from local parquet dir instead of InfluxDB (e.g. data/raw)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = args.start
    if args.days:
        from datetime import datetime, timedelta, timezone
        start = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")

    if args.from_local:
        from src.data.influx_client import LocalParquetClient
        client = LocalParquetClient(args.from_local)
    else:
        client = HesiodInfluxClient()

    builder = FeatureBuilder(client)
    try:
        if args.dataset in ("lead-lag", "both"):
            df = builder.build_lead_lag_dataset(start, args.stop, args.interval)
            if not df.empty:
                path = out_dir / "lead_lag_features.parquet"
                df.to_parquet(path, engine="pyarrow")
                logger.info("Saved lead-lag features: %s (%d rows)", path, len(df))

        if args.dataset in ("swing", "both"):
            df = builder.build_swing_dataset(start, args.stop, interval="4h")
            if not df.empty:
                path = out_dir / "swing_features.parquet"
                df.to_parquet(path, engine="pyarrow")
                logger.info("Saved swing features: %s (%d rows)", path, len(df))
    finally:
        builder.close()


if __name__ == "__main__":
    main()
