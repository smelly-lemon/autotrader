"""Feature engineering from OHLCV candles stored in SQLite.

Builds feature matrices for LightGBM from 1-minute candle data.
Supports single-pair and cross-pair (BTC lead-lag) feature sets.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_HORIZON = 15  # bars (minutes) forward to predict
PURGE_BARS = 20

DROP_COLS = [
    "open", "high", "low", "close", "volume",
    "product_id",
]


def compute_return_features(
    df: pd.DataFrame, lags: list[int] | None = None,
) -> pd.DataFrame:
    lags = lags or [1, 5, 15, 30, 60]
    log_close = np.log(df["close"].replace(0, np.nan))
    features = pd.DataFrame(index=df.index)

    for lag in lags:
        features[f"ret_{lag}"] = log_close.diff(lag)

    features["volatility_15"] = features["ret_1"].rolling(15, min_periods=5).std()
    features["volatility_60"] = features["ret_1"].rolling(60, min_periods=15).std()
    features["momentum_5_15"] = (
        features["ret_5"] / features["volatility_15"].replace(0, np.nan)
    )
    return features


def compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Technical indicators from OHLCV using pandas-ta."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    h = df["high"]
    l = df["low"]  # noqa: E741
    v = df["volume"]

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # Bollinger Band width
    sma20 = c.rolling(20, min_periods=10).mean()
    std20 = c.rolling(20, min_periods=10).std()
    out["bb_width"] = (2 * std20) / sma20.replace(0, np.nan)
    out["bb_position"] = (c - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)

    # ATR (normalized)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    out["atr_norm"] = atr14 / c.replace(0, np.nan)

    # EMA cross
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    out["ema_cross"] = (ema9 - ema21) / c.replace(0, np.nan)

    # Volume features
    vol_sma = v.rolling(20, min_periods=5).mean()
    out["volume_ratio"] = v / vol_sma.replace(0, np.nan)
    out["volume_change"] = v.pct_change(5)

    # Price position within bar range
    bar_range = (h - l).replace(0, np.nan)
    out["close_position"] = (c - l) / bar_range

    return out


def compute_temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    features = pd.DataFrame(index=index)
    hour = index.hour
    dow = index.dayofweek
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return features


def compute_targets(
    df: pd.DataFrame, horizons: list[int] | None = None,
) -> pd.DataFrame:
    horizons = horizons or [5, 15, 60]
    log_close = np.log(df["close"].replace(0, np.nan))
    targets = pd.DataFrame(index=df.index)

    for h in horizons:
        fwd = log_close.shift(-h) - log_close
        targets[f"fwd_ret_{h}"] = fwd
        targets[f"direction_{h}"] = (fwd > 0).astype(int)

    return targets


def build_single_pair_features(
    df: pd.DataFrame, include_targets: bool = True,
) -> pd.DataFrame:
    """Build feature matrix for one pair's OHLCV data."""
    ret = compute_return_features(df)
    ta = compute_ta_features(df)
    temporal = compute_temporal_features(df.index)

    combined = pd.concat([df, ret, ta, temporal], axis=1)

    if include_targets:
        targets = compute_targets(df)
        combined = pd.concat([combined, targets], axis=1)

    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined


def build_cross_pair_features(
    btc_df: pd.DataFrame, alt_df: pd.DataFrame,
) -> pd.DataFrame:
    """BTC-leading features aligned to the alt pair's index."""
    features = pd.DataFrame(index=alt_df.index)

    btc_log = np.log(btc_df["close"].replace(0, np.nan))
    alt_log = np.log(alt_df["close"].replace(0, np.nan))

    btc_ret_1 = btc_log.diff(1).reindex(alt_df.index)
    btc_ret_5 = btc_log.diff(5).reindex(alt_df.index)
    btc_ret_15 = btc_log.diff(15).reindex(alt_df.index)
    alt_ret_1 = alt_log.diff(1)
    alt_ret_5 = alt_log.diff(5)

    features["btc_ret_1"] = btc_ret_1
    features["btc_ret_5"] = btc_ret_5
    features["btc_ret_15"] = btc_ret_15

    features["absorption_5"] = alt_ret_5 / btc_ret_5.replace(0, np.nan)
    features["absorption_5"] = features["absorption_5"].clip(-10, 10)

    cov = btc_ret_1.rolling(60, min_periods=20).cov(alt_ret_1)
    var = btc_ret_1.rolling(60, min_periods=20).var()
    features["rolling_beta"] = (cov / var.replace(0, np.nan)).clip(-5, 5)
    features["rolling_corr"] = btc_ret_1.rolling(60, min_periods=20).corr(alt_ret_1)

    # BTC momentum as a leading indicator
    btc_vol = btc_ret_1.rolling(15, min_periods=5).std()
    features["btc_momentum"] = btc_ret_5 / btc_vol.replace(0, np.nan)
    features["btc_momentum"] = features["btc_momentum"].clip(-5, 5)

    # How much the alt lags BTC (lagged correlation)
    features["btc_lead_1"] = btc_ret_1.shift(1)
    features["btc_lead_3"] = btc_ret_1.shift(3)
    features["btc_lead_5"] = btc_ret_1.shift(5)

    return features.replace([np.inf, -np.inf], np.nan)


def build_training_dataset(
    store, pairs: list[str], target_col: str = "direction_15",
) -> tuple[pd.DataFrame, pd.Series]:
    """Build full training dataset from SQLite candles.

    Returns (X, y) ready for LightGBM.
    """
    from src.data.store import TradeStore
    if not isinstance(store, TradeStore):
        raise TypeError("Expected TradeStore instance")

    btc_pair = next((p for p in pairs if "BTC" in p), None)
    btc_df = store.get_all_candles_df(btc_pair, "1m") if btc_pair else pd.DataFrame()

    all_frames = []

    for pair in pairs:
        df = store.get_all_candles_df(pair, "1m")
        if df.empty or len(df) < 200:
            logger.warning("Skipping %s: only %d candles", pair, len(df))
            continue

        features = build_single_pair_features(df, include_targets=True)

        if not btc_df.empty and pair != btc_pair:
            cross = build_cross_pair_features(btc_df, df)
            features = features.join(cross)

        features["product_id"] = pair
        all_frames.append(features)

    if not all_frames:
        return pd.DataFrame(), pd.Series(dtype=float)

    dataset = pd.concat(all_frames).sort_index()
    dataset = dataset.dropna(subset=[target_col])

    y = dataset[target_col].astype(int)

    target_cols = [c for c in dataset.columns
                   if c.startswith("fwd_ret_") or c.startswith("direction_")]
    X = dataset.drop(columns=[c for c in DROP_COLS + target_cols if c in dataset.columns])

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    X = X.replace([np.inf, -np.inf], np.nan)

    logger.info("Dataset: %d rows, %d features, %.1f%% positive",
                len(X), len(X.columns), y.mean() * 100)
    return X, y


def build_live_features(
    store, pairs: list[str], lookback: int = 200,
) -> dict[str, pd.DataFrame]:
    """Build feature vectors for the most recent candles (live prediction).

    Returns a dict of {pair: feature_row_DataFrame} for each pair.
    """
    from src.data.store import TradeStore

    btc_pair = next((p for p in pairs if "BTC" in p), None)
    btc_df = store.get_candles(btc_pair, "1m", lookback) if btc_pair else pd.DataFrame()

    result = {}
    for pair in pairs:
        df = store.get_candles(pair, "1m", lookback)
        if df.empty or len(df) < 100:
            continue

        features = build_single_pair_features(df, include_targets=False)

        if not btc_df.empty and pair != btc_pair:
            cross = build_cross_pair_features(btc_df, df)
            features = features.join(cross)

        features = features.drop(
            columns=[c for c in DROP_COLS if c in features.columns], errors="ignore",
        )
        non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            features = features.drop(columns=non_numeric)
        features = features.replace([np.inf, -np.inf], np.nan)

        # Take the last row as the current prediction input
        if not features.empty:
            result[pair] = features.iloc[[-1]]

    return result
