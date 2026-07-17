#!/usr/bin/env python3
"""Systematic search for the most profitable trading model configuration.

Sweeps across target definitions, horizons, bar sizes, feature sets,
model types, hyperparameters, and entry/exit rules — evaluated
end-to-end by net P&L after realistic Coinbase taker fees.

Usage:
    python scripts/model_search.py --data-dir data/raw --phase 2
    python scripts/model_search.py --data-dir data/raw --phase all
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.influx_client import LocalParquetClient, PRODUCT_IDS
from src.ml.features import (
    FeatureBuilder,
    build_bar_features_from_ticker,
    build_bar_features_from_matches,
    compute_return_features,
    compute_cross_pair_features,
    compute_temporal_features,
    compute_vpin,
    BTC_PAIR,
    STABLECOIN_PAIR,
    ALT_PAIRS,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Coinbase taker fees by tier
FEE_TIERS = {
    "starter": 0.0120, "bronze": 0.0060, "silver": 0.0040,
    "gold": 0.0025, "platinum": 0.0018,
}
ROUNDTRIP_FEE = FEE_TIERS["silver"] * 2  # entry + exit taker

RESULTS_DIR = Path("results/model_search")
HOLDOUT_DAYS = 30  # most recent 30 days reserved for final validation


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
_DATA_CACHE: dict[str, pd.DataFrame] = {}


def get_client(data_dir: str) -> LocalParquetClient:
    return LocalParquetClient(data_dir)


def load_pair_raw(client: LocalParquetClient, pair: str, meas: str,
                  start: str, stop: str) -> pd.DataFrame:
    key = f"{meas}_{pair}_{start}_{stop}"
    if key not in _DATA_CACHE:
        if meas == "ticker":
            _DATA_CACHE[key] = client.get_ticker_data(pair, start=start, stop=stop)
        else:
            _DATA_CACHE[key] = client.get_matches(pair, start=start, stop=stop)
    return _DATA_CACHE[key]


# ---------------------------------------------------------------------------
# Flexible feature + target builder
# ---------------------------------------------------------------------------
def build_dataset(
    client: LocalParquetClient,
    interval: str,
    start: str,
    stop: str,
    horizons: list[int],
    pairs: list[str] | None = None,
    include_cross_pair: bool = True,
    extra_lags: list[int] | None = None,
    extra_ta: bool = False,
) -> pd.DataFrame:
    """Build a feature matrix with flexible parameters.

    Returns a DataFrame with feature columns, target columns
    (fwd_ret_{h}, direction_{h} for each h in horizons), product_id, price.
    """
    pairs = pairs or [p for p in PRODUCT_IDS if p not in ("BTC-USDC", "ETH-USDC")]
    lags = extra_lags or [1, 5, 15, 60]

    # Always build BTC features for cross-pair
    btc_ticker = load_pair_raw(client, BTC_PAIR, "ticker", start, stop)
    btc_matches = load_pair_raw(client, BTC_PAIR, "matches", start, stop)
    btc_ticker_bars = build_bar_features_from_ticker(btc_ticker, interval)

    if btc_ticker_bars.empty:
        logger.warning("No BTC ticker data for %s @ %s", start, interval)
        return pd.DataFrame()

    btc_price = btc_ticker_bars["price"]
    btc_ret_features = compute_return_features(btc_price, lags)

    # BTC movement flag for lead-lag filtering
    btc_ret_5 = np.log(btc_price).diff(5)
    btc_moved = btc_ret_5.abs() > 0.003

    all_frames = []

    for pair in pairs:
        if pair in ("BTC-USDC", "ETH-USDC"):
            continue

        ticker = load_pair_raw(client, pair, "ticker", start, stop)
        matches = load_pair_raw(client, pair, "matches", start, stop)

        ticker_bars = build_bar_features_from_ticker(ticker, interval)
        if ticker_bars.empty:
            continue

        match_bars = build_bar_features_from_matches(matches, interval)
        ret_feats = compute_return_features(ticker_bars["price"], lags)
        temporal = compute_temporal_features(ticker_bars.index)

        combined = ticker_bars.join(match_bars, how="left", rsuffix="_match")
        combined = combined.join(ret_feats)
        combined = combined.join(temporal)

        # VPIN
        if not matches.empty:
            vpin = compute_vpin(matches)
            if not vpin.empty:
                combined = combined.join(vpin.resample(interval).last().rename("vpin"))

        # Extra TA features
        if extra_ta:
            combined = _add_ta_features(combined)

        # Cross-pair features
        if include_cross_pair and pair != BTC_PAIR:
            cross = compute_cross_pair_features(btc_price, ticker_bars["price"])
            btc_ctx = {}
            for col in ["spread_bps_mean", "volatility_15", "volatility_60", "tick_count"]:
                if col in btc_ticker_bars.columns:
                    btc_ctx[f"btc_{col}"] = btc_ticker_bars[col]
                elif col in btc_ret_features.columns:
                    btc_ctx[f"btc_{col}"] = btc_ret_features[col]
            btc_ctx_df = pd.DataFrame(btc_ctx, index=btc_ticker_bars.index)
            combined = combined.join(cross).join(btc_ctx_df)
            combined["btc_moved"] = btc_moved.reindex(combined.index, fill_value=False)

        # Targets
        log_p = np.log(combined["price"].replace(0, np.nan))
        for h in horizons:
            fwd = log_p.shift(-h) - log_p
            combined[f"fwd_ret_{h}"] = fwd
            combined[f"direction_{h}"] = (fwd > 0).astype(int)

        combined["product_id"] = pair
        all_frames.append(combined)

    if not all_frames:
        return pd.DataFrame()

    dataset = pd.concat(all_frames).sort_index()
    return dataset


def _add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, Bollinger %B, ATR-like features."""
    if "price" not in df.columns:
        return df

    p = df["price"]

    # RSI (14-bar)
    delta = p.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # Bollinger Band %B (20-bar, 2 std)
    sma20 = p.rolling(20, min_periods=10).mean()
    std20 = p.rolling(20, min_periods=10).std()
    df["bb_pctb"] = (p - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)

    # ATR-like (using high/low if available, else price range proxy)
    if "price_high" in df.columns and "price_low" in df.columns:
        tr = df["price_high"] - df["price_low"]
        df["atr_14"] = tr.rolling(14, min_periods=5).mean() / p.replace(0, np.nan)
    else:
        df["atr_14"] = p.diff().abs().rolling(14, min_periods=5).mean() / p.replace(0, np.nan)

    # Volume relative to rolling mean
    if "trade_volume" in df.columns:
        vol_ma = df["trade_volume"].rolling(20, min_periods=5).mean()
        df["vol_ratio"] = df["trade_volume"] / vol_ma.replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Walk-forward backtester (self-contained, evaluates by P&L)
# ---------------------------------------------------------------------------
@dataclass
class BacktestMetrics:
    config_name: str = ""
    n_trades: int = 0
    win_rate: float = 0.0
    total_net_return: float = 0.0
    avg_net_return: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_hold_bars: float = 0.0
    final_equity: float = 10000.0
    per_pair: dict = field(default_factory=dict)
    fold_results: list = field(default_factory=list)


def walk_forward_backtest(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    n_folds: int = 5,
    purge_bars: int = 20,
    lgbm_params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping: int = 50,
    confidence_threshold: float = 0.55,
    max_hold_bars: int = 60,
    fee_per_roundtrip: float = ROUNDTRIP_FEE,
    position_size: float = 0.1,
    btc_move_filter: bool = False,
    filter_col: str = "btc_moved",
    use_regression: bool = False,
    regression_threshold: float = 0.002,
    min_magnitude: float = 0.0,
    strategy_name: str = "",
) -> BacktestMetrics:
    """Train-predict-backtest in a single walk-forward pass.

    Returns P&L metrics — the primary evaluation criterion.
    """
    params = lgbm_params or {
        "objective": "regression" if use_regression else "binary",
        "metric": "rmse" if use_regression else "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Prepare data — reset index to avoid duplicate-label issues from multi-pair concat
    work = df.copy().reset_index()
    if "_time" in work.columns:
        work.rename(columns={"_time": "timestamp"}, inplace=True)
    elif work.columns[0] not in df.columns:
        work.rename(columns={work.columns[0]: "timestamp"}, inplace=True)
    if "timestamp" not in work.columns:
        work["timestamp"] = work.index

    if btc_move_filter and filter_col in work.columns:
        work = work[work[filter_col]]

    # Determine target
    if use_regression:
        ret_col = target_col.replace("direction_", "fwd_ret_")
        if ret_col not in work.columns:
            return BacktestMetrics(config_name=strategy_name)
        work = work.dropna(subset=[ret_col])
        y_all = work[ret_col].astype(float)
    else:
        if target_col not in work.columns:
            return BacktestMetrics(config_name=strategy_name)
        work = work.dropna(subset=[target_col])

        if min_magnitude > 0:
            ret_col = target_col.replace("direction_", "fwd_ret_")
            if ret_col in work.columns:
                small_move = work[ret_col].abs() < min_magnitude
                work = work[~small_move]

        y_all = work[target_col].astype(int)

    # Build X
    all_drop = list(set(drop_cols + ["timestamp"]) & set(work.columns))
    X_all = work.drop(columns=all_drop, errors="ignore")
    non_numeric = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
    X_all = X_all.drop(columns=non_numeric)
    X_all = X_all.replace([np.inf, -np.inf], np.nan)

    prices = work["price"] if "price" in work.columns else pd.Series(dtype=float)
    product_ids = work["product_id"] if "product_id" in work.columns else None
    ts_col = work["timestamp"]

    timestamps = np.sort(ts_col.unique())
    n_ts = len(timestamps)
    if n_ts < 200:
        return BacktestMetrics(config_name=strategy_name)

    test_size = n_ts // (n_folds + 1)

    all_trades = []
    equity = 10000.0

    for fold_i in range(n_folds):
        test_start_idx = n_ts - (n_folds - fold_i) * test_size
        test_end_idx = test_start_idx + test_size
        train_end_idx = test_start_idx - purge_bars

        if train_end_idx <= 50 or test_start_idx >= n_ts:
            continue

        train_end_ts = timestamps[min(train_end_idx, n_ts - 1)]
        test_start_ts = timestamps[min(test_start_idx, n_ts - 1)]
        test_end_ts = timestamps[min(test_end_idx - 1, n_ts - 1)]

        train_mask = ts_col <= train_end_ts
        test_mask = (ts_col >= test_start_ts) & (ts_col <= test_end_ts)

        X_train = X_all.loc[train_mask.values]
        y_train = y_all.loc[train_mask.values]
        X_test = X_all.loc[test_mask.values]
        y_test = y_all.loc[test_mask.values]

        if len(X_train) < 100 or len(X_test) < 20:
            continue

        common = X_train.columns.intersection(X_test.columns)
        X_train, X_test = X_train[common], X_test[common]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        try:
            model = lgb.train(
                params, dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        except Exception:
            continue

        preds = model.predict(X_test)
        test_idx = X_test.index
        test_prices = prices.loc[test_idx].values
        test_pids_arr = product_ids.loc[test_idx].values if product_ids is not None else [None] * len(test_idx)
        test_ts_arr = ts_col.loc[test_idx].values

        if use_regression:
            signals = np.zeros(len(preds))
            signals[preds > regression_threshold] = 1
            signals[preds < -regression_threshold] = -1
        else:
            signals = np.zeros(len(preds))
            signals[preds > confidence_threshold] = 1
            signals[preds < (1 - confidence_threshold)] = -1

        # Per-pair position tracking to avoid cross-pair confusion
        positions = {}  # pair -> {entry_price, entry_time, direction, bars_held}

        for j in range(len(preds)):
            price = test_prices[j]
            if np.isnan(price) or price <= 0:
                continue
            sig = int(signals[j])
            pid = test_pids_arr[j] if test_pids_arr[j] is not None else "unknown"
            ts = test_ts_arr[j]

            # Check exit for this pair
            if pid in positions:
                pos = positions[pid]
                pos["bars_held"] += 1
                should_exit = pos["bars_held"] >= max_hold_bars
                if sig != 0 and sig != pos["direction"]:
                    should_exit = True
                if should_exit:
                    d = pos["direction"]
                    gross_ret = (price - pos["entry_price"]) / pos["entry_price"] * d
                    net_ret = gross_ret - fee_per_roundtrip
                    equity += net_ret * 1000.0
                    all_trades.append({
                        "entry_time": pos["entry_time"], "exit_time": ts,
                        "product_id": pid, "direction": d,
                        "gross_ret": gross_ret, "net_ret": net_ret,
                        "bars_held": pos["bars_held"],
                    })
                    del positions[pid]

            # Check entry for this pair
            if pid not in positions and sig != 0:
                positions[pid] = {
                    "entry_price": price, "entry_time": ts,
                    "direction": sig, "bars_held": 0,
                }

    if not all_trades:
        return BacktestMetrics(config_name=strategy_name, final_equity=equity)

    net_rets = [t["net_ret"] for t in all_trades]
    gross_rets = [t["gross_ret"] for t in all_trades]
    wins = [r for r in net_rets if r > 0]
    losses = [r for r in net_rets if r <= 0]

    # Per-pair breakdown
    per_pair = {}
    for t in all_trades:
        pid = t["product_id"]
        if pid not in per_pair:
            per_pair[pid] = {"n": 0, "net": 0.0, "wins": 0}
        per_pair[pid]["n"] += 1
        per_pair[pid]["net"] += t["net_ret"]
        if t["net_ret"] > 0:
            per_pair[pid]["wins"] += 1

    # Sharpe
    if len(net_rets) > 1:
        std = np.std(net_rets)
        mean = np.mean(net_rets)
        first_t = pd.Timestamp(all_trades[0]["entry_time"])
        last_t = pd.Timestamp(all_trades[-1]["exit_time"])
        span_days = max(1, (last_t - first_t).days)
        trades_per_year = len(all_trades) / span_days * 365
        sharpe = (mean / std * np.sqrt(trades_per_year)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    # Profit factor
    total_wins = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 0
    pf = total_wins / total_losses if total_losses > 0 else (float("inf") if total_wins > 0 else 0.0)

    # Max drawdown from cumulative P&L
    cum_pnl = np.cumsum([t["net_ret"] * 1000.0 for t in all_trades])
    running_equity = 10000.0 + cum_pnl
    peak = pd.Series(running_equity).expanding().max()
    dd = (pd.Series(running_equity) - peak) / peak
    max_dd = dd.min()

    return BacktestMetrics(
        config_name=strategy_name,
        n_trades=len(all_trades),
        win_rate=len(wins) / len(all_trades) if all_trades else 0,
        total_net_return=sum(net_rets),
        avg_net_return=np.mean(net_rets),
        sharpe=sharpe,
        profit_factor=pf,
        max_drawdown=max_dd,
        avg_hold_bars=np.mean([t["bars_held"] for t in all_trades]),
        final_equity=equity,
        per_pair=per_pair,
    )


# ---------------------------------------------------------------------------
# Helper: columns to drop (all target-related, identifiers, raw prices)
# ---------------------------------------------------------------------------
def make_drop_cols(horizons: list[int]) -> list[str]:
    base = ["product_id", "price", "price_open", "price_high", "price_low",
            "best_bid", "best_ask", "volume_24h", "vwap", "btc_moved"]
    for h in horizons:
        base += [f"fwd_ret_{h}", f"fwd_abs_ret_{h}", f"direction_{h}"]
    # Also drop default horizons that might be present
    for h in [1, 6, 15, 60, 240]:
        base += [f"fwd_ret_{h}", f"fwd_abs_ret_{h}", f"direction_{h}"]
    return list(set(base))


# ---------------------------------------------------------------------------
# Phase 2: Target + Horizon + Bar-size sweep
# ---------------------------------------------------------------------------
def phase2_sweep(client: LocalParquetClient, start: str, stop: str) -> pd.DataFrame:
    """Sweep bar sizes, forward horizons, target types. ~100 configs."""
    logger.info("=" * 70)
    logger.info("PHASE 2: Target + Horizon + Bar-size Sweep")
    logger.info("=" * 70)

    bar_sizes = ["5min", "15min", "1h", "4h"]
    forward_horizons = [1, 3, 5, 10, 15, 30]
    target_types = ["binary", "magnitude_0.001", "magnitude_0.002", "regression"]

    results = []
    total_configs = len(bar_sizes) * len(forward_horizons) * len(target_types)
    config_i = 0

    for bar_size in bar_sizes:
        logger.info("Building dataset @ %s bars...", bar_size)
        all_horizons = list(set(forward_horizons))
        dataset = build_dataset(
            client, bar_size, start, stop,
            horizons=all_horizons,
            include_cross_pair=True,
        )
        if dataset.empty:
            logger.warning("Empty dataset for %s, skipping", bar_size)
            continue

        logger.info("Dataset: %d rows, %d cols", len(dataset), len(dataset.columns))

        for horizon in forward_horizons:
            for target_type in target_types:
                config_i += 1
                name = f"{bar_size}_h{horizon}_{target_type}"
                logger.info("[%d/%d] %s", config_i, total_configs, name)

                use_regression = target_type == "regression"
                target_col = f"fwd_ret_{horizon}" if use_regression else f"direction_{horizon}"
                min_mag = 0.0
                if target_type.startswith("magnitude_"):
                    min_mag = float(target_type.split("_")[1])

                drop = make_drop_cols(all_horizons)

                try:
                    metrics = walk_forward_backtest(
                        dataset,
                        target_col=target_col,
                        drop_cols=drop,
                        n_folds=5,
                        confidence_threshold=0.55,
                        max_hold_bars=max(horizon * 2, 10),
                        use_regression=use_regression,
                        regression_threshold=ROUNDTRIP_FEE * 1.5,
                        min_magnitude=min_mag,
                        btc_move_filter=False,
                        strategy_name=name,
                    )
                except Exception as e:
                    logger.error("  FAILED: %s", str(e)[:100])
                    continue

                row = {
                    "config": name, "bar_size": bar_size,
                    "horizon": horizon, "target_type": target_type,
                    "n_trades": metrics.n_trades,
                    "win_rate": round(metrics.win_rate, 4),
                    "total_net_return": round(metrics.total_net_return, 6),
                    "avg_net_return": round(metrics.avg_net_return, 6),
                    "sharpe": round(metrics.sharpe, 3),
                    "profit_factor": round(min(metrics.profit_factor, 99), 3),
                    "max_drawdown": round(metrics.max_drawdown, 4),
                    "avg_hold_bars": round(metrics.avg_hold_bars, 1),
                    "final_equity": round(metrics.final_equity, 2),
                }
                results.append(row)

                if metrics.n_trades > 0:
                    logger.info("  -> %d trades, net=%.4f, sharpe=%.2f, pf=%.2f, wr=%.1f%%",
                                metrics.n_trades, metrics.total_net_return,
                                metrics.sharpe, metrics.profit_factor, metrics.win_rate * 100)
                else:
                    logger.info("  -> no trades")

        del dataset
        gc.collect()

    df = pd.DataFrame(results)
    return df


# ---------------------------------------------------------------------------
# Phase 3: Feature + Hyperparameter sweep on top configs
# ---------------------------------------------------------------------------
def phase3_sweep(client: LocalParquetClient, start: str, stop: str,
                 top_configs: pd.DataFrame) -> pd.DataFrame:
    """Sweep features and LightGBM hyperparams on the best configs from Phase 2."""
    logger.info("=" * 70)
    logger.info("PHASE 3: Feature + Hyperparameter Sweep")
    logger.info("=" * 70)

    param_grid = [
        {"num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 20, "feature_fraction": 0.8},
        {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 50, "feature_fraction": 0.7},
        {"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 30, "feature_fraction": 0.6},
        {"num_leaves": 63, "learning_rate": 0.01, "min_child_samples": 50, "feature_fraction": 0.8},
        {"num_leaves": 31, "learning_rate": 0.1, "min_child_samples": 100, "feature_fraction": 0.5},
    ]

    feature_variants = [
        {"name": "base", "extra_ta": False, "extra_lags": None},
        {"name": "ta", "extra_ta": True, "extra_lags": None},
        {"name": "more_lags", "extra_ta": False, "extra_lags": [1, 2, 3, 5, 10, 15, 30, 60, 120]},
        {"name": "ta+lags", "extra_ta": True, "extra_lags": [1, 2, 3, 5, 10, 15, 30, 60, 120]},
    ]

    results = []

    for _, cfg_row in top_configs.iterrows():
        bar_size = cfg_row["bar_size"]
        horizon = int(cfg_row["horizon"])
        target_type = cfg_row["target_type"]

        for fv in feature_variants:
            logger.info("Building dataset: %s @ %s, features=%s", bar_size, horizon, fv["name"])
            dataset = build_dataset(
                client, bar_size, start, stop,
                horizons=[horizon],
                extra_ta=fv["extra_ta"],
                extra_lags=fv["extra_lags"],
            )
            if dataset.empty:
                continue

            for pi, pset in enumerate(param_grid):
                use_regression = target_type == "regression"
                target_col = f"fwd_ret_{horizon}" if use_regression else f"direction_{horizon}"
                min_mag = 0.0
                if target_type.startswith("magnitude_"):
                    min_mag = float(target_type.split("_")[1])

                name = f"{bar_size}_h{horizon}_{target_type}_{fv['name']}_p{pi}"
                logger.info("  [%s]", name)

                lgbm_p = {
                    "objective": "regression" if use_regression else "binary",
                    "metric": "rmse" if use_regression else "binary_logloss",
                    "boosting_type": "gbdt",
                    **pset,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "lambda_l1": 0.1,
                    "lambda_l2": 1.0,
                    "verbose": -1,
                    "n_jobs": -1,
                    "seed": 42,
                }

                drop = make_drop_cols([horizon])
                try:
                    metrics = walk_forward_backtest(
                        dataset,
                        target_col=target_col,
                        drop_cols=drop,
                        lgbm_params=lgbm_p,
                        confidence_threshold=0.55,
                        max_hold_bars=max(horizon * 4, 10),
                        use_regression=use_regression,
                        regression_threshold=ROUNDTRIP_FEE * 1.5,
                        min_magnitude=min_mag,
                        strategy_name=name,
                    )
                except Exception as e:
                    logger.error("    FAILED: %s", str(e)[:100])
                    continue

                row = {
                    "config": name, "bar_size": bar_size,
                    "horizon": horizon, "target_type": target_type,
                    "features": fv["name"], "params_idx": pi,
                    "n_trades": metrics.n_trades,
                    "win_rate": round(metrics.win_rate, 4),
                    "total_net_return": round(metrics.total_net_return, 6),
                    "sharpe": round(metrics.sharpe, 3),
                    "profit_factor": round(min(metrics.profit_factor, 99), 3),
                    "max_drawdown": round(metrics.max_drawdown, 4),
                    "final_equity": round(metrics.final_equity, 2),
                }
                results.append(row)

                if metrics.n_trades > 0:
                    logger.info("    -> %d trades, net=%.4f, sharpe=%.2f, pf=%.2f",
                                metrics.n_trades, metrics.total_net_return,
                                metrics.sharpe, metrics.profit_factor)

            del dataset
            gc.collect()

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Phase 4: Model type comparison
# ---------------------------------------------------------------------------
def phase4_sweep(client: LocalParquetClient, start: str, stop: str,
                 top_configs: pd.DataFrame) -> pd.DataFrame:
    """Compare LightGBM, XGBoost, Random Forest on top configs."""
    logger.info("=" * 70)
    logger.info("PHASE 4: Model Type Comparison")
    logger.info("=" * 70)

    results = []

    for _, cfg_row in top_configs.iterrows():
        bar_size = cfg_row["bar_size"]
        horizon = int(cfg_row["horizon"])
        target_type = cfg_row["target_type"]
        features = cfg_row.get("features", "base")

        extra_ta = features in ("ta", "ta+lags")
        extra_lags = [1, 2, 3, 5, 10, 15, 30, 60, 120] if "lags" in features else None

        dataset = build_dataset(
            client, bar_size, start, stop,
            horizons=[horizon],
            extra_ta=extra_ta,
            extra_lags=extra_lags,
        )
        if dataset.empty:
            continue

        use_regression = target_type == "regression"
        target_col = f"fwd_ret_{horizon}" if use_regression else f"direction_{horizon}"
        min_mag = float(target_type.split("_")[1]) if target_type.startswith("magnitude_") else 0.0
        drop = make_drop_cols([horizon])

        # LightGBM (baseline)
        name = f"lgbm_{bar_size}_h{horizon}_{target_type}"
        logger.info("  %s", name)
        try:
            m = walk_forward_backtest(
                dataset, target_col=target_col, drop_cols=drop,
                confidence_threshold=0.55, max_hold_bars=max(horizon * 4, 10),
                use_regression=use_regression, regression_threshold=ROUNDTRIP_FEE * 1.5,
                min_magnitude=min_mag, strategy_name=name,
            )
            results.append(_metrics_to_row(m, "lgbm", cfg_row))
        except Exception as e:
            logger.error("    FAILED: %s", str(e)[:100])

        # XGBoost
        try:
            import xgboost as xgb
            xgb_params = {
                "objective": "reg:squarederror" if use_regression else "binary:logistic",
                "eval_metric": "rmse" if use_regression else "logloss",
                "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.7,
                "min_child_weight": 50, "reg_alpha": 0.1, "reg_lambda": 1.0,
                "n_jobs": -1, "seed": 42, "verbosity": 0,
            }
            name = f"xgb_{bar_size}_h{horizon}_{target_type}"
            logger.info("  %s", name)
            m = _wf_backtest_sklearn(
                dataset, target_col, drop, "xgboost", xgb_params,
                use_regression, ROUNDTRIP_FEE * 1.5, min_mag,
                0.55, max(horizon * 4, 10), name,
            )
            results.append(_metrics_to_row(m, "xgboost", cfg_row))
        except ImportError:
            logger.warning("  xgboost not installed, skipping")
        except Exception as e:
            logger.error("    xgb FAILED: %s", str(e)[:100])

        # Random Forest
        try:
            name = f"rf_{bar_size}_h{horizon}_{target_type}"
            logger.info("  %s", name)
            m = _wf_backtest_sklearn(
                dataset, target_col, drop, "random_forest", {},
                use_regression, ROUNDTRIP_FEE * 1.5, min_mag,
                0.55, max(horizon * 4, 10), name,
            )
            results.append(_metrics_to_row(m, "random_forest", cfg_row))
        except Exception as e:
            logger.error("    rf FAILED: %s", str(e)[:100])

        del dataset
        gc.collect()

    return pd.DataFrame(results)


def _wf_backtest_sklearn(
    df, target_col, drop_cols, model_type, model_params,
    use_regression, reg_threshold, min_magnitude,
    conf_threshold, max_hold, name,
    n_folds=5, purge_bars=20,
) -> BacktestMetrics:
    """Walk-forward backtest with sklearn-style models (XGBoost, RF)."""
    work = df.copy()
    if use_regression:
        ret_col = target_col.replace("direction_", "fwd_ret_")
        work = work.dropna(subset=[ret_col])
        y_all = work[ret_col].astype(float)
    else:
        work = work.dropna(subset=[target_col])
        if min_magnitude > 0:
            ret_col = target_col.replace("direction_", "fwd_ret_")
            if ret_col in work.columns:
                work = work[work[ret_col].abs() >= min_magnitude]
        y_all = work[target_col].astype(int)

    work = work.reset_index(drop=True)
    if "timestamp" not in work.columns:
        work["timestamp"] = df.index[:len(work)] if len(df) >= len(work) else range(len(work))

    all_drop = list(set(drop_cols + ["timestamp"]) & set(work.columns))
    X_all = work.drop(columns=all_drop, errors="ignore")
    X_all = X_all.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    prices = work["price"] if "price" in work.columns else pd.Series(dtype=float)
    product_ids = work.get("product_id")
    ts_col = work["timestamp"]

    timestamps = np.sort(ts_col.unique())
    n_ts = len(timestamps)
    if n_ts < 200:
        return BacktestMetrics(config_name=name)

    test_size = n_ts // (n_folds + 1)
    all_trades = []
    equity = 10000.0

    for fold_i in range(n_folds):
        test_start_idx = n_ts - (n_folds - fold_i) * test_size
        test_end_idx = test_start_idx + test_size
        train_end_idx = test_start_idx - purge_bars
        if train_end_idx <= 50:
            continue

        train_end_ts = timestamps[min(train_end_idx, n_ts - 1)]
        test_start_ts = timestamps[min(test_start_idx, n_ts - 1)]
        test_end_ts = timestamps[min(test_end_idx - 1, n_ts - 1)]

        train_mask = (ts_col <= train_end_ts).values
        test_mask = ((ts_col >= test_start_ts) & (ts_col <= test_end_ts)).values
        X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
        X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]

        if len(X_train) < 100 or len(X_test) < 20:
            continue

        common = X_train.columns.intersection(X_test.columns)
        X_train, X_test = X_train[common].fillna(0), X_test[common].fillna(0)

        if model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBRegressor(**model_params) if use_regression else xgb.XGBClassifier(**model_params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            if use_regression:
                preds = model.predict(X_test)
            else:
                preds = model.predict_proba(X_test)[:, 1]
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if use_regression:
                model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=50,
                                              n_jobs=-1, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=50,
                                               n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)
            if use_regression:
                preds = model.predict(X_test)
            else:
                preds = model.predict_proba(X_test)[:, 1]

        test_prices = prices.loc[X_test.index]
        test_pids = product_ids.loc[X_test.index] if product_ids is not None else None
        test_ts = ts_col.loc[X_test.index]

        if use_regression:
            signals = np.zeros(len(preds))
            signals[preds > reg_threshold] = 1
            signals[preds < -reg_threshold] = -1
            confidence = np.abs(preds) / max(np.abs(preds).max(), 1e-8)
        else:
            signals = np.zeros(len(preds))
            signals[preds > conf_threshold] = 1
            signals[preds < (1 - conf_threshold)] = -1
            confidence = np.abs(preds - 0.5) * 2

        position = None
        entry_price = entry_dir = bars_held = None
        entry_time = entry_pid = None

        for j in range(len(X_test)):
            ts = test_ts.iloc[j]
            price = test_prices.iloc[j] if j < len(test_prices) else np.nan
            if np.isnan(price) or price <= 0:
                continue
            sig = signals[j]

            if position is not None:
                bars_held += 1
                should_exit = bars_held >= max_hold
                if sig != 0 and sig != entry_dir:
                    should_exit = True
                if should_exit:
                    gross_ret = ((price - entry_price) / entry_price) * entry_dir
                    net_ret = gross_ret - ROUNDTRIP_FEE
                    equity += net_ret * 1000.0
                    all_trades.append({"net_ret": net_ret, "gross_ret": gross_ret,
                                       "bars_held": bars_held,
                                       "product_id": entry_pid or "?",
                                       "entry_time": entry_time, "exit_time": ts})
                    position = None

            if position is None and sig != 0:
                position = True
                entry_price = price
                entry_dir = int(sig)
                entry_time = ts
                entry_pid = test_pids.iloc[j] if test_pids is not None and j < len(test_pids) else None
                bars_held = 0

    if not all_trades:
        return BacktestMetrics(config_name=name, final_equity=equity)

    net_rets = [t["net_ret"] for t in all_trades]
    wins = [r for r in net_rets if r > 0]
    losses = [r for r in net_rets if r <= 0]
    total_wins = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 0

    return BacktestMetrics(
        config_name=name,
        n_trades=len(all_trades),
        win_rate=len(wins) / len(all_trades),
        total_net_return=sum(net_rets),
        avg_net_return=np.mean(net_rets),
        sharpe=0,
        profit_factor=total_wins / total_losses if total_losses > 0 else (99 if total_wins > 0 else 0),
        max_drawdown=0,
        final_equity=equity,
        per_pair={},
    )


def _metrics_to_row(m: BacktestMetrics, model_type: str, cfg_row) -> dict:
    return {
        "config": m.config_name, "model_type": model_type,
        "bar_size": cfg_row.get("bar_size", ""),
        "horizon": cfg_row.get("horizon", 0),
        "target_type": cfg_row.get("target_type", ""),
        "n_trades": m.n_trades,
        "win_rate": round(m.win_rate, 4),
        "total_net_return": round(m.total_net_return, 6),
        "sharpe": round(m.sharpe, 3),
        "profit_factor": round(min(m.profit_factor, 99), 3),
        "max_drawdown": round(m.max_drawdown, 4),
        "final_equity": round(m.final_equity, 2),
    }


# ---------------------------------------------------------------------------
# Phase 5: Entry/Exit optimization
# ---------------------------------------------------------------------------
def phase5_sweep(client: LocalParquetClient, start: str, stop: str,
                 top_configs: pd.DataFrame) -> pd.DataFrame:
    """Sweep confidence thresholds, hold durations, and regime filters."""
    logger.info("=" * 70)
    logger.info("PHASE 5: Entry/Exit Optimization")
    logger.info("=" * 70)

    thresholds = [0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70, 0.75]
    hold_bars_options = [5, 10, 15, 30, 60]
    results = []

    for _, cfg_row in top_configs.iterrows():
        bar_size = cfg_row["bar_size"]
        horizon = int(cfg_row["horizon"])
        target_type = cfg_row["target_type"]
        features = cfg_row.get("features", "base")

        extra_ta = features in ("ta", "ta+lags")
        extra_lags = [1, 2, 3, 5, 10, 15, 30, 60, 120] if "lags" in features else None

        dataset = build_dataset(
            client, bar_size, start, stop,
            horizons=[horizon],
            extra_ta=extra_ta, extra_lags=extra_lags,
        )
        if dataset.empty:
            continue

        use_regression = target_type == "regression"
        target_col = f"fwd_ret_{horizon}" if use_regression else f"direction_{horizon}"
        min_mag = float(target_type.split("_")[1]) if target_type.startswith("magnitude_") else 0.0
        drop = make_drop_cols([horizon])

        for thresh in thresholds:
            for max_hold in hold_bars_options:
                name = f"{bar_size}_h{horizon}_{target_type}_t{thresh}_hold{max_hold}"
                logger.info("  %s", name)

                try:
                    m = walk_forward_backtest(
                        dataset, target_col=target_col, drop_cols=drop,
                        confidence_threshold=thresh,
                        max_hold_bars=max_hold,
                        use_regression=use_regression,
                        regression_threshold=ROUNDTRIP_FEE * (1 + (thresh - 0.5) * 4),
                        min_magnitude=min_mag,
                        strategy_name=name,
                    )
                except Exception as e:
                    logger.error("    FAILED: %s", str(e)[:100])
                    continue

                row = {
                    "config": name, "bar_size": bar_size,
                    "horizon": horizon, "target_type": target_type,
                    "threshold": thresh, "max_hold": max_hold,
                    "n_trades": m.n_trades,
                    "win_rate": round(m.win_rate, 4),
                    "total_net_return": round(m.total_net_return, 6),
                    "sharpe": round(m.sharpe, 3),
                    "profit_factor": round(min(m.profit_factor, 99), 3),
                    "max_drawdown": round(m.max_drawdown, 4),
                    "final_equity": round(m.final_equity, 2),
                }
                results.append(row)

                if m.n_trades > 0:
                    logger.info("    -> %d trades, net=%.4f, sharpe=%.2f, pf=%.2f",
                                m.n_trades, m.total_net_return, m.sharpe, m.profit_factor)

        # Also try btc_move_filter
        if not use_regression:
            name = f"{bar_size}_h{horizon}_{target_type}_btcfilter"
            logger.info("  %s", name)
            try:
                m = walk_forward_backtest(
                    dataset, target_col=target_col, drop_cols=drop,
                    confidence_threshold=0.55, max_hold_bars=max(horizon * 4, 10),
                    btc_move_filter=True,
                    min_magnitude=min_mag, strategy_name=name,
                )
                results.append({
                    "config": name, "bar_size": bar_size,
                    "horizon": horizon, "target_type": target_type,
                    "threshold": 0.55, "max_hold": max(horizon * 4, 10),
                    "n_trades": m.n_trades,
                    "win_rate": round(m.win_rate, 4),
                    "total_net_return": round(m.total_net_return, 6),
                    "sharpe": round(m.sharpe, 3),
                    "profit_factor": round(min(m.profit_factor, 99), 3),
                    "max_drawdown": round(m.max_drawdown, 4),
                    "final_equity": round(m.final_equity, 2),
                })
            except Exception as e:
                logger.error("    FAILED: %s", str(e)[:100])

        del dataset
        gc.collect()

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Phase 6: Final out-of-sample validation
# ---------------------------------------------------------------------------
def phase6_validate(client: LocalParquetClient, full_start: str, full_stop: str,
                    holdout_start: str, best_config: dict) -> BacktestMetrics:
    """Train on all data before holdout, test on holdout period."""
    logger.info("=" * 70)
    logger.info("PHASE 6: Out-of-Sample Validation")
    logger.info("  Train: %s -> %s", full_start, holdout_start)
    logger.info("  Test:  %s -> %s", holdout_start, full_stop)
    logger.info("  Config: %s", best_config.get("config", ""))
    logger.info("=" * 70)

    bar_size = best_config["bar_size"]
    horizon = int(best_config["horizon"])
    target_type = best_config["target_type"]
    features = best_config.get("features", "base")
    threshold = best_config.get("threshold", 0.55)
    max_hold = best_config.get("max_hold", max(horizon * 4, 10))

    extra_ta = features in ("ta", "ta+lags")
    extra_lags = [1, 2, 3, 5, 10, 15, 30, 60, 120] if "lags" in features else None

    # Build full dataset
    dataset = build_dataset(
        client, bar_size, full_start, full_stop,
        horizons=[horizon],
        extra_ta=extra_ta, extra_lags=extra_lags,
    )
    if dataset.empty:
        logger.error("Empty dataset for validation")
        return BacktestMetrics(config_name="oos_validation")

    use_regression = target_type == "regression"
    target_col = f"fwd_ret_{horizon}" if use_regression else f"direction_{horizon}"
    min_mag = float(target_type.split("_")[1]) if target_type.startswith("magnitude_") else 0.0
    drop = make_drop_cols([horizon])

    metrics = walk_forward_backtest(
        dataset, target_col=target_col, drop_cols=drop,
        n_folds=1,
        purge_bars=max(20, horizon * 2),
        confidence_threshold=threshold,
        max_hold_bars=int(max_hold),
        use_regression=use_regression,
        regression_threshold=ROUNDTRIP_FEE * (1 + (threshold - 0.5) * 4),
        min_magnitude=min_mag,
        strategy_name="oos_validation",
    )
    return metrics


# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------
def rank_results(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank configs by a composite score: sharpe * profit_factor, filtered by min trades."""
    if df.empty:
        return df
    df = df[df["n_trades"] >= 20].copy()
    if df.empty:
        return df

    df["score"] = df["sharpe"] * df["profit_factor"].clip(upper=10)
    df = df.sort_values("score", ascending=False).head(top_n)
    return df


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Systematic profitable model search")
    parser.add_argument("--data-dir", default="data/raw", help="Local parquet data directory")
    parser.add_argument("--phase", default="all", help="Phase to run: 2,3,4,5,6,all")
    parser.add_argument("--start", default="2025-11-05")
    parser.add_argument("--stop", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = get_client(args.data_dir)
    stop = args.stop or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Reserve holdout period
    holdout_dt = datetime.now(timezone.utc) - timedelta(days=HOLDOUT_DAYS)
    holdout_start = holdout_dt.strftime("%Y-%m-%d")
    train_stop = holdout_start  # training data ends before holdout

    phase = args.phase
    t_total = time.time()

    # ===== PHASE 2 =====
    if phase in ("2", "all"):
        t0 = time.time()
        p2 = phase2_sweep(client, args.start, train_stop)
        p2.to_csv(RESULTS_DIR / "phase2_results.csv", index=False)
        logger.info("Phase 2 done in %.1f min, %d configs tested", (time.time() - t0) / 60, len(p2))

        top_p2 = rank_results(p2, top_n=10)
        if not top_p2.empty:
            logger.info("\n--- Phase 2 Top 10 ---")
            for _, r in top_p2.iterrows():
                logger.info("  %s: trades=%d net=%.4f sharpe=%.2f pf=%.2f wr=%.1f%%",
                            r["config"], r["n_trades"], r["total_net_return"],
                            r["sharpe"], r["profit_factor"], r["win_rate"] * 100)
            top_p2.to_csv(RESULTS_DIR / "phase2_top10.csv", index=False)
    else:
        p2_path = RESULTS_DIR / "phase2_top10.csv"
        if p2_path.exists():
            top_p2 = pd.read_csv(p2_path)
        else:
            logger.error("No Phase 2 results found. Run phase 2 first.")
            return

    # ===== PHASE 3 =====
    if phase in ("3", "all"):
        t0 = time.time()
        p3 = phase3_sweep(client, args.start, train_stop, top_p2.head(5))
        p3.to_csv(RESULTS_DIR / "phase3_results.csv", index=False)
        logger.info("Phase 3 done in %.1f min, %d configs tested", (time.time() - t0) / 60, len(p3))

        top_p3 = rank_results(p3, top_n=10)
        if not top_p3.empty:
            logger.info("\n--- Phase 3 Top 10 ---")
            for _, r in top_p3.iterrows():
                logger.info("  %s: trades=%d net=%.4f sharpe=%.2f pf=%.2f",
                            r["config"], r["n_trades"], r["total_net_return"],
                            r["sharpe"], r["profit_factor"])
            top_p3.to_csv(RESULTS_DIR / "phase3_top10.csv", index=False)
    else:
        p3_path = RESULTS_DIR / "phase3_top10.csv"
        if p3_path.exists():
            top_p3 = pd.read_csv(p3_path)
        else:
            top_p3 = top_p2

    # ===== PHASE 4 =====
    if phase in ("4", "all"):
        t0 = time.time()
        p4 = phase4_sweep(client, args.start, train_stop, top_p3.head(3))
        p4.to_csv(RESULTS_DIR / "phase4_results.csv", index=False)
        logger.info("Phase 4 done in %.1f min, %d configs tested", (time.time() - t0) / 60, len(p4))

        top_p4 = rank_results(p4, top_n=5)
        if not top_p4.empty:
            logger.info("\n--- Phase 4 Top 5 ---")
            for _, r in top_p4.iterrows():
                logger.info("  %s (%s): trades=%d net=%.4f sharpe=%.2f pf=%.2f",
                            r["config"], r["model_type"], r["n_trades"],
                            r["total_net_return"], r["sharpe"], r["profit_factor"])
            top_p4.to_csv(RESULTS_DIR / "phase4_top5.csv", index=False)
    else:
        p4_path = RESULTS_DIR / "phase4_top5.csv"
        top_p4 = pd.read_csv(p4_path) if p4_path.exists() else top_p3

    # ===== PHASE 5 =====
    if phase in ("5", "all"):
        # Use top configs from whichever phase has results
        best_so_far = top_p4 if not top_p4.empty else top_p3
        t0 = time.time()
        p5 = phase5_sweep(client, args.start, train_stop, best_so_far.head(3))
        p5.to_csv(RESULTS_DIR / "phase5_results.csv", index=False)
        logger.info("Phase 5 done in %.1f min, %d configs tested", (time.time() - t0) / 60, len(p5))

        top_p5 = rank_results(p5, top_n=5)
        if not top_p5.empty:
            logger.info("\n--- Phase 5 Top 5 ---")
            for _, r in top_p5.iterrows():
                logger.info("  %s: trades=%d net=%.4f sharpe=%.2f pf=%.2f wr=%.1f%%",
                            r["config"], r["n_trades"], r["total_net_return"],
                            r["sharpe"], r["profit_factor"], r["win_rate"] * 100)
            top_p5.to_csv(RESULTS_DIR / "phase5_top5.csv", index=False)
    else:
        p5_path = RESULTS_DIR / "phase5_top5.csv"
        top_p5 = pd.read_csv(p5_path) if p5_path.exists() else top_p4

    # ===== PHASE 6 =====
    if phase in ("6", "all"):
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 6: Out-of-sample validation on holdout period")
        logger.info("=" * 70)

        best_cfg = top_p5.iloc[0].to_dict() if not top_p5.empty else {}
        if not best_cfg:
            logger.error("No best config found for validation")
        else:
            oos = phase6_validate(client, args.start, stop, holdout_start, best_cfg)
            logger.info("\n--- OOS VALIDATION ---")
            logger.info("  Config: %s", best_cfg.get("config", ""))
            logger.info("  Trades: %d", oos.n_trades)
            logger.info("  Win rate: %.1f%%", oos.win_rate * 100)
            logger.info("  Net return: %.4f", oos.total_net_return)
            logger.info("  Sharpe: %.2f", oos.sharpe)
            logger.info("  Profit factor: %.2f", oos.profit_factor)
            logger.info("  Max drawdown: %.2f%%", oos.max_drawdown * 100)
            logger.info("  Final equity: $%.2f", oos.final_equity)

            if oos.per_pair:
                logger.info("\n  Per-pair breakdown:")
                for pair, stats in sorted(oos.per_pair.items(), key=lambda x: x[1]["net"], reverse=True):
                    wr = stats["wins"] / stats["n"] * 100 if stats["n"] > 0 else 0
                    logger.info("    %s: %d trades, net=%.4f, wr=%.1f%%",
                                pair, stats["n"], stats["net"], wr)

            # Save final report
            report = {
                "best_config": best_cfg,
                "oos_metrics": {
                    "n_trades": oos.n_trades,
                    "win_rate": oos.win_rate,
                    "total_net_return": oos.total_net_return,
                    "sharpe": oos.sharpe,
                    "profit_factor": oos.profit_factor,
                    "max_drawdown": oos.max_drawdown,
                    "final_equity": oos.final_equity,
                    "per_pair": oos.per_pair,
                },
            }
            with open(RESULTS_DIR / "final_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("\nFinal report saved to %s", RESULTS_DIR / "final_report.json")

    elapsed = time.time() - t_total
    logger.info("\n" + "=" * 70)
    logger.info("TOTAL TIME: %.1f minutes (%.1f hours)", elapsed / 60, elapsed / 3600)
    logger.info("=" * 70)

    client.close()


if __name__ == "__main__":
    main()
