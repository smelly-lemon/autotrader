#!/usr/bin/env python3
"""Model Search v2: Addresses all limiting choices from Round 1.

Stream A: Expanding-window OOS, bootstrap CIs, realistic spread/delay backtest
Stream B: Kyle lambda, trade clustering, Amihud, VPIN accel, volume-clock features
Stream C: Multi-model ensemble, per-pair specialization, consensus trading

Usage:
    python scripts/model_search_v2.py --data-dir data/raw --stream all
    python scripts/model_search_v2.py --data-dir data/raw --stream A
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.influx_client import LocalParquetClient, PRODUCT_IDS
from src.ml.features import (
    build_bar_features_from_ticker,
    build_bar_features_from_matches,
    compute_return_features,
    compute_cross_pair_features,
    compute_temporal_features,
    compute_vpin,
    BTC_PAIR,
    STABLECOIN_PAIR,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ROUNDTRIP_FEE = 0.004 * 2  # silver taker x2
RESULTS_DIR = Path("results/model_search_v2")
_DATA_CACHE: dict[str, pd.DataFrame] = {}

# Pairs that showed edge in Round 1 OOS
PROFITABLE_PAIRS = ["DOGE-USD", "XRP-USD", "LINK-USD", "SOL-USD", "UNI-USD"]
ALL_TRADE_PAIRS = [p for p in PRODUCT_IDS if p not in ("BTC-USDC", "ETH-USDC")]


# =====================================================================
# Stream B: Advanced microstructure features from match data
# =====================================================================
def compute_kyle_lambda(matches_df: pd.DataFrame, interval: str = "4h") -> pd.Series:
    """Kyle's lambda: regression of price change on signed order flow per bar.

    Higher lambda = more price impact per unit of flow = less liquidity.
    Changes in lambda predict regime shifts.
    """
    if matches_df.empty or len(matches_df) < 100:
        return pd.Series(dtype=float, name="kyle_lambda")

    df = matches_df[["price", "size", "side"]].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["signed_flow"] = df["size"].where(df["side"] == "BUY", -df["size"])

    bars = df.resample(interval).agg(
        price_change=("price", lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0),
        net_flow=("signed_flow", "sum"),
        n_trades=("price", "count"),
    )
    bars = bars[bars["n_trades"] > 5]

    # Rolling regression: price_change = lambda * net_flow + epsilon
    window = 30
    lambdas = []
    idx = []
    for i in range(window, len(bars)):
        chunk = bars.iloc[i - window:i]
        x = chunk["net_flow"].values
        y = chunk["price_change"].values
        if np.std(x) < 1e-12:
            lambdas.append(np.nan)
        else:
            slope = np.cov(x, y)[0, 1] / np.var(x)
            lambdas.append(abs(slope))
        idx.append(bars.index[i])

    return pd.Series(lambdas, index=idx, name="kyle_lambda")


def compute_amihud_illiquidity(matches_df: pd.DataFrame, interval: str = "4h") -> pd.Series:
    """Amihud illiquidity ratio: |return| / dollar volume.

    Spikes predict incoming volatility.
    """
    if matches_df.empty:
        return pd.Series(dtype=float, name="amihud")

    df = matches_df[["price", "size"]].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["notional"] = df["price"] * df["size"]

    bars = df.resample(interval).agg(
        close=("price", "last"),
        opn=("price", "first"),
        dollar_vol=("notional", "sum"),
        n=("price", "count"),
    )
    bars = bars[bars["n"] > 0]
    bars["ret"] = np.log(bars["close"] / bars["opn"].replace(0, np.nan)).abs()
    bars["amihud"] = bars["ret"] / bars["dollar_vol"].replace(0, np.nan)
    # Normalize to rolling z-score for cross-pair comparability
    roll_mean = bars["amihud"].rolling(30, min_periods=10).mean()
    roll_std = bars["amihud"].rolling(30, min_periods=10).std()
    bars["amihud_z"] = (bars["amihud"] - roll_mean) / roll_std.replace(0, np.nan)
    return bars["amihud_z"].rename("amihud_z")


def compute_trade_clustering(matches_df: pd.DataFrame, interval: str = "4h") -> pd.DataFrame:
    """Trade clustering features: Herfindahl of trade sizes, inter-arrival stats.

    Bursts of many small trades often precede large moves.
    """
    if matches_df.empty:
        return pd.DataFrame()

    df = matches_df[["price", "size"]].copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")

    # Inter-arrival times in seconds
    df["iat"] = df.index.to_series().diff().dt.total_seconds()

    def bar_stats(g):
        sizes = g["size"].values
        if len(sizes) < 2:
            return pd.Series({
                "herfindahl": 1.0, "iat_mean": np.nan, "iat_std": np.nan,
                "iat_skew": np.nan, "size_cv": np.nan,
            })
        total = sizes.sum()
        shares = sizes / total if total > 0 else sizes
        hhi = (shares ** 2).sum()
        iats = g["iat"].dropna().values
        return pd.Series({
            "herfindahl": hhi,
            "iat_mean": np.mean(iats) if len(iats) > 0 else np.nan,
            "iat_std": np.std(iats) if len(iats) > 1 else np.nan,
            "iat_skew": sp_stats.skew(iats) if len(iats) > 2 else np.nan,
            "size_cv": np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else np.nan,
        })

    result = df.resample(interval).apply(bar_stats)
    if isinstance(result, pd.DataFrame):
        return result
    # If apply returned a Series of Series, unstack
    return result.unstack()


def compute_vpin_acceleration(vpin_series: pd.Series, interval: str = "4h") -> pd.DataFrame:
    """Rate of change and momentum of VPIN — trend matters more than level."""
    if vpin_series.empty:
        return pd.DataFrame()

    v = vpin_series.resample(interval).last().dropna()
    features = pd.DataFrame(index=v.index)
    features["vpin"] = v
    features["vpin_diff_1"] = v.diff(1)
    features["vpin_diff_3"] = v.diff(3)
    features["vpin_ma_ratio"] = v / v.rolling(10, min_periods=3).mean().replace(0, np.nan)
    return features


def build_dataset_v2(
    client: LocalParquetClient,
    interval: str,
    start: str,
    stop: str,
    horizons: list[int],
    pairs: list[str] | None = None,
    use_advanced_features: bool = True,
) -> pd.DataFrame:
    """Build feature matrix with Stream B advanced microstructure features."""
    pairs = pairs or ALL_TRADE_PAIRS
    lags = [1, 5, 15, 60]

    btc_ticker = _load(client, BTC_PAIR, "ticker", start, stop)
    btc_matches = _load(client, BTC_PAIR, "matches", start, stop)
    btc_bars = build_bar_features_from_ticker(btc_ticker, interval)
    if btc_bars.empty:
        return pd.DataFrame()

    btc_price = btc_bars["price"]
    btc_ret = compute_return_features(btc_price, lags)

    # Stablecoin spread (was missing in v1)
    sc_price = None
    try:
        sc_ticker = _load(client, STABLECOIN_PAIR, "ticker", start, stop)
        if not sc_ticker.empty:
            sc_ticker["price"] = pd.to_numeric(sc_ticker["price"], errors="coerce")
            sc_price = sc_ticker.resample(interval)["price"].last().dropna()
    except Exception:
        pass

    all_frames = []

    for pair in pairs:
        if pair in ("BTC-USDC", "ETH-USDC"):
            continue

        ticker = _load(client, pair, "ticker", start, stop)
        matches = _load(client, pair, "matches", start, stop)

        ticker_bars = build_bar_features_from_ticker(ticker, interval)
        if ticker_bars.empty:
            continue

        match_bars = build_bar_features_from_matches(matches, interval)
        ret_feats = compute_return_features(ticker_bars["price"], lags)
        temporal = compute_temporal_features(ticker_bars.index)

        combined = ticker_bars.join(match_bars, how="left", rsuffix="_match")
        combined = combined.join(ret_feats)
        combined = combined.join(temporal)

        # VPIN + acceleration
        if not matches.empty:
            vpin = compute_vpin(matches)
            if not vpin.empty:
                vpin_feats = compute_vpin_acceleration(vpin, interval)
                combined = combined.join(vpin_feats)

        # TA features
        combined = _add_ta_features(combined)

        # Stream B: Advanced microstructure
        if use_advanced_features and not matches.empty:
            kyle = compute_kyle_lambda(matches, interval)
            if not kyle.empty:
                combined = combined.join(kyle)

            amihud = compute_amihud_illiquidity(matches, interval)
            if not amihud.empty:
                combined = combined.join(amihud)

            clustering = compute_trade_clustering(matches, interval)
            if not clustering.empty:
                combined = combined.join(clustering)

        # Cross-pair features (with stablecoin spread restored)
        if pair != BTC_PAIR:
            cross = compute_cross_pair_features(btc_price, ticker_bars["price"], sc_price)
            btc_ctx = {}
            for col in ["spread_bps_mean", "volatility_15", "volatility_60", "tick_count"]:
                src = btc_bars if col in btc_bars.columns else btc_ret
                if col in src.columns:
                    btc_ctx[f"btc_{col}"] = src[col]
            if btc_ctx:
                combined = combined.join(cross).join(pd.DataFrame(btc_ctx, index=btc_bars.index))

        # Targets
        log_p = np.log(combined["price"].replace(0, np.nan))
        for h in horizons:
            fwd = log_p.shift(-h) - log_p
            combined[f"fwd_ret_{h}"] = fwd
            combined[f"direction_{h}"] = (fwd > 0).astype(int)

        combined["product_id"] = pair
        # Store spread for realistic backtest (Stream A)
        if "spread_bps_mean" in combined.columns:
            combined["_spread_bps"] = combined["spread_bps_mean"]
        else:
            combined["_spread_bps"] = 0.0

        all_frames.append(combined)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames).sort_index()


def _load(client, pair, meas, start, stop):
    key = f"{meas}_{pair}_{start}_{stop}"
    if key not in _DATA_CACHE:
        if meas == "ticker":
            _DATA_CACHE[key] = client.get_ticker_data(pair, start=start, stop=stop)
        else:
            _DATA_CACHE[key] = client.get_matches(pair, start=start, stop=stop)
    return _DATA_CACHE[key]


def _add_ta_features(df):
    if "price" not in df.columns:
        return df
    p = df["price"]
    delta = p.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + rs)
    sma20 = p.rolling(20, min_periods=10).mean()
    std20 = p.rolling(20, min_periods=10).std()
    df["bb_pctb"] = (p - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    if "price_high" in df.columns and "price_low" in df.columns:
        tr = df["price_high"] - df["price_low"]
        df["atr_14"] = tr.rolling(14, min_periods=5).mean() / p.replace(0, np.nan)
    if "trade_volume" in df.columns:
        vol_ma = df["trade_volume"].rolling(20, min_periods=5).mean()
        df["vol_ratio"] = df["trade_volume"] / vol_ma.replace(0, np.nan)
    return df


# =====================================================================
# Stream A: Realistic walk-forward backtest with spread, delay, portfolio
# =====================================================================
@dataclass
class TradeRecord:
    entry_time: object
    exit_time: object
    product_id: str
    direction: int
    entry_price: float
    exit_price: float
    gross_ret: float
    spread_cost: float
    fee_cost: float
    net_ret: float
    bars_held: int


@dataclass
class BacktestResult:
    name: str = ""
    n_trades: int = 0
    win_rate: float = 0.0
    total_net_pnl: float = 0.0
    avg_net_ret: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_hold: float = 0.0
    per_pair: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    # Bootstrap CI (Stream A)
    sharpe_ci_lo: float = 0.0
    sharpe_ci_hi: float = 0.0
    pnl_ci_lo: float = 0.0
    pnl_ci_hi: float = 0.0


def make_drop_cols(horizons):
    base = ["product_id", "price", "price_open", "price_high", "price_low",
            "best_bid", "best_ask", "volume_24h", "vwap", "btc_moved",
            "_spread_bps", "timestamp"]
    for h in list(set(horizons + [1, 6, 15, 60, 240])):
        base += [f"fwd_ret_{h}", f"fwd_abs_ret_{h}", f"direction_{h}"]
    return list(set(base))


def realistic_backtest(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
    n_folds: int = 5,
    purge_bars: int = 30,
    lgbm_params: dict | None = None,
    confidence_threshold: float = 0.58,
    max_hold_bars: int = 60,
    execution_delay: int = 1,
    max_concurrent_positions: int = 5,
    use_regression: bool = False,
    regression_threshold: float = 0.012,
    min_magnitude: float = 0.002,
    strategy_name: str = "",
    pair_filter: list[str] | None = None,
) -> BacktestResult:
    """Stream A: Realistic backtest with spread, execution delay, portfolio limits."""
    params = lgbm_params or {
        "objective": "regression" if use_regression else "binary",
        "metric": "rmse" if use_regression else "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127, "learning_rate": 0.05,
        "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 30,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
    }

    work = df.copy().reset_index()
    if work.columns[0] not in df.columns:
        work.rename(columns={work.columns[0]: "timestamp"}, inplace=True)
    if "timestamp" not in work.columns:
        work["timestamp"] = work.index

    if pair_filter:
        work = work[work["product_id"].isin(pair_filter)]

    # Target
    if use_regression:
        ret_col = target_col.replace("direction_", "fwd_ret_")
        if ret_col not in work.columns:
            return BacktestResult(name=strategy_name)
        work = work.dropna(subset=[ret_col])
        y_all = work[ret_col].astype(float)
    else:
        if target_col not in work.columns:
            return BacktestResult(name=strategy_name)
        work = work.dropna(subset=[target_col])
        if min_magnitude > 0:
            ret_col = target_col.replace("direction_", "fwd_ret_")
            if ret_col in work.columns:
                work = work[work[ret_col].abs() >= min_magnitude]
        y_all = work[target_col].astype(int)

    all_drop = list(set(drop_cols) & set(work.columns))
    X_all = work.drop(columns=all_drop, errors="ignore")
    X_all = X_all.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    prices = work["price"] if "price" in work.columns else pd.Series(dtype=float)
    pids = work["product_id"] if "product_id" in work.columns else None
    ts_col = work["timestamp"]
    spread_bps = work["_spread_bps"] if "_spread_bps" in work.columns else pd.Series(0.0, index=work.index)

    timestamps = np.sort(ts_col.unique())
    n_ts = len(timestamps)
    if n_ts < 200:
        return BacktestResult(name=strategy_name)

    test_size = n_ts // (n_folds + 1)
    all_trades: list[TradeRecord] = []
    models = []

    for fold_i in range(n_folds):
        test_start_idx = n_ts - (n_folds - fold_i) * test_size
        test_end_idx = test_start_idx + test_size
        train_end_idx = test_start_idx - purge_bars

        if train_end_idx <= 50 or test_start_idx >= n_ts:
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
        X_train, X_test = X_train[common], X_test[common]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        try:
            model = lgb.train(
                params, dtrain, num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
        except Exception:
            continue

        models.append(model)
        preds = model.predict(X_test)
        test_idx = X_test.index
        t_prices = prices.loc[test_idx].values
        t_pids = pids.loc[test_idx].values if pids is not None else ["unknown"] * len(test_idx)
        t_ts = ts_col.loc[test_idx].values
        t_spread = spread_bps.loc[test_idx].values

        if use_regression:
            signals = np.zeros(len(preds))
            signals[preds > regression_threshold] = 1
            signals[preds < -regression_threshold] = -1
        else:
            signals = np.zeros(len(preds))
            signals[preds > confidence_threshold] = 1
            signals[preds < (1 - confidence_threshold)] = -1

        # Per-pair positions with execution delay and portfolio limits
        positions = {}
        pending_entries = {}  # delay buffer

        for j in range(len(preds)):
            price = t_prices[j]
            if np.isnan(price) or price <= 0:
                continue
            pid = t_pids[j]
            ts = t_ts[j]
            sig = int(signals[j])
            spr = t_spread[j] if not np.isnan(t_spread[j]) else 0.0

            # Execute pending entries (from previous bar — execution delay)
            if pid in pending_entries:
                pe = pending_entries.pop(pid)
                if len(positions) < max_concurrent_positions:
                    positions[pid] = {
                        "entry_price": price,  # fill at THIS bar's price (delayed)
                        "entry_time": ts,
                        "entry_spread": spr,
                        "direction": pe["direction"],
                        "bars_held": 0,
                    }

            # Check exits
            if pid in positions:
                pos = positions[pid]
                pos["bars_held"] += 1
                should_exit = pos["bars_held"] >= max_hold_bars
                if sig != 0 and sig != pos["direction"]:
                    should_exit = True
                if should_exit:
                    d = pos["direction"]
                    gross_ret = (price - pos["entry_price"]) / pos["entry_price"] * d
                    # Realistic costs: half-spread on entry + half-spread on exit + fees
                    spread_cost = (pos["entry_spread"] + spr) / 2 / 10000
                    net_ret = gross_ret - ROUNDTRIP_FEE - spread_cost

                    all_trades.append(TradeRecord(
                        entry_time=pos["entry_time"], exit_time=ts,
                        product_id=pid, direction=d,
                        entry_price=pos["entry_price"], exit_price=price,
                        gross_ret=gross_ret, spread_cost=spread_cost,
                        fee_cost=ROUNDTRIP_FEE, net_ret=net_ret,
                        bars_held=pos["bars_held"],
                    ))
                    del positions[pid]

            # Queue entries with delay
            if pid not in positions and pid not in pending_entries and sig != 0:
                if execution_delay > 0:
                    pending_entries[pid] = {"direction": sig}
                elif len(positions) < max_concurrent_positions:
                    positions[pid] = {
                        "entry_price": price, "entry_time": ts,
                        "entry_spread": spr,
                        "direction": sig, "bars_held": 0,
                    }

    if not all_trades:
        return BacktestResult(name=strategy_name)

    return _compute_metrics(all_trades, strategy_name, models[-1] if models else None)


def _compute_metrics(trades: list[TradeRecord], name: str, model=None) -> BacktestResult:
    net_rets = [t.net_ret for t in trades]
    wins = [r for r in net_rets if r > 0]
    losses = [r for r in net_rets if r <= 0]

    per_pair = {}
    for t in trades:
        if t.product_id not in per_pair:
            per_pair[t.product_id] = {"n": 0, "net": 0.0, "wins": 0}
        per_pair[t.product_id]["n"] += 1
        per_pair[t.product_id]["net"] += t.net_ret
        if t.net_ret > 0:
            per_pair[t.product_id]["wins"] += 1

    # Sharpe
    if len(net_rets) > 1:
        std = np.std(net_rets)
        mean = np.mean(net_rets)
        first_t = pd.Timestamp(trades[0].entry_time)
        last_t = pd.Timestamp(trades[-1].exit_time)
        span_days = max(1, (last_t - first_t).days)
        tpy = len(trades) / span_days * 365
        sharpe = (mean / std * np.sqrt(tpy)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    total_w = sum(wins) if wins else 0
    total_l = abs(sum(losses)) if losses else 0
    pf = total_w / total_l if total_l > 0 else (99 if total_w > 0 else 0)

    cum_pnl = np.cumsum([t.net_ret * 1000.0 for t in trades])
    eq = 10000.0 + cum_pnl
    peak = pd.Series(eq).expanding().max()
    dd = (pd.Series(eq) - peak) / peak
    max_dd = dd.min()

    # Bootstrap CIs (Stream A)
    sharpe_ci_lo, sharpe_ci_hi = 0.0, 0.0
    pnl_ci_lo, pnl_ci_hi = 0.0, 0.0
    if len(net_rets) >= 20:
        rng = np.random.default_rng(42)
        boot_sharpes = []
        boot_pnls = []
        arr = np.array(net_rets)
        for _ in range(2000):
            sample = rng.choice(arr, size=len(arr), replace=True)
            s_mean = sample.mean()
            s_std = sample.std()
            if s_std > 0:
                boot_sharpes.append(s_mean / s_std * np.sqrt(tpy if len(net_rets) > 1 else 1))
            boot_pnls.append(sample.sum())

        sharpe_ci_lo, sharpe_ci_hi = np.percentile(boot_sharpes, [2.5, 97.5]) if boot_sharpes else (0, 0)
        pnl_ci_lo, pnl_ci_hi = np.percentile(boot_pnls, [2.5, 97.5])

    return BacktestResult(
        name=name, n_trades=len(trades),
        win_rate=len(wins) / len(trades),
        total_net_pnl=sum(net_rets),
        avg_net_ret=np.mean(net_rets),
        sharpe=sharpe, profit_factor=min(pf, 99),
        max_drawdown=max_dd,
        avg_hold=np.mean([t.bars_held for t in trades]),
        per_pair=per_pair, trades=trades,
        sharpe_ci_lo=sharpe_ci_lo, sharpe_ci_hi=sharpe_ci_hi,
        pnl_ci_lo=pnl_ci_lo, pnl_ci_hi=pnl_ci_hi,
    )


# =====================================================================
# Stream A: Expanding-window OOS validation
# =====================================================================
def expanding_window_oos(
    client: LocalParquetClient,
    data_start: str = "2025-11-05",
    data_stop: str | None = None,
) -> list[dict]:
    """Train on expanding windows, test on next month. 3+ independent OOS windows."""
    logger.info("=" * 70)
    logger.info("STREAM A: Expanding-Window Out-of-Sample Validation")
    logger.info("=" * 70)

    data_stop = data_stop or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_dt = datetime.strptime(data_start, "%Y-%m-%d")
    stop_dt = datetime.strptime(data_stop, "%Y-%m-%d")
    total_days = (stop_dt - start_dt).days

    # Split into monthly windows
    window_size_days = 30
    min_train_days = 90  # need at least 3 months to train
    results = []

    test_start = start_dt + timedelta(days=min_train_days)
    window_i = 0

    while test_start + timedelta(days=window_size_days) <= stop_dt:
        test_end = test_start + timedelta(days=window_size_days)
        train_start_str = data_start
        train_stop_str = test_start.strftime("%Y-%m-%d")
        test_start_str = test_start.strftime("%Y-%m-%d")
        test_stop_str = test_end.strftime("%Y-%m-%d")
        window_i += 1

        logger.info("  Window %d: train %s->%s, test %s->%s",
                     window_i, train_start_str, train_stop_str, test_start_str, test_stop_str)

        # Build datasets
        train_ds = build_dataset_v2(client, "4h", train_start_str, train_stop_str, horizons=[30])
        test_ds = build_dataset_v2(client, "4h", test_start_str, test_stop_str, horizons=[30])

        if train_ds.empty or test_ds.empty:
            logger.warning("    Empty dataset, skipping")
            test_start = test_end
            continue

        # Train on full training set
        drop = make_drop_cols([30])
        target_col = "direction_30"

        # Prepare training data
        train_work = train_ds.copy().reset_index()
        if train_work.columns[0] not in train_ds.columns:
            train_work.rename(columns={train_work.columns[0]: "timestamp"}, inplace=True)

        train_work = train_work.dropna(subset=[target_col])
        ret_col = "fwd_ret_30"
        if ret_col in train_work.columns:
            train_work = train_work[train_work[ret_col].abs() >= 0.002]

        y_train = train_work[target_col].astype(int)
        all_drop = list(set(drop) & set(train_work.columns))
        X_train = train_work.drop(columns=all_drop, errors="ignore")
        X_train = X_train.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

        # Prepare test data
        test_work = test_ds.copy().reset_index()
        if test_work.columns[0] not in test_ds.columns:
            test_work.rename(columns={test_work.columns[0]: "timestamp"}, inplace=True)

        test_work = test_work.dropna(subset=[target_col])
        y_test = test_work[target_col].astype(int)
        X_test = test_work.drop(columns=list(set(drop) & set(test_work.columns)), errors="ignore")
        X_test = X_test.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

        common = X_train.columns.intersection(X_test.columns)
        X_train, X_test = X_train[common], X_test[common]

        if len(X_train) < 50 or len(X_test) < 10:
            test_start = test_end
            continue

        # Train
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "num_leaves": 127, "learning_rate": 0.05,
            "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
            "min_child_samples": 30, "lambda_l1": 0.1, "lambda_l2": 1.0,
            "verbose": -1, "n_jobs": -1, "seed": 42,
        }
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        try:
            model = lgb.train(params, dtrain, num_boost_round=500,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        except Exception as e:
            logger.error("    Training failed: %s", str(e)[:80])
            test_start = test_end
            continue

        preds = model.predict(X_test)

        # Simulate trades with realistic costs
        test_prices = test_work["price"].values if "price" in test_work.columns else np.full(len(preds), np.nan)
        test_pids = test_work["product_id"].values if "product_id" in test_work.columns else ["unknown"] * len(preds)
        test_ts_vals = test_work["timestamp"].values if "timestamp" in test_work.columns else range(len(preds))
        test_spread = test_work["_spread_bps"].values if "_spread_bps" in test_work.columns else np.zeros(len(preds))

        signals = np.zeros(len(preds))
        signals[preds > 0.58] = 1
        signals[preds < 0.42] = -1

        positions = {}
        trades = []

        for j in range(len(preds)):
            price = test_prices[j]
            if np.isnan(price) or price <= 0:
                continue
            pid = test_pids[j]
            sig = int(signals[j])
            ts = test_ts_vals[j]
            spr = test_spread[j] if not np.isnan(test_spread[j]) else 0.0

            if pid in positions:
                pos = positions[pid]
                pos["bars_held"] += 1
                should_exit = pos["bars_held"] >= 60
                if sig != 0 and sig != pos["direction"]:
                    should_exit = True
                if should_exit:
                    d = pos["direction"]
                    gross_ret = (price - pos["entry_price"]) / pos["entry_price"] * d
                    spread_cost = (pos["entry_spread"] + spr) / 2 / 10000
                    net_ret = gross_ret - ROUNDTRIP_FEE - spread_cost
                    trades.append(TradeRecord(
                        entry_time=pos["entry_time"], exit_time=ts,
                        product_id=pid, direction=d,
                        entry_price=pos["entry_price"], exit_price=price,
                        gross_ret=gross_ret, spread_cost=spread_cost,
                        fee_cost=ROUNDTRIP_FEE, net_ret=net_ret,
                        bars_held=pos["bars_held"],
                    ))
                    del positions[pid]

            if pid not in positions and sig != 0 and len(positions) < 5:
                positions[pid] = {
                    "entry_price": price, "entry_time": ts,
                    "entry_spread": spr, "direction": sig, "bars_held": 0,
                }

        n = len(trades)
        nets = [t.net_ret for t in trades]
        w = sum(1 for r in nets if r > 0)

        result = {
            "window": window_i,
            "train_period": f"{train_start_str} -> {train_stop_str}",
            "test_period": f"{test_start_str} -> {test_stop_str}",
            "n_trades": n,
            "win_rate": w / n if n > 0 else 0,
            "total_net": sum(nets) if nets else 0,
            "avg_net": np.mean(nets) if nets else 0,
        }

        if n > 1:
            std = np.std(nets)
            if std > 0:
                result["sharpe_est"] = np.mean(nets) / std * np.sqrt(n / 30 * 365)
            else:
                result["sharpe_est"] = 0
        else:
            result["sharpe_est"] = 0

        # Per-pair breakdown
        pp = {}
        for t in trades:
            if t.product_id not in pp:
                pp[t.product_id] = {"n": 0, "net": 0}
            pp[t.product_id]["n"] += 1
            pp[t.product_id]["net"] += t.net_ret
        result["per_pair"] = pp

        results.append(result)
        logger.info("    -> %d trades, net=%.4f, wr=%.1f%%, sharpe~%.2f",
                     n, result["total_net"], result["win_rate"] * 100, result["sharpe_est"])

        test_start = test_end

    return results


# =====================================================================
# Stream C: Multi-model ensemble
# =====================================================================
def ensemble_backtest(
    client: LocalParquetClient,
    start: str, stop: str,
) -> BacktestResult:
    """Train multiple models on different horizons, trade on consensus."""
    logger.info("=" * 70)
    logger.info("STREAM C: Ensemble (multi-horizon consensus)")
    logger.info("=" * 70)

    horizons = [10, 15, 30]
    ds = build_dataset_v2(client, "4h", start, stop, horizons=horizons)
    if ds.empty:
        return BacktestResult(name="ensemble")

    work = ds.copy().reset_index()
    if work.columns[0] not in ds.columns:
        work.rename(columns={work.columns[0]: "timestamp"}, inplace=True)
    if "timestamp" not in work.columns:
        work["timestamp"] = work.index

    all_drop_base = ["product_id", "price", "price_open", "price_high", "price_low",
                     "best_bid", "best_ask", "volume_24h", "vwap", "btc_moved",
                     "_spread_bps", "timestamp"]
    for h in horizons + [1, 6, 60, 240]:
        all_drop_base += [f"fwd_ret_{h}", f"fwd_abs_ret_{h}", f"direction_{h}"]
    all_drop_base = list(set(all_drop_base))

    ts_col = work["timestamp"]
    timestamps = np.sort(ts_col.unique())
    n_ts = len(timestamps)
    if n_ts < 200:
        return BacktestResult(name="ensemble")

    n_folds = 5
    test_size = n_ts // (n_folds + 1)
    all_trades = []

    for fold_i in range(n_folds):
        test_start_idx = n_ts - (n_folds - fold_i) * test_size
        test_end_idx = test_start_idx + test_size
        train_end_idx = test_start_idx - 30
        if train_end_idx <= 50:
            continue

        train_end_ts = timestamps[min(train_end_idx, n_ts - 1)]
        test_start_ts = timestamps[min(test_start_idx, n_ts - 1)]
        test_end_ts = timestamps[min(test_end_idx - 1, n_ts - 1)]

        train_mask = (ts_col <= train_end_ts).values
        test_mask = ((ts_col >= test_start_ts) & (ts_col <= test_end_ts)).values

        # Train one model per horizon
        horizon_preds = {}
        for h in horizons:
            tc = f"direction_{h}"
            if tc not in work.columns:
                continue
            w = work.dropna(subset=[tc])
            ret_col = f"fwd_ret_{h}"
            if ret_col in w.columns:
                w = w[w[ret_col].abs() >= 0.002]

            y_all = w[tc].astype(int)
            X_all = w.drop(columns=[c for c in all_drop_base if c in w.columns], errors="ignore")
            X_all = X_all.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

            ts_w = w["timestamp"] if "timestamp" in w.columns else pd.Series(range(len(w)))
            train_m = (ts_w <= train_end_ts).values
            test_m = ((ts_w >= test_start_ts) & (ts_w <= test_end_ts)).values

            X_tr, y_tr = X_all.loc[train_m], y_all.loc[train_m]
            X_te, y_te = X_all.loc[test_m], y_all.loc[test_m]

            if len(X_tr) < 50 or len(X_te) < 10:
                continue

            common = X_tr.columns.intersection(X_te.columns)
            X_tr, X_te = X_tr[common], X_te[common]

            params = {
                "objective": "binary", "metric": "binary_logloss",
                "boosting_type": "gbdt", "num_leaves": 127, "learning_rate": 0.05,
                "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
                "min_child_samples": 30, "verbose": -1, "n_jobs": -1, "seed": 42,
            }
            try:
                dtrain = lgb.Dataset(X_tr, label=y_tr)
                dval = lgb.Dataset(X_te, label=y_te, reference=dtrain)
                model = lgb.train(params, dtrain, num_boost_round=500,
                                  valid_sets=[dval],
                                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
                preds = model.predict(X_te)
                horizon_preds[h] = pd.Series(preds, index=X_te.index)
            except Exception:
                continue

        if len(horizon_preds) < 2:
            continue

        # Consensus: average probabilities across horizons
        common_idx = horizon_preds[horizons[0]].index
        for h in horizons[1:]:
            if h in horizon_preds:
                common_idx = common_idx.intersection(horizon_preds[h].index)

        if len(common_idx) < 10:
            continue

        avg_pred = np.zeros(len(common_idx))
        count = 0
        for h in horizons:
            if h in horizon_preds:
                avg_pred += horizon_preds[h].reindex(common_idx).fillna(0.5).values
                count += 1
        avg_pred /= count

        # Only trade when consensus is strong
        signals = np.zeros(len(avg_pred))
        signals[avg_pred > 0.58] = 1
        signals[avg_pred < 0.42] = -1

        # Also require agreement: at least 2/3 models must agree on direction
        for i, idx in enumerate(common_idx):
            votes = 0
            for h in horizons:
                if h in horizon_preds:
                    p = horizon_preds[h].get(idx, 0.5)
                    if signals[i] == 1 and p > 0.55:
                        votes += 1
                    elif signals[i] == -1 and p < 0.45:
                        votes += 1
            if votes < 2:
                signals[i] = 0

        # Simulate trades
        test_w = work.loc[common_idx]
        t_prices = test_w["price"].values if "price" in test_w.columns else np.full(len(common_idx), np.nan)
        t_pids = test_w["product_id"].values if "product_id" in test_w.columns else ["unknown"] * len(common_idx)
        t_ts = test_w["timestamp"].values if "timestamp" in test_w.columns else range(len(common_idx))
        t_spr = test_w["_spread_bps"].values if "_spread_bps" in test_w.columns else np.zeros(len(common_idx))

        positions = {}
        for j in range(len(common_idx)):
            price = t_prices[j]
            if np.isnan(price) or price <= 0:
                continue
            pid = t_pids[j]
            sig = int(signals[j])
            ts = t_ts[j]
            spr = t_spr[j] if not np.isnan(t_spr[j]) else 0.0

            if pid in positions:
                pos = positions[pid]
                pos["bars_held"] += 1
                should_exit = pos["bars_held"] >= 60
                if sig != 0 and sig != pos["direction"]:
                    should_exit = True
                if should_exit:
                    d = pos["direction"]
                    gross_ret = (price - pos["entry_price"]) / pos["entry_price"] * d
                    spread_cost = (pos["entry_spread"] + spr) / 2 / 10000
                    net_ret = gross_ret - ROUNDTRIP_FEE - spread_cost
                    all_trades.append(TradeRecord(
                        entry_time=pos["entry_time"], exit_time=ts,
                        product_id=pid, direction=d,
                        entry_price=pos["entry_price"], exit_price=price,
                        gross_ret=gross_ret, spread_cost=spread_cost,
                        fee_cost=ROUNDTRIP_FEE, net_ret=net_ret,
                        bars_held=pos["bars_held"],
                    ))
                    del positions[pid]

            if pid not in positions and sig != 0 and len(positions) < 5:
                positions[pid] = {
                    "entry_price": price, "entry_time": ts,
                    "entry_spread": spr, "direction": sig, "bars_held": 0,
                }

    if not all_trades:
        return BacktestResult(name="ensemble")

    return _compute_metrics(all_trades, "ensemble")


# =====================================================================
# Stream C: Per-pair specialization
# =====================================================================
def per_pair_backtest(
    client: LocalParquetClient,
    start: str, stop: str,
) -> dict[str, BacktestResult]:
    """Train and evaluate pair-specific models."""
    logger.info("=" * 70)
    logger.info("STREAM C: Per-Pair Specialization")
    logger.info("=" * 70)

    results = {}
    for pair in ALL_TRADE_PAIRS:
        logger.info("  %s...", pair)
        ds = build_dataset_v2(client, "4h", start, stop, horizons=[30], pairs=[pair])
        if ds.empty or len(ds) < 200:
            logger.info("    skipped (too few rows: %d)", len(ds))
            continue

        r = realistic_backtest(
            ds, target_col="direction_30",
            drop_cols=make_drop_cols([30]),
            n_folds=5, confidence_threshold=0.58,
            max_hold_bars=60, min_magnitude=0.002,
            strategy_name=f"pair_{pair}",
        )
        results[pair] = r
        logger.info("    -> %d trades, net=%.4f, wr=%.1f%%, sharpe=%.2f [CI: %.2f..%.2f]",
                     r.n_trades, r.total_net_pnl, r.win_rate * 100, r.sharpe,
                     r.sharpe_ci_lo, r.sharpe_ci_hi)

    return results


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--stream", default="all", help="A, B, C, D, or all")
    parser.add_argument("--start", default="2025-11-05")
    parser.add_argument("--stop", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = LocalParquetClient(args.data_dir)
    stop = args.stop or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stream = args.stream.upper()
    t0 = time.time()

    # ── Stream A: Statistical Rigor ──────────────────────────────
    if stream in ("A", "ALL"):
        logger.info("\n" + "=" * 70)
        logger.info("STREAM A: Realistic Backtest with v2 Features")
        logger.info("=" * 70)

        # 1. Full realistic backtest with best config from Round 1 + new features
        holdout_start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        train_stop = holdout_start

        ds = build_dataset_v2(client, "4h", args.start, train_stop, horizons=[30])
        if not ds.empty:
            logger.info("Dataset: %d rows, %d cols", len(ds), len(ds.columns))

            # Test: all pairs
            r_all = realistic_backtest(
                ds, "direction_30", make_drop_cols([30]),
                confidence_threshold=0.58, max_hold_bars=60,
                min_magnitude=0.002, strategy_name="v2_all_pairs",
            )
            logger.info("\n--- v2 All Pairs (realistic) ---")
            logger.info("  Trades: %d, Net: %.4f, WR: %.1f%%, Sharpe: %.2f [CI: %.2f..%.2f]",
                        r_all.n_trades, r_all.total_net_pnl, r_all.win_rate * 100,
                        r_all.sharpe, r_all.sharpe_ci_lo, r_all.sharpe_ci_hi)
            logger.info("  PF: %.2f, MaxDD: %.2f%%, PnL CI: [%.4f..%.4f]",
                        r_all.profit_factor, r_all.max_drawdown * 100,
                        r_all.pnl_ci_lo, r_all.pnl_ci_hi)

            # Test: profitable pairs only
            r_alts = realistic_backtest(
                ds, "direction_30", make_drop_cols([30]),
                confidence_threshold=0.58, max_hold_bars=60,
                min_magnitude=0.002, strategy_name="v2_profitable_pairs",
                pair_filter=PROFITABLE_PAIRS,
            )
            logger.info("\n--- v2 Profitable Pairs Only ---")
            logger.info("  Trades: %d, Net: %.4f, WR: %.1f%%, Sharpe: %.2f [CI: %.2f..%.2f]",
                        r_alts.n_trades, r_alts.total_net_pnl, r_alts.win_rate * 100,
                        r_alts.sharpe, r_alts.sharpe_ci_lo, r_alts.sharpe_ci_hi)

            # Per-pair breakdown
            logger.info("\n  Per-pair:")
            for pair, stats in sorted(r_all.per_pair.items(), key=lambda x: x[1]["net"], reverse=True):
                wr = stats["wins"] / stats["n"] * 100 if stats["n"] > 0 else 0
                logger.info("    %s: %d trades, net=%.4f, wr=%.1f%%", pair, stats["n"], stats["net"], wr)

        # 2. Expanding-window OOS
        oos_results = expanding_window_oos(client, args.start, stop)
        if oos_results:
            with open(RESULTS_DIR / "expanding_oos.json", "w") as f:
                json.dump(oos_results, f, indent=2, default=str)

            logger.info("\n--- Expanding-Window OOS Summary ---")
            positive_windows = sum(1 for r in oos_results if r["total_net"] > 0)
            total_oos_trades = sum(r["n_trades"] for r in oos_results)
            total_oos_net = sum(r["total_net"] for r in oos_results)
            logger.info("  Windows: %d total, %d profitable", len(oos_results), positive_windows)
            logger.info("  Total OOS trades: %d, Total OOS net: %.4f", total_oos_trades, total_oos_net)
            for r in oos_results:
                logger.info("  Window %d (%s): %d trades, net=%.4f, wr=%.1f%%, sharpe~%.2f",
                            r["window"], r["test_period"], r["n_trades"],
                            r["total_net"], r["win_rate"] * 100, r["sharpe_est"])

    # ── Stream B: is integrated into build_dataset_v2 above ──────
    # (features are computed in the dataset build; results visible in Stream A output)

    # ── Stream C: Ensemble + Per-Pair ────────────────────────────
    if stream in ("C", "ALL"):
        holdout_start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        train_stop = holdout_start

        # Ensemble
        ens = ensemble_backtest(client, args.start, train_stop)
        logger.info("\n--- Ensemble Results ---")
        logger.info("  Trades: %d, Net: %.4f, WR: %.1f%%, Sharpe: %.2f [CI: %.2f..%.2f]",
                     ens.n_trades, ens.total_net_pnl, ens.win_rate * 100,
                     ens.sharpe, ens.sharpe_ci_lo, ens.sharpe_ci_hi)
        logger.info("  PF: %.2f, MaxDD: %.2f%%", ens.profit_factor, ens.max_drawdown * 100)

        # Per-pair specialization
        pp_results = per_pair_backtest(client, args.start, train_stop)
        logger.info("\n--- Per-Pair Specialization Summary ---")
        for pair, r in sorted(pp_results.items(), key=lambda x: x[1].total_net_pnl, reverse=True):
            logger.info("  %s: %d trades, net=%.4f, sharpe=%.2f [CI: %.2f..%.2f], wr=%.1f%%",
                         pair, r.n_trades, r.total_net_pnl, r.sharpe,
                         r.sharpe_ci_lo, r.sharpe_ci_hi, r.win_rate * 100)

        # Save all results
        report = {
            "ensemble": {
                "n_trades": ens.n_trades, "total_net_pnl": ens.total_net_pnl,
                "sharpe": ens.sharpe, "sharpe_ci": [ens.sharpe_ci_lo, ens.sharpe_ci_hi],
                "win_rate": ens.win_rate, "profit_factor": ens.profit_factor,
            },
            "per_pair": {
                pair: {
                    "n_trades": r.n_trades, "total_net_pnl": r.total_net_pnl,
                    "sharpe": r.sharpe, "sharpe_ci": [r.sharpe_ci_lo, r.sharpe_ci_hi],
                    "win_rate": r.win_rate,
                }
                for pair, r in pp_results.items()
            },
        }
        with open(RESULTS_DIR / "stream_c_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

    # ── Stream D: Sequence + Adaptive models ────────────────────
    if stream in ("D", "ALL"):
        comparison_run(client, args.start, stop)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("TOTAL TIME: %.1f min (%.1f hours)", elapsed / 60, elapsed / 3600)
    logger.info("=" * 70)
    client.close()


# =====================================================================
# Stream D: Sequence models + adaptive retraining
# =====================================================================

def _prepare_xy(ds, target_col="direction_30", min_magnitude=0.002):
    """Shared data prep: returns work DataFrame, X array, y array, metadata arrays."""
    drop = make_drop_cols([30])
    work = ds.copy().reset_index()
    if work.columns[0] not in ds.columns:
        work.rename(columns={work.columns[0]: "timestamp"}, inplace=True)
    if "timestamp" not in work.columns:
        work["timestamp"] = work.index

    work = work.dropna(subset=[target_col])
    ret_col = target_col.replace("direction_", "fwd_ret_")
    if ret_col in work.columns and min_magnitude > 0:
        work = work[work[ret_col].abs() >= min_magnitude]

    y = work[target_col].astype(int).values
    all_drop = list(set(drop) & set(work.columns))
    X_df = work.drop(columns=all_drop, errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    feature_cols = list(X_df.columns)
    X = X_df.fillna(0).values.astype(np.float32)

    prices = work["price"].values if "price" in work.columns else np.full(len(work), np.nan)
    pids = work["product_id"].values if "product_id" in work.columns else np.array(["unknown"] * len(work))
    ts = work["timestamp"].values
    spread = work["_spread_bps"].values if "_spread_bps" in work.columns else np.zeros(len(work))

    return X, y, prices, pids, ts, spread, feature_cols


def _simulate_trades(signals, prices, pids, ts_arr, spread_arr,
                     confidence_threshold=0.58, max_hold_bars=60,
                     max_concurrent=5, execution_delay=1):
    """Shared trade simulation from signal array — same logic as realistic_backtest."""
    from scripts.model_search_v2 import TradeRecord, ROUNDTRIP_FEE

    positions = {}
    pending = {}
    trades = []

    for j in range(len(signals)):
        price = prices[j]
        if np.isnan(price) or price <= 0:
            continue
        pid = pids[j]
        sig = int(signals[j])
        t = ts_arr[j]
        spr = spread_arr[j] if not np.isnan(spread_arr[j]) else 0.0

        if pid in pending:
            pe = pending.pop(pid)
            if len(positions) < max_concurrent:
                positions[pid] = {
                    "entry_price": price, "entry_time": t,
                    "entry_spread": spr, "direction": pe["direction"], "bars_held": 0,
                }

        if pid in positions:
            pos = positions[pid]
            pos["bars_held"] += 1
            should_exit = pos["bars_held"] >= max_hold_bars
            if sig != 0 and sig != pos["direction"]:
                should_exit = True
            if should_exit:
                d = pos["direction"]
                gross_ret = (price - pos["entry_price"]) / pos["entry_price"] * d
                spread_cost = (pos["entry_spread"] + spr) / 2 / 10000
                net_ret = gross_ret - ROUNDTRIP_FEE - spread_cost
                trades.append(TradeRecord(
                    entry_time=pos["entry_time"], exit_time=t,
                    product_id=pid, direction=d,
                    entry_price=pos["entry_price"], exit_price=price,
                    gross_ret=gross_ret, spread_cost=spread_cost,
                    fee_cost=ROUNDTRIP_FEE, net_ret=net_ret,
                    bars_held=pos["bars_held"],
                ))
                del positions[pid]

        if pid not in positions and pid not in pending and sig != 0:
            if execution_delay > 0:
                pending[pid] = {"direction": sig}
            elif len(positions) < max_concurrent:
                positions[pid] = {
                    "entry_price": price, "entry_time": t,
                    "entry_spread": spr, "direction": sig, "bars_held": 0,
                }

    return trades


def sequence_backtest_oos(
    client: LocalParquetClient,
    data_start: str,
    data_stop: str,
    model_type: str = "lstm",
    seq_len: int = 20,
) -> list[dict]:
    """Expanding-window OOS using LSTM/GRU sequence model."""
    from src.ml.sequence_model import (
        SequenceDataset, train_sequence_model, predict_sequence_model,
    )

    logger.info("  [%s] Expanding-window OOS (seq_len=%d)", model_type.upper(), seq_len)

    start_dt = datetime.strptime(data_start, "%Y-%m-%d")
    stop_dt = datetime.strptime(data_stop, "%Y-%m-%d")
    window_days = 30
    min_train_days = 90
    results = []

    test_start = start_dt + timedelta(days=min_train_days)
    window_i = 0

    while test_start + timedelta(days=window_days) <= stop_dt:
        test_end = test_start + timedelta(days=window_days)
        train_start_str = data_start
        train_stop_str = test_start.strftime("%Y-%m-%d")
        test_start_str = test_start.strftime("%Y-%m-%d")
        test_stop_str = test_end.strftime("%Y-%m-%d")
        window_i += 1

        logger.info("    Window %d: train->%s, test %s->%s",
                     window_i, train_stop_str, test_start_str, test_stop_str)

        train_ds = build_dataset_v2(client, "4h", train_start_str, train_stop_str, horizons=[30])
        test_ds = build_dataset_v2(client, "4h", test_start_str, test_stop_str, horizons=[30])

        if train_ds.empty or test_ds.empty:
            test_start = test_end
            continue

        X_train, y_train, _, pids_train, ts_train, _, feat_cols = _prepare_xy(train_ds)
        X_test, y_test, prices_test, pids_test, ts_test, spread_test, _ = _prepare_xy(test_ds)

        if len(X_train) < 100 or len(X_test) < 20:
            test_start = test_end
            continue

        # Normalize features (fit on train, transform both)
        means = np.nanmean(X_train, axis=0)
        stds = np.nanstd(X_train, axis=0)
        stds[stds < 1e-8] = 1.0
        X_train_n = (X_train - means) / stds
        X_test_n = (X_test - means) / stds
        X_train_n = np.nan_to_num(X_train_n, 0)
        X_test_n = np.nan_to_num(X_test_n, 0)

        train_seqds = SequenceDataset(X_train_n, y_train, pids_train, ts_train, seq_len=seq_len)
        test_seqds = SequenceDataset(X_test_n, y_test, pids_test, ts_test, seq_len=seq_len)

        if len(train_seqds) < 50 or len(test_seqds) < 5:
            test_start = test_end
            continue

        model = train_sequence_model(
            train_seqds, val_dataset=test_seqds,
            input_dim=X_train.shape[1], model_type=model_type,
            hidden_dim=32, epochs=30, patience=8, batch_size=128,
        )
        preds = predict_sequence_model(model, test_seqds)

        signals = np.zeros(len(preds))
        signals[preds > 0.58] = 1
        signals[preds < 0.42] = -1

        # The sequence dataset may have fewer rows than raw test data (windowing drops some)
        # Build a minimal trade simulation from the sequence predictions
        # Each sequence's prediction corresponds to the LAST bar in the window
        # We need to reconstruct which bars those are
        unique_pids = np.unique(pids_test)
        all_sig_prices = []
        all_sig_pids = []
        all_sig_ts = []
        all_sig_spread = []
        all_sig_signals = []

        pred_idx = 0
        for pid in unique_pids:
            mask = pids_test == pid
            n_pid = mask.sum()
            n_seq = max(0, n_pid - seq_len)
            if n_seq <= 0:
                continue
            pid_indices = np.where(mask)[0]
            order = np.argsort(ts_test[pid_indices])
            pid_indices = pid_indices[order]

            for i in range(n_seq):
                bar_idx = pid_indices[seq_len + i]
                if pred_idx < len(signals):
                    all_sig_prices.append(prices_test[bar_idx])
                    all_sig_pids.append(pids_test[bar_idx])
                    all_sig_ts.append(ts_test[bar_idx])
                    all_sig_spread.append(spread_test[bar_idx])
                    all_sig_signals.append(signals[pred_idx])
                    pred_idx += 1

        if not all_sig_prices:
            test_start = test_end
            continue

        trades = _simulate_trades(
            np.array(all_sig_signals), np.array(all_sig_prices),
            np.array(all_sig_pids), np.array(all_sig_ts),
            np.array(all_sig_spread),
        )

        n = len(trades)
        nets = [t.net_ret for t in trades]
        w = sum(1 for r in nets if r > 0)

        result = {
            "window": window_i,
            "test_period": f"{test_start_str} -> {test_stop_str}",
            "n_trades": n,
            "win_rate": w / n if n > 0 else 0,
            "total_net": sum(nets) if nets else 0,
            "avg_net": np.mean(nets) if nets else 0,
        }
        if n > 1 and np.std(nets) > 0:
            result["sharpe_est"] = np.mean(nets) / np.std(nets) * np.sqrt(n / 30 * 365)
        else:
            result["sharpe_est"] = 0

        results.append(result)
        logger.info("      -> %d trades, net=%.4f, wr=%.1f%%, sharpe~%.2f",
                     n, result["total_net"], result["win_rate"] * 100, result["sharpe_est"])

        test_start = test_end
        del model
        gc.collect()

    return results


def rolling_window_backtest_oos(
    client: LocalParquetClient,
    data_start: str,
    data_stop: str,
    train_window_days: int = 90,
    model_type: str = "lgbm",
    seq_len: int = 20,
) -> list[dict]:
    """Fixed-width rolling window: always train on most recent N days.

    Tests whether 'freshness' matters more than 'more data'.
    Works with both LightGBM and sequence models.
    """
    logger.info("  [%s rolling-%dd] OOS", model_type.upper(), train_window_days)

    start_dt = datetime.strptime(data_start, "%Y-%m-%d")
    stop_dt = datetime.strptime(data_stop, "%Y-%m-%d")
    window_days = 30
    results = []

    test_start = start_dt + timedelta(days=train_window_days)
    window_i = 0

    while test_start + timedelta(days=window_days) <= stop_dt:
        test_end = test_start + timedelta(days=window_days)
        # Rolling: train start slides forward
        train_start_dt = test_start - timedelta(days=train_window_days)
        train_start_str = train_start_dt.strftime("%Y-%m-%d")
        train_stop_str = test_start.strftime("%Y-%m-%d")
        test_start_str = test_start.strftime("%Y-%m-%d")
        test_stop_str = test_end.strftime("%Y-%m-%d")
        window_i += 1

        logger.info("    Window %d: train %s->%s, test %s->%s",
                     window_i, train_start_str, train_stop_str, test_start_str, test_stop_str)

        train_ds = build_dataset_v2(client, "4h", train_start_str, train_stop_str, horizons=[30])
        test_ds = build_dataset_v2(client, "4h", test_start_str, test_stop_str, horizons=[30])

        if train_ds.empty or test_ds.empty:
            test_start = test_end
            continue

        X_train, y_train, _, pids_train, ts_train, _, feat_cols = _prepare_xy(train_ds)
        X_test, y_test, prices_test, pids_test, ts_test, spread_test, _ = _prepare_xy(test_ds)

        if len(X_train) < 50 or len(X_test) < 10:
            test_start = test_end
            continue

        if model_type == "lgbm":
            # LightGBM on tabular data (rolling window)
            X_tr_df = pd.DataFrame(X_train, columns=feat_cols)
            X_te_df = pd.DataFrame(X_test, columns=feat_cols)
            common = X_tr_df.columns.intersection(X_te_df.columns)
            X_tr_df, X_te_df = X_tr_df[common], X_te_df[common]

            params = {
                "objective": "binary", "metric": "binary_logloss",
                "boosting_type": "gbdt", "num_leaves": 127, "learning_rate": 0.05,
                "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
                "min_child_samples": 30, "verbose": -1, "n_jobs": -1, "seed": 42,
            }
            dtrain = lgb.Dataset(X_tr_df, label=y_train)
            dval = lgb.Dataset(X_te_df, label=y_test, reference=dtrain)
            try:
                model = lgb.train(params, dtrain, num_boost_round=500,
                                  valid_sets=[dval],
                                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            except Exception:
                test_start = test_end
                continue
            preds = model.predict(X_te_df)
            signals = np.zeros(len(preds))
            signals[preds > 0.58] = 1
            signals[preds < 0.42] = -1

            trades = _simulate_trades(signals, prices_test, pids_test, ts_test, spread_test)

        else:
            # Sequence model (rolling window)
            from src.ml.sequence_model import (
                SequenceDataset, train_sequence_model, predict_sequence_model,
            )

            means = np.nanmean(X_train, axis=0)
            stds = np.nanstd(X_train, axis=0)
            stds[stds < 1e-8] = 1.0
            X_train_n = np.nan_to_num((X_train - means) / stds, 0)
            X_test_n = np.nan_to_num((X_test - means) / stds, 0)

            train_seqds = SequenceDataset(X_train_n, y_train, pids_train, ts_train, seq_len=seq_len)
            test_seqds = SequenceDataset(X_test_n, y_test, pids_test, ts_test, seq_len=seq_len)

            if len(train_seqds) < 30 or len(test_seqds) < 5:
                test_start = test_end
                continue

            model = train_sequence_model(
                train_seqds, val_dataset=test_seqds,
                input_dim=X_train.shape[1], model_type="lstm",
                hidden_dim=32, epochs=30, patience=8, batch_size=128,
            )
            preds = predict_sequence_model(model, test_seqds)
            signals = np.zeros(len(preds))
            signals[preds > 0.58] = 1
            signals[preds < 0.42] = -1

            # Reconstruct bar indices for trade sim
            unique_pids = np.unique(pids_test)
            sig_prices, sig_pids, sig_ts, sig_spread, sig_signals = [], [], [], [], []
            pred_idx = 0
            for pid in unique_pids:
                mask = pids_test == pid
                pid_indices = np.where(mask)[0]
                order = np.argsort(ts_test[pid_indices])
                pid_indices = pid_indices[order]
                n_seq = max(0, mask.sum() - seq_len)
                for i in range(n_seq):
                    bar_idx = pid_indices[seq_len + i]
                    if pred_idx < len(signals):
                        sig_prices.append(prices_test[bar_idx])
                        sig_pids.append(pids_test[bar_idx])
                        sig_ts.append(ts_test[bar_idx])
                        sig_spread.append(spread_test[bar_idx])
                        sig_signals.append(signals[pred_idx])
                        pred_idx += 1

            if not sig_prices:
                test_start = test_end
                continue

            trades = _simulate_trades(
                np.array(sig_signals), np.array(sig_prices),
                np.array(sig_pids), np.array(sig_ts), np.array(sig_spread),
            )
            del model
            gc.collect()

        n = len(trades)
        nets = [t.net_ret for t in trades]
        w = sum(1 for r in nets if r > 0)
        result = {
            "window": window_i,
            "test_period": f"{test_start_str} -> {test_stop_str}",
            "n_trades": n,
            "win_rate": w / n if n > 0 else 0,
            "total_net": sum(nets) if nets else 0,
            "sharpe_est": (np.mean(nets) / np.std(nets) * np.sqrt(n / 30 * 365)) if n > 1 and np.std(nets) > 0 else 0,
        }
        results.append(result)
        logger.info("      -> %d trades, net=%.4f, wr=%.1f%%, sharpe~%.2f",
                     n, result["total_net"], result["win_rate"] * 100, result["sharpe_est"])

        test_start = test_end

    return results


def comparison_run(
    client: LocalParquetClient,
    data_start: str,
    data_stop: str,
):
    """Run all 5 model approaches on the same expanding-window OOS for comparison."""
    logger.info("\n" + "=" * 70)
    logger.info("STREAM D: Model Comparison (5 approaches)")
    logger.info("=" * 70)

    all_results = {}

    # 1. LightGBM fixed (expanding window — already done in Stream A, but rerun for consistency)
    logger.info("\n[1/5] LightGBM — expanding window")
    lgbm_expanding = expanding_window_oos(client, data_start, data_stop)
    all_results["lgbm_expanding"] = lgbm_expanding

    # 2. LSTM fixed (expanding window)
    logger.info("\n[2/5] LSTM — expanding window")
    lstm_expanding = sequence_backtest_oos(client, data_start, data_stop, model_type="lstm", seq_len=20)
    all_results["lstm_expanding"] = lstm_expanding

    # 3. LSTM rolling 90-day window
    logger.info("\n[3/5] LSTM — rolling 90-day window")
    lstm_rolling = rolling_window_backtest_oos(client, data_start, data_stop,
                                               train_window_days=90, model_type="lstm", seq_len=20)
    all_results["lstm_rolling_90d"] = lstm_rolling

    # 4. LightGBM rolling 90-day window
    logger.info("\n[4/5] LightGBM — rolling 90-day window")
    lgbm_rolling = rolling_window_backtest_oos(client, data_start, data_stop,
                                               train_window_days=90, model_type="lgbm")
    all_results["lgbm_rolling_90d"] = lgbm_rolling

    # 5. Ensemble consensus (LSTM + LightGBM must agree)
    # We compare the per-window results to see if agreement correlates with profitability
    logger.info("\n[5/5] Consensus analysis (LSTM + LightGBM agreement)")

    consensus_results = []
    for i in range(min(len(lgbm_expanding), len(lstm_expanding))):
        lgbm_r = lgbm_expanding[i]
        lstm_r = lstm_expanding[i]
        both_positive = lgbm_r["total_net"] > 0 and lstm_r["total_net"] > 0
        both_negative = lgbm_r["total_net"] < 0 and lstm_r["total_net"] < 0
        consensus_results.append({
            "window": i + 1,
            "test_period": lgbm_r.get("test_period", ""),
            "lgbm_net": lgbm_r["total_net"],
            "lstm_net": lstm_r["total_net"],
            "agree": both_positive or both_negative,
            "both_positive": both_positive,
            "combined_net": lgbm_r["total_net"] + lstm_r["total_net"],
        })
    all_results["consensus"] = consensus_results

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)

    def _summarize(name, results_list):
        if not results_list:
            logger.info("  %-25s: no results", name)
            return
        total_net = sum(r.get("total_net", 0) for r in results_list)
        total_trades = sum(r.get("n_trades", 0) for r in results_list)
        pos_windows = sum(1 for r in results_list if r.get("total_net", 0) > 0)
        logger.info("  %-25s: %d windows (%d pos), %d trades, net=%.4f",
                     name, len(results_list), pos_windows, total_trades, total_net)

    _summarize("LightGBM expanding", lgbm_expanding)
    _summarize("LSTM expanding", lstm_expanding)
    _summarize("LSTM rolling-90d", lstm_rolling)
    _summarize("LightGBM rolling-90d", lgbm_rolling)

    if consensus_results:
        agree_windows = [c for c in consensus_results if c["agree"]]
        logger.info("  Consensus: %d/%d windows agree, combined net when agree=%.4f",
                     len(agree_windows), len(consensus_results),
                     sum(c["combined_net"] for c in agree_windows))

    # Save
    with open(RESULTS_DIR / "model_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\nResults saved to %s", RESULTS_DIR / "model_comparison.json")


if __name__ == "__main__":
    main()
