"""ML-driven live trading runner.

Connects trained LightGBM models to either PaperExecutor or CoinbaseExecutor.
Fetches latest bars from Coinbase API, runs inference, manages positions with
portfolio-level risk limits.

Usage:
    # Paper trading (default)
    python -m src.ml.live_trader --mode paper --capital 10000

    # Live trading
    python -m src.ml.live_trader --mode live
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt.async_support as ccxt_async
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ml.features import (
    build_bar_features_from_ticker,
    compute_return_features,
    compute_cross_pair_features,
    compute_temporal_features,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
LOG_PATH = Path("data/ml_trading_log.jsonl")
PROFITABLE_PAIRS = ["DOGE/USD", "XRP/USD", "LINK/USD", "SOL/USD", "UNI/USD"]
BAR_INTERVAL_HOURS = 4
PREDICTION_HORIZON = 30  # bars
CONFIDENCE_THRESHOLD = 0.58
MAX_HOLD_BARS = 60
MAX_CONCURRENT_POSITIONS = 3
POSITION_SIZE_PCT = 0.10
STOP_LOSS_PCT = 0.05
DAILY_DRAWDOWN_LIMIT = 0.05


@dataclass
class Position:
    symbol: str
    entry_price: float
    entry_time: datetime
    direction: int  # 1=long
    bars_held: int = 0
    amount: float = 0.0
    trade_id: str = ""


RETRAIN_WINDOW_DAYS = 90
RETRAIN_MIN_TRADES = 20
RETRAIN_SHARPE_THRESHOLD = 0.0


class MLLiveTrader:
    """Runs the winning ML strategy live or in paper mode.

    Supports adaptive retraining: monitors rolling Sharpe over recent trades
    and triggers a retrain when performance decays below threshold.
    """

    def __init__(
        self,
        executor,
        model_path: str | Path | None = None,
        pairs: list[str] | None = None,
        initial_capital: float = 10_000.0,
        data_dir: str = "data/raw",
        enable_adaptive: bool = True,
    ):
        self.executor = executor
        self.pairs = pairs or PROFITABLE_PAIRS
        self.initial_capital = initial_capital
        self.model: lgb.Booster | None = None
        self.positions: dict[str, Position] = {}
        self._running = False
        self._daily_pnl = 0.0
        self._daily_reset_date = None
        self._price_exchange: ccxt_async.Exchange | None = None
        self._feature_names: list[str] | None = None
        self._model_path = model_path
        self._data_dir = data_dir
        self._enable_adaptive = enable_adaptive
        self._trade_history: list[dict] = []  # rolling window of recent trades
        self._last_retrain: datetime | None = None
        self._retrain_count = 0

        if model_path and Path(model_path).exists():
            self.model = lgb.Booster(model_file=str(model_path))
            self._feature_names = self.model.feature_name()
            logger.info("Loaded model from %s (%d features)", model_path, len(self._feature_names))

    async def run(self, interval_seconds: int = 14400):
        """Main trading loop. Runs every bar_interval."""
        self._running = True
        logger.info("ML Live Trader starting")
        logger.info("  Pairs: %s", self.pairs)
        logger.info("  Interval: %d seconds (%d hours)", interval_seconds, interval_seconds // 3600)
        logger.info("  Max positions: %d", MAX_CONCURRENT_POSITIONS)

        while self._running:
            try:
                await self._trading_cycle()
            except Exception:
                logger.exception("Trading cycle error")

            if not self._running:
                break

            logger.info("Sleeping %d seconds until next bar...", interval_seconds)
            await asyncio.sleep(interval_seconds)

    async def _trading_cycle(self):
        now = datetime.now(timezone.utc)
        today = now.date()
        if self._daily_reset_date != today:
            self._daily_pnl = 0.0
            self._daily_reset_date = today

        # Check daily drawdown limit
        balance = await self.executor.get_balance()
        total_value = balance.get("total_value", self.initial_capital)
        if total_value > 0:
            daily_dd = self._daily_pnl / total_value
            if daily_dd < -DAILY_DRAWDOWN_LIMIT:
                logger.warning("Daily drawdown limit hit (%.2f%%), skipping cycle",
                               daily_dd * 100)
                return

        # Fetch current prices
        prices = await self._fetch_prices()
        if not prices:
            logger.warning("Could not fetch prices")
            return

        # Check stop losses on existing positions
        sl_results = await self.executor.check_stop_losses(prices)
        for result in sl_results:
            if result.success and result.symbol in self.positions:
                pos = self.positions.pop(result.symbol)
                pnl = (result.price - pos.entry_price) / pos.entry_price * pos.direction
                self._daily_pnl += pnl * pos.amount * pos.entry_price
                self._record_trade(pnl)
                self._log_event("stop_loss", pos, result.price, pnl)

        # Update bar counts for existing positions
        for sym, pos in list(self.positions.items()):
            pos.bars_held += 1

            # Time-based exit
            if pos.bars_held >= MAX_HOLD_BARS:
                price = prices.get(sym.replace("/", "-"), 0)
                if price > 0:
                    result = await self.executor.execute_sell(sym, pos.amount, price)
                    if result.success:
                        pnl = (price - pos.entry_price) / pos.entry_price * pos.direction
                        self._daily_pnl += pnl * pos.amount * pos.entry_price
                        self._record_trade(pnl)
                        self._log_event("time_exit", pos, price, pnl)
                        del self.positions[sym]

        # Generate signals (only if we have a model)
        if self.model is None:
            logger.info("No model loaded, skipping signal generation")
            return

        signals = await self._generate_signals(prices)

        for sym, sig in signals.items():
            direction = sig["direction"]
            confidence = sig["confidence"]

            # Exit on reversal
            if sym in self.positions and self.positions[sym].direction != direction:
                pos = self.positions[sym]
                price = prices.get(sym.replace("/", "-"), 0)
                if price > 0:
                    result = await self.executor.execute_sell(sym, pos.amount, price)
                    if result.success:
                        pnl = (price - pos.entry_price) / pos.entry_price * pos.direction
                        self._daily_pnl += pnl * pos.amount * pos.entry_price
                        self._record_trade(pnl)
                        self._log_event("reversal_exit", pos, price, pnl)
                        del self.positions[sym]

            # Enter new position
            if sym not in self.positions and direction == 1:  # long-only for now
                if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                    continue

                price = prices.get(sym.replace("/", "-"), 0)
                if price <= 0:
                    continue

                amount_usd = total_value * POSITION_SIZE_PCT
                result = await self.executor.execute_buy(
                    sym, amount_usd, price,
                    stop_loss_pct=STOP_LOSS_PCT,
                )
                if result.success:
                    self.positions[sym] = Position(
                        symbol=sym,
                        entry_price=result.price,
                        entry_time=datetime.now(timezone.utc),
                        direction=1,
                        amount=result.amount,
                        trade_id=result.order_id,
                    )
                    self._log_event("entry", self.positions[sym], result.price, 0)

        # Log portfolio state
        rolling_s = self._rolling_sharpe()
        logger.info("Portfolio: %d positions, daily P&L: $%.2f, rolling Sharpe: %s",
                     len(self.positions), self._daily_pnl,
                     f"{rolling_s:.2f}" if rolling_s is not None else "n/a")

        # Adaptive retraining check
        await self._check_retrain()

    async def _get_price_exchange(self) -> ccxt_async.Exchange:
        if self._price_exchange is None:
            self._price_exchange = ccxt_async.coinbase({"enableRateLimit": True})
        return self._price_exchange

    async def _fetch_prices(self) -> dict[str, float]:
        """Fetch current prices for all pairs from Coinbase."""
        exchange = await self._get_price_exchange()
        prices = {}
        for pair in self.pairs:
            symbol = pair.replace("-", "/")
            try:
                ticker = await exchange.fetch_ticker(symbol)
                prices[pair.replace("/", "-")] = ticker.get("last", 0)
            except Exception:
                logger.debug("Could not fetch price for %s", pair)
        return prices

    async def _fetch_ohlcv(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch recent 4h OHLCV candles for a symbol."""
        exchange = await self._get_price_exchange()
        raw = await exchange.fetch_ohlcv(symbol, timeframe="4h", limit=limit)
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    async def _generate_signals(self, prices: dict) -> dict:
        """Fetch latest bars, compute features, run model, return signals."""
        if self.model is None:
            return {}

        signals = {}

        # Fetch BTC OHLCV for cross-pair features
        try:
            btc_ohlcv = await self._fetch_ohlcv("BTC/USD", limit=100)
            btc_price = btc_ohlcv["close"] if not btc_ohlcv.empty else pd.Series(dtype=float)
        except Exception:
            btc_price = pd.Series(dtype=float)

        for pair in self.pairs:
            symbol = pair.replace("-", "/")
            try:
                ohlcv = await self._fetch_ohlcv(symbol, limit=100)
                if ohlcv.empty or len(ohlcv) < 20:
                    continue

                # Build features from OHLCV
                features = pd.DataFrame(index=ohlcv.index)
                features["price"] = ohlcv["close"]
                features["price_open"] = ohlcv["open"]
                features["price_high"] = ohlcv["high"]
                features["price_low"] = ohlcv["low"]
                features["tick_count"] = 1  # placeholder

                ret_feats = compute_return_features(ohlcv["close"], lags=[1, 5, 15, 60])
                temporal = compute_temporal_features(ohlcv.index)
                features = features.join(ret_feats).join(temporal)

                # Cross-pair features
                if not btc_price.empty and symbol != "BTC/USD":
                    cross = compute_cross_pair_features(btc_price, ohlcv["close"])
                    features = features.join(cross)

                # TA features
                p = ohlcv["close"]
                delta = p.diff()
                gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=5).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=5).mean()
                rs = gain / loss.replace(0, np.nan)
                features["rsi_14"] = 100 - 100 / (1 + rs)

                sma20 = p.rolling(20, min_periods=10).mean()
                std20 = p.rolling(20, min_periods=10).std()
                features["bb_pctb"] = (p - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)

                tr = ohlcv["high"] - ohlcv["low"]
                features["atr_14"] = tr.rolling(14, min_periods=5).mean() / p.replace(0, np.nan)

                if "volume" in ohlcv.columns:
                    vol_ma = ohlcv["volume"].rolling(20, min_periods=5).mean()
                    features["vol_ratio"] = ohlcv["volume"] / vol_ma.replace(0, np.nan)

                # Use latest row for prediction
                latest = features.iloc[[-1]].copy()
                latest = latest.drop(columns=["price", "price_open", "price_high", "price_low"],
                                     errors="ignore")
                latest = latest.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

                # Align columns with model
                if self._feature_names:
                    for col in self._feature_names:
                        if col not in latest.columns:
                            latest[col] = np.nan
                    latest = latest.reindex(columns=self._feature_names)

                pred = self.model.predict(latest)[0]

                if pred > CONFIDENCE_THRESHOLD:
                    signals[pair] = {"direction": 1, "confidence": pred}
                elif pred < (1 - CONFIDENCE_THRESHOLD):
                    signals[pair] = {"direction": -1, "confidence": 1 - pred}

            except Exception as e:
                logger.debug("Signal generation failed for %s: %s", pair, str(e)[:80])

        return signals

    def _record_trade(self, pnl: float):
        """Track trade P&L for adaptive retraining decisions."""
        self._trade_history.append({
            "timestamp": datetime.now(timezone.utc),
            "pnl": pnl,
        })
        # Keep last 50 trades
        if len(self._trade_history) > 50:
            self._trade_history = self._trade_history[-50:]

    def _rolling_sharpe(self, n: int = 20) -> float | None:
        """Compute Sharpe ratio over the last N trades."""
        if len(self._trade_history) < n:
            return None
        recent = [t["pnl"] for t in self._trade_history[-n:]]
        std = np.std(recent)
        if std < 1e-10:
            return 0.0
        return np.mean(recent) / std

    async def _check_retrain(self):
        """Check if model performance has degraded and trigger retrain if needed."""
        if not self._enable_adaptive:
            return

        sharpe = self._rolling_sharpe(RETRAIN_MIN_TRADES)
        if sharpe is None:
            return  # not enough trades yet

        # Cooldown: don't retrain more than once per 24h
        if self._last_retrain:
            hours_since = (datetime.now(timezone.utc) - self._last_retrain).total_seconds() / 3600
            if hours_since < 24:
                return

        if sharpe < RETRAIN_SHARPE_THRESHOLD:
            logger.warning("Rolling Sharpe %.2f < threshold %.2f — triggering retrain",
                           sharpe, RETRAIN_SHARPE_THRESHOLD)
            await self._retrain_model()

    async def _retrain_model(self):
        """Retrain the model on the most recent data and hot-swap."""
        try:
            from src.data.influx_client import LocalParquetClient

            data_dir = Path(self._data_dir)
            if not data_dir.exists():
                logger.error("Data dir %s not found, cannot retrain", data_dir)
                return

            client = LocalParquetClient(str(data_dir))
            stop = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            start = (datetime.now(timezone.utc) - timedelta(days=RETRAIN_WINDOW_DAYS)).strftime("%Y-%m-%d")

            # Import feature builder from model_search_v2
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from scripts.model_search_v2 import build_dataset_v2, make_drop_cols

            logger.info("Retraining on %s -> %s (%d days)", start, stop, RETRAIN_WINDOW_DAYS)
            ds = build_dataset_v2(client, "4h", start, stop, horizons=[30])
            client.close()

            if ds.empty or len(ds) < 100:
                logger.warning("Not enough data for retrain (%d rows)", len(ds))
                return

            # Prepare training data
            work = ds.copy().reset_index()
            if work.columns[0] not in ds.columns:
                work.rename(columns={work.columns[0]: "timestamp"}, inplace=True)

            target_col = "direction_30"
            work = work.dropna(subset=[target_col])
            ret_col = "fwd_ret_30"
            if ret_col in work.columns:
                work = work[work[ret_col].abs() >= 0.002]

            y = work[target_col].astype(int)
            drop = make_drop_cols([30])
            all_drop = list(set(drop) & set(work.columns))
            X = work.drop(columns=all_drop, errors="ignore")
            X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

            if len(X) < 50:
                logger.warning("Too few samples after filtering: %d", len(X))
                return

            params = {
                "objective": "binary", "metric": "binary_logloss",
                "boosting_type": "gbdt", "num_leaves": 127, "learning_rate": 0.05,
                "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
                "min_child_samples": 30, "lambda_l1": 0.1, "lambda_l2": 1.0,
                "verbose": -1, "n_jobs": -1, "seed": 42,
            }

            dtrain = lgb.Dataset(X, label=y)
            new_model = lgb.train(params, dtrain, num_boost_round=500)

            # Hot-swap
            self.model = new_model
            self._feature_names = new_model.feature_name()
            self._retrain_count += 1
            self._last_retrain = datetime.now(timezone.utc)

            # Save the retrained model
            model_path = MODEL_DIR / f"retrained_{self._retrain_count}.txt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            new_model.save_model(str(model_path))

            logger.info("Retrain #%d complete: %d samples, %d features. Saved to %s",
                        self._retrain_count, len(X), len(X.columns), model_path)

        except Exception:
            logger.exception("Retrain failed")

    def _log_event(self, event_type: str, pos: Position, price: float, pnl: float):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "symbol": pos.symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "pnl": pnl,
            "bars_held": pos.bars_held,
            "amount": pos.amount,
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info("  [%s] %s: %.4f -> %.4f (pnl=%.4f, bars=%d)",
                     event_type, pos.symbol, pos.entry_price, price, pnl, pos.bars_held)

    def stop(self):
        self._running = False


async def async_main(args):
    from src.config import load_config
    from src.data.store import TradeStore

    config = load_config()

    if args.mode == "live":
        from src.execution.coinbase import CoinbaseExecutor
        store = TradeStore("data/ml_live_trading.db")
        executor = CoinbaseExecutor(config, store)
        logger.info("Using LIVE CoinbaseExecutor")
    else:
        from src.execution.paper import PaperExecutor
        store = TradeStore("data/ml_paper_trading.db")
        executor = PaperExecutor(args.capital, store)
        logger.info("Using PaperExecutor with $%.2f", args.capital)

    trader = MLLiveTrader(
        executor=executor,
        model_path=args.model,
        pairs=args.pairs.split(",") if args.pairs else None,
        initial_capital=args.capital,
        data_dir=args.data_dir,
        enable_adaptive=args.adaptive,
    )

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, trader.stop)

    try:
        await trader.run(interval_seconds=args.interval)
    finally:
        if trader._price_exchange:
            await trader._price_exchange.close()
        if hasattr(executor, "close"):
            await executor.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="ML-driven live trader")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--model", default="models/best_model.txt",
                        help="Path to LightGBM model file")
    parser.add_argument("--pairs", default=None,
                        help="Comma-separated pairs (default: profitable pairs from search)")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--interval", type=int, default=14400,
                        help="Trading interval in seconds (default: 4h = 14400)")
    parser.add_argument("--data-dir", default="data/raw",
                        help="Path to parquet data for adaptive retraining")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Enable adaptive retraining (default: True)")
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false",
                        help="Disable adaptive retraining")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
