"""ML-driven crypto auto-trader.

Continuously collects 1-minute candles, trains LightGBM models on OHLCV
features, generates trading signals, and executes through paper (or live)
with hard risk limits.

Usage:
    python -m src.main
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.config import AppConfig, load_config, MODEL_DIR
from src.data.collector import MarketDataCollector
from src.data.store import TradeStore
from src.execution.paper import PaperExecutor
from src.ml.ohlcv_features import build_training_dataset, build_live_features
from src.ml.trainer import (
    walk_forward_cv, train_final_model, save_model, load_model, print_cv_summary,
)

logger = logging.getLogger(__name__)


class MLTrader:
    """Main trading daemon: data collection → ML signals → execution."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.store = TradeStore()
        self.collector = MarketDataCollector(config)

        if config.trading.mode == "paper":
            self.executor = PaperExecutor(
                config.trading.initial_balance,
                self.store,
                fee_pct=config.trading.taker_fee_pct,
                slippage_pct=config.trading.slippage_pct,
            )
            self.executor.restore_state()
        else:
            from src.execution.coinbase import CoinbaseExecutor
            self.executor = CoinbaseExecutor(config, self.store)

        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []
        self.model_auc: float = 0.0

        self._running = False
        self._cycle_count = 0
        self._last_train_time: float = 0
        self._last_candle_fetch: float = 0

    # ── Data Collection ──────────────────────────────────────────────

    async def backfill_candles(self, days: int = 30):
        """Backfill historical 1m candles for all pairs."""
        for pair in self.config.trading.pairs:
            existing = self.store.get_candle_count(pair, "1m")
            if existing > days * 1400:
                logger.info("Backfill %s: already have %d candles", pair, existing)
                continue

            latest_ts = self.store.get_latest_candle_ts(pair, "1m")
            if latest_ts:
                import pandas as pd
                start_dt = pd.Timestamp(latest_ts) + timedelta(minutes=1)
            else:
                start_dt = datetime.now(timezone.utc) - timedelta(days=days)

            logger.info("Backfilling %s from %s...", pair, str(start_dt)[:10])
            since_ms = int(start_dt.timestamp() * 1000)
            total_new = 0

            while since_ms < int(datetime.now(timezone.utc).timestamp() * 1000):
                try:
                    df = await self.collector.fetch_ohlcv_since(
                        pair, since_ms, timeframe="1m", limit=300,
                    )
                    if df.empty:
                        break
                    self.store.upsert_candles(pair, "1m", df)
                    total_new += len(df)
                    since_ms = int(df.index[-1].timestamp() * 1000) + 60_000
                    await asyncio.sleep(0.3)
                except Exception:
                    logger.exception("Backfill error for %s", pair)
                    break

            total = self.store.get_candle_count(pair, "1m")
            logger.info("Backfill %s complete: +%d candles (%d total)", pair, total_new, total)

    async def fetch_latest_candles(self):
        """Fetch latest 1m candles and store them."""
        for pair in self.config.trading.pairs:
            try:
                df = await self.collector.fetch_ohlcv(pair, timeframe="1m", limit=5)
                if not df.empty:
                    self.store.upsert_candles(pair, "1m", df)
            except Exception:
                logger.exception("Failed to fetch candles for %s", pair)

    # ── Model Training ───────────────────────────────────────────────

    def train_model(self):
        """Train LightGBM on accumulated candle data with walk-forward CV."""
        logger.info("Training model...")
        target = f"direction_{self.config.ml.target_horizon}"

        X, y = build_training_dataset(
            self.store, self.config.trading.pairs, target_col=target,
        )
        if X.empty or len(X) < 1000:
            logger.warning("Insufficient data for training: %d rows", len(X))
            return

        # Walk-forward CV to estimate out-of-sample performance
        results = walk_forward_cv(X, y, n_folds=self.config.ml.cv_folds)
        print_cv_summary(results)

        if results:
            mean_auc = np.mean([r.auc for r in results])
            self.model_auc = mean_auc
            logger.info("CV mean AUC: %.4f (threshold: %.4f)",
                        mean_auc, self.config.ml.min_auc_to_trade)

            if mean_auc < self.config.ml.min_auc_to_trade:
                logger.warning("Model AUC below threshold — will not trade")
        else:
            self.model_auc = 0.0

        # Train final model on all data
        self.model = train_final_model(X, y)
        self.feature_names = self.model.feature_name()
        self._last_train_time = time.time()

        model_path = MODEL_DIR / "current_model.txt"
        save_model(self.model, model_path)

    # ── Signal Generation & Execution ────────────────────────────────

    async def run_signal_cycle(self):
        """Generate ML signals and execute trades."""
        if self.model is None:
            return

        if self.model_auc < self.config.ml.min_auc_to_trade:
            return

        # Build live features for each pair
        live_features = build_live_features(
            self.store, self.config.trading.pairs, lookback=300,
        )

        # Get current prices
        prices: dict[str, float] = {}
        for pair in self.config.trading.pairs:
            try:
                ticker = await self.collector.fetch_ticker(pair)
                prices[pair] = ticker["last"]
            except Exception:
                pass

        # Check stop-losses first
        if prices:
            triggered = await self.executor.check_stop_losses(prices)
            for t in triggered:
                logger.warning("Stop triggered: %s @ $%.2f", t.symbol, t.price)

        # Generate predictions and act
        for pair, feat_row in live_features.items():
            price = prices.get(pair)
            if price is None:
                continue

            # Align features with model's expected columns
            for col in self.feature_names:
                if col not in feat_row.columns:
                    feat_row[col] = np.nan
            X_pred = feat_row[self.feature_names]

            try:
                proba = self.model.predict(X_pred)[0]
            except Exception:
                logger.exception("Prediction failed for %s", pair)
                continue

            confidence = abs(proba - 0.5) * 2
            threshold = self.config.ml.confidence_threshold

            # Long signal
            if proba > threshold and pair not in self.executor.positions:
                await self._execute_buy(pair, price, proba, confidence)

            # Exit signal (model says down)
            elif proba < (1 - threshold) and pair in self.executor.positions:
                pos = self.executor.positions[pair]
                await self._execute_sell(pair, pos.amount, price, proba)

            # Time-based exit
            elif pair in self.executor.positions:
                pos = self.executor.positions[pair]
                trade = self.store.conn.execute(
                    "SELECT timestamp FROM trades WHERE id=?", (pos.trade_id,)
                ).fetchone()
                if trade:
                    import pandas as pd
                    entry_time = pd.Timestamp(trade["timestamp"])
                    minutes_held = (datetime.now(timezone.utc) - entry_time.to_pydatetime()).total_seconds() / 60
                    if minutes_held > self.config.ml.max_holding_minutes:
                        logger.info("Time exit: %s held %.0f min", pair, minutes_held)
                        await self._execute_sell(pair, pos.amount, price, proba)

    async def _execute_buy(self, pair: str, price: float, proba: float, confidence: float):
        balance = await self.executor.get_balance()
        size_pct = min(self.config.ml.position_size_pct, self.config.risk.max_position_pct)
        trade_value = balance["cash"] * size_pct

        if trade_value < 1.0:
            return

        # Check risk limits
        open_trades = self.store.get_open_trades()
        if len(open_trades) >= self.config.risk.max_open_positions:
            return
        if any(t["symbol"] == pair for t in open_trades):
            return

        # Check daily drawdown
        daily_pnl = self.store.get_daily_pnl()
        portfolio_value = balance["cash"]
        for pos in self.executor.positions.values():
            portfolio_value += pos.amount * price
        if portfolio_value > 0 and daily_pnl < 0:
            dd = abs(daily_pnl) / portfolio_value
            if dd >= self.config.risk.daily_drawdown_limit_pct:
                logger.warning("Daily drawdown limit hit: %.2f%%", dd * 100)
                return

        result = await self.executor.execute_buy(
            symbol=pair,
            amount_usd=trade_value,
            current_price=price,
            stop_loss_pct=self.config.ml.stop_loss_pct,
            take_profit_pct=self.config.ml.take_profit_pct,
        )

        if result.success:
            self.store.log_decision(
                symbol=pair, model_tier="ml", model_name="lgbm_ohlcv",
                action="buy", confidence=confidence,
                reasoning=f"P(up)={proba:.3f}, AUC={self.model_auc:.3f}",
                raw_output="", was_executed=True,
            )

    async def _execute_sell(self, pair: str, amount: float, price: float, proba: float):
        result = await self.executor.execute_sell(pair, amount, price)
        if result.success:
            self.store.log_decision(
                symbol=pair, model_tier="ml", model_name="lgbm_ohlcv",
                action="sell", confidence=abs(proba - 0.5) * 2,
                reasoning=f"P(up)={proba:.3f} exit signal",
                raw_output="", was_executed=True,
            )

    # ── Main Loop ────────────────────────────────────────────────────

    async def run(self):
        self._running = True

        logger.info("=" * 60)
        logger.info("ML Auto-Trader starting in %s mode", self.config.trading.mode.upper())
        logger.info("  Pairs: %s", ", ".join(self.config.trading.pairs))
        logger.info("  Target: %d-min direction", self.config.ml.target_horizon)
        logger.info("  Confidence threshold: %.2f", self.config.ml.confidence_threshold)
        logger.info("  Signal interval: %ds", self.config.schedule.signal_interval_seconds)
        if self.config.trading.mode == "paper":
            logger.info("  Paper balance: $%.2f", self.config.trading.initial_balance)
        logger.info("=" * 60)

        # Phase 1: Backfill historical data
        await self.backfill_candles(days=self.config.schedule.data_retention_days)

        # Phase 2: Initial model training
        self.train_model()

        # Try loading saved model if training didn't produce one
        if self.model is None:
            model_path = MODEL_DIR / "current_model.txt"
            if model_path.exists():
                self.model = load_model(model_path)
                self.feature_names = self.model.feature_name()
                logger.info("Loaded saved model from %s", model_path)

        # Phase 3: Trading loop
        while self._running:
            self._cycle_count += 1
            try:
                # Fetch latest candles
                now = time.time()
                if now - self._last_candle_fetch >= self.config.schedule.candle_fetch_interval:
                    await self.fetch_latest_candles()
                    self._last_candle_fetch = now

                # Retrain periodically
                retrain_interval = self.config.schedule.retrain_hours * 3600
                if now - self._last_train_time >= retrain_interval:
                    self.train_model()

                # Generate signals and trade
                await self.run_signal_cycle()

                # Log status
                balance = await self.executor.get_balance()
                total = balance["total_value"]
                daily_pnl = self.store.get_daily_pnl()

                if self._cycle_count % 5 == 0:
                    status = "ACTIVE" if (self.model and self.model_auc >= self.config.ml.min_auc_to_trade) else "OBSERVING"
                    logger.info(
                        "[%s] Cycle %d | $%.2f | PnL: $%.2f | Positions: %d | AUC: %.3f",
                        status, self._cycle_count, total, daily_pnl,
                        len(self.executor.positions), self.model_auc,
                    )

            except Exception:
                logger.exception("Error in cycle %d", self._cycle_count)

            try:
                await asyncio.sleep(self.config.schedule.signal_interval_seconds)
            except asyncio.CancelledError:
                break

        logger.info("Shutting down...")

    async def shutdown(self):
        self._running = False
        await self.collector.close()
        self.store.close()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    setup_logging()
    config = load_config()

    if config.trading.mode == "live":
        if not config.coinbase_api_key_name:
            logger.error("COINBASE_API_KEY_NAME not set in .env")
            sys.exit(1)
        print("WARNING: LIVE TRADING. Press Ctrl+C within 10s to abort.")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("Aborted.")
            sys.exit(0)

    trader = MLTrader(config)
    loop = asyncio.new_event_loop()

    def handle_signal(sig, frame):
        print("\nShutting down...")
        loop.create_task(trader.shutdown())

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(trader.run())
    finally:
        loop.run_until_complete(trader.shutdown())
        loop.close()


if __name__ == "__main__":
    main()
