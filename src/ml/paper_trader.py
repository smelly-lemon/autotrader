"""Paper trading runner for ML strategies.

Connects the ML models to the paper executor for live validation.
Designed to run continuously, generating signals from live InfluxDB data
and executing through the paper trading engine.

Usage:
    # Train models first, then run paper trading
    python -m src.ml.paper_trader \
        --lead-lag-model models/lead_lag.txt \
        --swing-model models/swing.txt \
        --interval 60 \
        --capital 10000

    # Or train inline (auto-trains on recent data before starting)
    python -m src.ml.paper_trader --auto-train --days 30 --interval 60
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.influx_client import HesiodInfluxClient, PRODUCT_IDS
from src.data.store import TradeStore
from src.execution.paper import PaperExecutor
from src.ml.backtest import FEE_TIERS, DEFAULT_FEE_TIER
from src.ml.features import FeatureBuilder, ALT_PAIRS, BTC_PAIR, STABLECOIN_PAIR
from src.ml.signal_provider import MLSignalProvider

logger = logging.getLogger(__name__)

PAPER_DB_PATH = "data/paper_trading.db"
LOG_PATH = "data/paper_trading_log.jsonl"


class PaperTradingRunner:
    """Runs ML strategies in paper trading mode with persistent state."""

    def __init__(
        self,
        lead_lag_model_path: str | Path | None = None,
        swing_model_path: str | Path | None = None,
        initial_capital: float = 10_000.0,
        fee_tier: str = DEFAULT_FEE_TIER,
        signal_interval_seconds: int = 60,
        confidence_threshold: float = 0.55,
        max_holding_minutes: int = 60,
        position_size_frac: float = 0.10,
    ):
        self.store = TradeStore(PAPER_DB_PATH)
        self.executor = PaperExecutor(initial_capital, self.store)

        fees = FEE_TIERS[fee_tier]
        self.maker_fee = fees["maker"]
        self.taker_fee = fees["taker"]
        self.fee_tier = fee_tier

        self.signal_provider = MLSignalProvider(lead_lag_model_path, swing_model_path)
        self.feature_builder = FeatureBuilder()

        self.interval_seconds = signal_interval_seconds
        self.confidence_threshold = confidence_threshold
        self.max_holding_minutes = max_holding_minutes
        self.position_size_frac = position_size_frac

        self._running = True
        self._cycle_count = 0
        self._log_file = Path(LOG_PATH)
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def _log_event(self, event_type: str, data: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": self._cycle_count,
            "event": event_type,
            **data,
        }
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    async def _get_current_prices(self) -> dict[str, float]:
        """Fetch latest prices from InfluxDB ticker data."""
        prices = {}
        client = self.feature_builder._client
        now = datetime.now(timezone.utc)
        start = (now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")

        for pair in PRODUCT_IDS:
            try:
                ticker = client.get_ticker_data(pair, start=start)
                if not ticker.empty and "price" in ticker.columns:
                    ticker["price"] = pd.to_numeric(ticker["price"], errors="coerce")
                    last_price = ticker["price"].dropna().iloc[-1]
                    prices[pair] = float(last_price)
            except Exception:
                pass

        return prices

    async def _generate_lead_lag_signals(self) -> list[dict]:
        """Generate lead-lag signals for all alt pairs."""
        signals = []
        now = datetime.now(timezone.utc)
        start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        stop = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            df = self.feature_builder.build_lead_lag_dataset(start, stop, "1min")
            if df.empty:
                return signals

            for pair in df["product_id"].unique():
                pair_data = df[df["product_id"] == pair]
                sig = self.signal_provider.get_lead_lag_signal(pair_data)
                if sig:
                    signals.append(sig)
        except Exception:
            logger.exception("Failed to generate lead-lag signals")

        return signals

    async def _generate_swing_signals(self) -> list[dict]:
        """Generate swing signals for all pairs."""
        signals = []
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        stop = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            df = self.feature_builder.build_swing_dataset(start, stop, "4h")
            if df.empty:
                return signals

            for pair in df["product_id"].unique():
                pair_data = df[df["product_id"] == pair]
                sig = self.signal_provider.get_swing_signal(pair_data)
                if sig:
                    signals.append(sig)
        except Exception:
            logger.exception("Failed to generate swing signals")

        return signals

    async def _execute_signal(self, sig: dict, current_price: float):
        """Execute a trade based on a signal."""
        pair = sig["product_id"]
        direction = sig["signal"]
        confidence = sig["confidence"]
        model = sig["model"]

        if confidence < (self.confidence_threshold - 0.5) * 2:
            return

        balance = await self.executor.get_balance()
        trade_value = balance["cash"] * self.position_size_frac

        if direction == "LONG" and pair not in self.executor.positions:
            if trade_value < 10:
                logger.info("Skipping %s: insufficient cash ($%.2f)", pair, balance["cash"])
                return

            result = await self.executor.execute_buy(
                symbol=pair,
                amount_usd=trade_value,
                current_price=current_price,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
            )
            if result.success:
                self._log_event("trade", {
                    "action": "buy", "pair": pair, "price": current_price,
                    "amount_usd": trade_value, "model": model,
                    "confidence": confidence, "signal": direction,
                })

        elif direction == "SHORT" and pair in self.executor.positions:
            pos = self.executor.positions[pair]
            result = await self.executor.execute_sell(
                symbol=pair,
                amount=pos.amount,
                current_price=current_price,
            )
            if result.success:
                pnl = (current_price - pos.entry_price) * pos.amount
                self._log_event("trade", {
                    "action": "sell", "pair": pair, "price": current_price,
                    "pnl": pnl, "model": model,
                    "confidence": confidence, "signal": direction,
                })

    async def run_cycle(self):
        """Run one signal-generation and execution cycle."""
        self._cycle_count += 1
        logger.info("=" * 40)
        logger.info("Cycle %d starting at %s", self._cycle_count,
                     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

        prices = await self._get_current_prices()
        if not prices:
            logger.warning("No prices available, skipping cycle")
            return

        # Check stop-losses on existing positions
        triggered = await self.executor.check_stop_losses(prices)
        for t in triggered:
            self._log_event("stop_triggered", {"pair": t.symbol, "price": t.price})

        # Generate signals from both strategies
        lead_lag_signals = await self._generate_lead_lag_signals()
        swing_signals = await self._generate_swing_signals()

        all_signals = lead_lag_signals + swing_signals

        # Execute actionable signals
        for sig in all_signals:
            pair = sig["product_id"]
            if pair in prices:
                await self._execute_signal(sig, prices[pair])

        # Log portfolio state
        balance = await self.executor.get_balance()
        total_value = balance["cash"]
        for sym, pos_info in self.executor.positions.items():
            if sym in prices:
                total_value += pos_info.amount * prices[sym]

        self._log_event("portfolio", {
            "cash": balance["cash"],
            "total_value": total_value,
            "n_positions": len(self.executor.positions),
            "positions": {s: {"amount": p.amount, "entry": p.entry_price}
                          for s, p in self.executor.positions.items()},
            "n_signals": len(all_signals),
        })

        pnl_pct = (total_value / self.executor._initial_balance - 1) * 100
        logger.info(
            "Portfolio: $%.2f (%.2f%%) | Cash: $%.2f | Positions: %d | Signals: %d",
            total_value, pnl_pct, balance["cash"],
            len(self.executor.positions), len(all_signals),
        )

    async def run(self):
        """Main paper trading loop."""
        logger.info("Starting paper trading runner")
        logger.info("  Fee tier: %s (maker=%.2f%%, taker=%.2f%%)",
                     self.fee_tier, self.maker_fee * 100, self.taker_fee * 100)
        logger.info("  Capital: $%.2f", self.executor._initial_balance)
        logger.info("  Signal interval: %ds", self.interval_seconds)
        logger.info("  Confidence threshold: %.2f", self.confidence_threshold)
        logger.info("  Log file: %s", self._log_file)

        while self._running:
            try:
                await self.run_cycle()
            except Exception:
                logger.exception("Cycle %d failed", self._cycle_count)

            await asyncio.sleep(self.interval_seconds)

    def stop(self):
        self._running = False
        self.feature_builder.close()

    def print_summary(self):
        """Print summary of paper trading performance."""
        if not self._log_file.exists():
            print("No paper trading log found.")
            return

        portfolio_events = []
        trade_events = []

        with open(self._log_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry["event"] == "portfolio":
                    portfolio_events.append(entry)
                elif entry["event"] == "trade":
                    trade_events.append(entry)

        if not portfolio_events:
            print("No portfolio snapshots recorded.")
            return

        initial = self.executor._initial_balance
        latest = portfolio_events[-1]
        total_value = latest.get("total_value", initial)
        total_return = (total_value / initial - 1) * 100

        print("\n" + "=" * 60)
        print("Paper Trading Summary")
        print("=" * 60)
        print(f"  Period: {portfolio_events[0]['timestamp'][:10]} .. {portfolio_events[-1]['timestamp'][:10]}")
        print(f"  Cycles: {len(portfolio_events)}")
        print(f"  Trades: {len(trade_events)}")
        print(f"  Initial capital: ${initial:,.2f}")
        print(f"  Current value:   ${total_value:,.2f}")
        print(f"  Total return:    {total_return:+.2f}%")

        if trade_events:
            buys = [t for t in trade_events if t["action"] == "buy"]
            sells = [t for t in trade_events if t["action"] == "sell"]
            pnls = [t["pnl"] for t in sells if "pnl" in t]
            wins = [p for p in pnls if p > 0]

            print(f"\n  Buys: {len(buys)}  |  Sells: {len(sells)}")
            if pnls:
                print(f"  Win rate: {len(wins)/len(pnls):.1%}")
                print(f"  Avg PnL per closed trade: ${np.mean(pnls):+.2f}")
                print(f"  Total realized PnL: ${sum(pnls):+.2f}")

        # Equity curve from portfolio snapshots
        values = [p.get("total_value", initial) for p in portfolio_events]
        peak = max(values)
        drawdown = min((v - peak) / peak for v in values)
        print(f"\n  Max drawdown: {drawdown:.2%}")
        print("=" * 60)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="ML Paper Trading Runner")
    parser.add_argument("--lead-lag-model", type=str, default=None, help="Path to lead-lag model")
    parser.add_argument("--swing-model", type=str, default=None, help="Path to swing model")
    parser.add_argument("--auto-train", action="store_true", help="Auto-train models on recent data")
    parser.add_argument("--days", type=int, default=30, help="Days of data for auto-training")
    parser.add_argument("--interval", type=int, default=60, help="Signal interval in seconds")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial paper capital")
    parser.add_argument("--fee-tier", choices=list(FEE_TIERS.keys()), default=DEFAULT_FEE_TIER)
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--summary", action="store_true", help="Print summary and exit")
    args = parser.parse_args()

    lead_lag_path = args.lead_lag_model
    swing_path = args.swing_model

    if args.auto_train:
        logger.info("Auto-training models on last %d days...", args.days)
        stop = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")

        builder = FeatureBuilder()
        try:
            Path("models").mkdir(exist_ok=True)

            # Train lead-lag
            logger.info("Training lead-lag model...")
            from src.ml.lead_lag import train_final_model as train_ll
            ll_df = builder.build_lead_lag_dataset(start, stop, "1min")
            if not ll_df.empty:
                model, _ = train_ll(ll_df, btc_move_only=True)
                lead_lag_path = "models/lead_lag.txt"
                model.save_model(lead_lag_path)
                logger.info("Lead-lag model saved to %s", lead_lag_path)

            # Train swing
            logger.info("Training swing model...")
            from src.ml.swing import train_final_swing_model as train_sw
            sw_df = builder.build_swing_dataset(start, stop, "4h")
            if not sw_df.empty:
                model, _ = train_sw(sw_df, use_hmm=False)
                swing_path = "models/swing.txt"
                model.save_model(swing_path)
                logger.info("Swing model saved to %s", swing_path)
        finally:
            builder.close()

    runner = PaperTradingRunner(
        lead_lag_model_path=lead_lag_path,
        swing_model_path=swing_path,
        initial_capital=args.capital,
        fee_tier=args.fee_tier,
        signal_interval_seconds=args.interval,
        confidence_threshold=args.threshold,
    )

    if args.summary:
        runner.print_summary()
        return

    # Handle graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received, stopping...")
        runner.stop()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        pass
    finally:
        runner.print_summary()
        runner.stop()


if __name__ == "__main__":
    main()
