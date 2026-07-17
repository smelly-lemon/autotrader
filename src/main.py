from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

from rich.console import Console
from rich.logging import RichHandler

from src.config import AppConfig, load_config
from src.data.collector import MarketDataCollector
from src.data.store import TradeStore
from src.execution.paper import PaperExecutor
from src.llm.client import OllamaClient
from src.risk.manager import RiskManager
from src.strategy.analyzer import DeepAnalyzer
from src.strategy.portfolio import PortfolioStrategist
from src.strategy.scanner import MarketScanner

console = Console()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


class AutoTrader:
    """Main trading daemon that orchestrates scanning, analysis, and execution."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.store = TradeStore()
        self.collector = MarketDataCollector(config)
        self.llm = OllamaClient(config)
        self.risk_manager = RiskManager(config.risk, self.store)

        if config.trading.mode == "paper":
            self.executor = PaperExecutor(config.trading.initial_balance, self.store)
        else:
            from src.execution.coinbase import CoinbaseExecutor
            self.executor = CoinbaseExecutor(config, self.store)

        self.scanner = MarketScanner(config, self.collector, self.llm, self.store)
        self.analyzer = DeepAnalyzer(
            config, self.collector, self.llm, self.store, self.risk_manager, self.executor,
        )
        self.strategist = PortfolioStrategist(
            config, self.collector, self.llm, self.store, self.risk_manager, self.executor,
        )
        self._running = False
        self._scan_count = 0
        self._last_deep_analysis: dict[str, float] = {}
        self.logger = logging.getLogger("auto-trader")

    async def _run_scan_cycle(self):
        """One full scan cycle: scan all pairs, escalate opportunities."""
        self._scan_count += 1
        self.logger.info("=== Scan cycle #%d ===", self._scan_count)

        # Check stop-losses first
        prices = {}
        for pair in self.config.trading.pairs:
            try:
                ticker = await self.collector.fetch_ticker(pair)
                prices[pair] = ticker["last"]
            except Exception:
                pass

        if prices:
            triggered = await self.executor.check_stop_losses(prices)
            for t in triggered:
                self.logger.warning("Stop triggered: %s %s @ $%.2f", t.side, t.symbol, t.price)
                self.risk_manager.record_loss()

        # Scan all pairs
        results = await self.scanner.scan_all()

        # Escalate opportunities to deep analysis
        now = asyncio.get_event_loop().time()
        cooldown = self.config.schedule.deep_analysis_cooldown

        for pair, scan in results.items():
            if scan.signal != "opportunity":
                continue
            if scan.confidence < self.config.scanner.confidence_threshold:
                self.logger.info(
                    "Skipping %s: confidence %.2f < threshold %.2f",
                    pair, scan.confidence, self.config.scanner.confidence_threshold,
                )
                continue

            last_analysis = self._last_deep_analysis.get(pair, 0)
            if now - last_analysis < cooldown:
                self.logger.info("Skipping %s: deep analysis cooldown", pair)
                continue

            self.logger.info(
                "Escalating %s to deep analysis (confidence=%.2f, direction=%s)",
                pair, scan.confidence, scan.direction,
            )
            self._last_deep_analysis[pair] = now
            await self.analyzer.analyze_and_execute(pair, scan.direction)

        # Log portfolio status
        balance = await self.executor.get_balance()
        daily_pnl = self.store.get_daily_pnl()
        self.logger.info(
            "Portfolio: $%.2f (cash: $%.2f) | Daily P&L: $%.2f | Positions: %d",
            balance["total_value"], balance["cash"], daily_pnl,
            len(balance["positions"]),
        )

    async def _run_portfolio_review(self):
        """Periodic portfolio review with the strategist model."""
        self.logger.info("=== Portfolio Review ===")
        review = await self.strategist.review()
        if review:
            self.logger.info("Market regime: %s", review.market_regime)
            for item in review.action_items:
                self.logger.info("Action: %s", item)

    async def run(self):
        """Main loop: run scan cycles and periodic portfolio reviews."""
        self._running = True
        scan_interval = self.config.schedule.scan_interval_seconds
        review_interval = self.config.schedule.portfolio_review_hours * 3600
        last_review = 0.0

        # Check Ollama health
        if not await self.llm.is_healthy():
            self.logger.error("Ollama is not reachable at %s", self.config.ollama_base_url)
            self.logger.error("Make sure Ollama is running: ollama serve")
            return

        mode = self.config.trading.mode.upper()
        console.print(f"\n[bold green]Auto-Trader started in {mode} mode[/bold green]")
        console.print(f"  Pairs: {', '.join(self.config.trading.pairs)}")
        console.print(f"  Scanner: {self.config.models.scanner.name}")
        console.print(f"  Analyzer: {self.config.models.analyzer.name}")
        console.print(f"  Strategist: {self.config.models.strategist.name}")
        console.print(f"  Scan interval: {scan_interval}s")
        console.print(f"  Dashboard: http://localhost:8080")
        if self.config.trading.mode == "paper":
            console.print(f"  Paper balance: ${self.config.trading.initial_balance:.2f}")
        console.print()

        while self._running:
            try:
                await self._run_scan_cycle()

                # Portfolio review on schedule
                now = asyncio.get_event_loop().time()
                if now - last_review >= review_interval:
                    last_review = now
                    await self._run_portfolio_review()

            except Exception:
                self.logger.exception("Error in scan cycle")

            # Wait for next cycle
            try:
                await asyncio.sleep(scan_interval)
            except asyncio.CancelledError:
                break

        self.logger.info("Auto-Trader shutting down...")

    async def shutdown(self):
        self._running = False
        await self.collector.close()
        await self.llm.close()
        self.store.close()


def main():
    setup_logging()
    config = load_config()

    if config.trading.mode == "live":
        if not config.coinbase_api_key:
            console.print("[bold red]ERROR: COINBASE_API_KEY not set in .env[/bold red]")
            sys.exit(1)
        console.print("[bold red]WARNING: LIVE TRADING MODE[/bold red]")
        console.print("Real money will be used. Press Ctrl+C within 10s to abort.")
        try:
            import time
            time.sleep(10)
        except KeyboardInterrupt:
            console.print("Aborted.")
            sys.exit(0)

    trader = AutoTrader(config)

    # Launch dashboard in a background thread
    import threading
    from src.dashboard.app import run_dashboard
    dash_thread = threading.Thread(target=run_dashboard, daemon=True)
    dash_thread.start()

    loop = asyncio.new_event_loop()

    def handle_signal(sig, frame):
        console.print("\n[yellow]Shutting down...[/yellow]")
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
