from __future__ import annotations

import logging

from src.config import AppConfig
from src.data.collector import MarketDataCollector
from src.data.indicators import async_multi_timeframe_summary
from src.data.store import TradeStore
from src.execution.executor import BaseExecutor
from src.llm.client import OllamaClient
from src.llm.parser import TradeDecision, parse_trade_decision
from src.llm.prompts import ANALYZER_SYSTEM, build_analyzer_prompt
from src.risk.manager import RiskManager, RiskVerdict

logger = logging.getLogger(__name__)


class DeepAnalyzer:
    """Tier 2: Deep analysis triggered when the scanner detects an opportunity."""

    def __init__(
        self,
        config: AppConfig,
        collector: MarketDataCollector,
        llm: OllamaClient,
        store: TradeStore,
        risk_manager: RiskManager,
        executor: BaseExecutor,
    ):
        self.config = config
        self.collector = collector
        self.llm = llm
        self.store = store
        self.risk_manager = risk_manager
        self.executor = executor

    async def analyze_and_execute(self, symbol: str, direction: str) -> TradeDecision | None:
        """Run deep multi-timeframe analysis and execute if approved by risk manager."""

        # Gather multi-timeframe data
        try:
            multi_tf = await async_multi_timeframe_summary(self.collector, symbol)
        except Exception:
            logger.exception("Failed multi-TF analysis for %s", symbol)
            return None

        # Gather context
        balance = await self.executor.get_balance()
        recent_trades = self.store.get_recent_trades(limit=10)
        portfolio = {
            "cash": balance["cash"],
            "total_value": balance["total_value"],
            "positions": balance["positions"],
        }

        try:
            order_book = await self.collector.fetch_order_book(symbol)
        except Exception:
            order_book = None
            logger.warning("Could not fetch order book for %s", symbol)

        prompt = build_analyzer_prompt(
            symbol=symbol,
            multi_tf_data=multi_tf,
            portfolio=portfolio,
            recent_trades=recent_trades,
            order_book=order_book,
        )

        # Call the deep analysis model
        try:
            raw = await self.llm.generate_parsed(
                prompt=prompt,
                model_config=self.config.models.analyzer,
                system=ANALYZER_SYSTEM,
            )
        except Exception:
            logger.exception("LLM analysis failed for %s", symbol)
            return None

        decision = parse_trade_decision(raw)
        if decision is None:
            return None

        logger.info(
            "Deep analysis %s: action=%s confidence=%.2f size=%.2f%%",
            symbol, decision.action, decision.confidence, decision.size_pct * 100,
        )

        # Run through risk manager
        current_price = multi_tf.get("5m", {}).get("price", 0)
        if not current_price:
            try:
                ticker = await self.collector.fetch_ticker(symbol)
                current_price = ticker.get("last", 0)
            except Exception:
                logger.error("Cannot determine price for %s", symbol)
                return decision

        verdict: RiskVerdict = self.risk_manager.evaluate(
            decision=decision,
            symbol=symbol,
            current_price=current_price,
            portfolio_value=balance["total_value"],
            cash_available=balance["cash"],
        )

        for w in verdict.warnings:
            logger.warning("Risk warning for %s: %s", symbol, w)

        # Log decision
        self.store.log_decision(
            symbol=symbol,
            model_tier="analyzer",
            model_name=self.config.models.analyzer.name,
            action=decision.action,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            raw_output=str(raw),
            was_executed=verdict.approved,
            risk_vetoed=not verdict.approved,
            veto_reason="; ".join(verdict.veto_reasons),
        )

        if not verdict.approved:
            logger.info("Trade VETOED for %s: %s", symbol, "; ".join(verdict.veto_reasons))
            return decision

        # Execute the trade
        trade_value = balance["cash"] * verdict.adjusted_size_pct
        if decision.action == "buy":
            result = await self.executor.execute_buy(
                symbol=symbol,
                amount_usd=trade_value,
                current_price=current_price,
                stop_loss_pct=verdict.adjusted_stop_loss_pct,
                take_profit_pct=decision.take_profit_pct,
            )
        elif decision.action == "sell":
            pos = balance["positions"].get(symbol, {})
            if pos:
                result = await self.executor.execute_sell(
                    symbol=symbol,
                    amount=pos["amount"],
                    current_price=current_price,
                )
            else:
                logger.info("No position to sell for %s", symbol)
                return decision
        else:
            return decision

        if result.success:
            logger.info("Trade executed: %s %s @ $%.2f", result.side, result.symbol, result.price)
        else:
            logger.error("Trade failed: %s", result.error)

        return decision
