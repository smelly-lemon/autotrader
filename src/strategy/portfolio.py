from __future__ import annotations

import logging

from src.config import AppConfig
from src.data.collector import MarketDataCollector
from src.data.indicators import compute_indicators, summarize_indicators
from src.data.store import TradeStore
from src.execution.executor import BaseExecutor
from src.llm.client import OllamaClient
from src.llm.parser import PortfolioReview, parse_portfolio_review
from src.llm.prompts import STRATEGIST_SYSTEM, build_strategist_prompt
from src.risk.manager import RiskManager

logger = logging.getLogger(__name__)


class PortfolioStrategist:
    """Tier 3: Periodic portfolio review using the largest model."""

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

    async def review(self) -> PortfolioReview | None:
        balance = await self.executor.get_balance()
        recent_trades = self.store.get_recent_trades(limit=20)
        daily_pnl = self.store.get_daily_pnl()

        portfolio = {
            "cash": balance["cash"],
            "total_value": balance["total_value"],
            "positions": balance["positions"],
            "initial_balance": balance.get("initial_balance", 0),
            "total_return_pct": (
                (balance["total_value"] - balance.get("initial_balance", balance["total_value"]))
                / balance.get("initial_balance", 1) * 100
            ),
        }

        # Gather 4h summaries for all pairs
        market_summaries = {}
        for pair in self.config.trading.pairs:
            try:
                df = await self.collector.fetch_ohlcv(pair, timeframe="4h", limit=50)
                df = compute_indicators(df)
                market_summaries[pair] = summarize_indicators(df)
            except Exception:
                market_summaries[pair] = {"error": "unavailable"}

        prompt = build_strategist_prompt(
            portfolio=portfolio,
            recent_trades=recent_trades,
            daily_pnl=daily_pnl,
            market_summaries=market_summaries,
        )

        try:
            raw = await self.llm.generate_parsed(
                prompt=prompt,
                model_config=self.config.models.strategist,
                system=STRATEGIST_SYSTEM,
            )
        except Exception:
            logger.exception("Portfolio review LLM call failed")
            return None

        review = parse_portfolio_review(raw)
        if review is None:
            return None

        logger.info(
            "Portfolio review: regime=%s, risk_adj=%.2f, watch=%s, avoid=%s",
            review.market_regime,
            review.risk_adjustment,
            review.pairs_to_watch,
            review.pairs_to_avoid,
        )

        # Apply strategic adjustments
        self.risk_manager.set_risk_multiplier(review.risk_adjustment)

        # Log snapshot
        self.store.log_portfolio_snapshot(
            total_value=balance["total_value"],
            cash=balance["cash"],
            positions=balance["positions"],
            daily_pnl=daily_pnl,
            daily_pnl_pct=(daily_pnl / balance["total_value"] * 100) if balance["total_value"] > 0 else 0,
        )

        self.store.log_decision(
            symbol="PORTFOLIO",
            model_tier="strategist",
            model_name=self.config.models.strategist.name,
            action=review.market_regime,
            confidence=0.0,
            reasoning=review.reasoning,
            raw_output=str(raw),
            was_executed=True,
        )

        return review
