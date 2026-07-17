from __future__ import annotations

import logging

from src.config import AppConfig
from src.data.indicators import compute_indicators, summarize_indicators
from src.data.collector import MarketDataCollector
from src.data.store import TradeStore
from src.llm.client import OllamaClient
from src.llm.prompts import SCANNER_SYSTEM, build_scanner_prompt
from src.llm.parser import ScanResult, parse_scan_result

logger = logging.getLogger(__name__)


class MarketScanner:
    """Tier 1: Fast scan of all trading pairs using the small model."""

    def __init__(
        self,
        config: AppConfig,
        collector: MarketDataCollector,
        llm: OllamaClient,
        store: TradeStore,
    ):
        self.config = config
        self.collector = collector
        self.llm = llm
        self.store = store

    async def scan_pair(self, symbol: str) -> ScanResult | None:
        try:
            df = await self.collector.fetch_ohlcv(symbol, timeframe="5m", limit=100)
            df = compute_indicators(df)
            indicators = summarize_indicators(df)
        except Exception:
            logger.exception("Failed to get data for %s", symbol)
            return None

        prompt = build_scanner_prompt(symbol, indicators)

        try:
            raw = await self.llm.generate_parsed(
                prompt=prompt,
                model_config=self.config.models.scanner,
                system=SCANNER_SYSTEM,
            )
        except Exception:
            logger.exception("LLM scan failed for %s", symbol)
            return None

        result = parse_scan_result(raw)
        if result is None:
            return None

        self.store.log_decision(
            symbol=symbol,
            model_tier="scanner",
            model_name=self.config.models.scanner.name,
            action=result.signal,
            confidence=result.confidence,
            reasoning=result.reasoning,
            raw_output=str(raw),
            indicators=indicators,
        )

        logger.info(
            "Scan %s: signal=%s direction=%s confidence=%.2f",
            symbol, result.signal, result.direction, result.confidence,
        )

        return result

    async def scan_all(self) -> dict[str, ScanResult]:
        results = {}
        for pair in self.config.trading.pairs:
            result = await self.scan_pair(pair)
            if result:
                results[pair] = result
        return results
