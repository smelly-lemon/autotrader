from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ScanResult(BaseModel):
    signal: Literal["opportunity", "nothing"]
    direction: Literal["long", "short", "none"] = "none"
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return v


class TradeDecision(BaseModel):
    action: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    size_pct: float = Field(ge=0.0, le=1.0, default=0.0)
    stop_loss_pct: float = Field(ge=0.0, le=0.5, default=0.03)
    take_profit_pct: float = Field(ge=0.0, le=1.0, default=0.06)
    reasoning: str = ""
    risk_notes: str = ""

    @field_validator("confidence", "size_pct", "stop_loss_pct", "take_profit_pct", mode="before")
    @classmethod
    def clamp_floats(cls, v):
        if isinstance(v, (int, float)):
            return max(0.0, float(v))
        return v


class PortfolioReview(BaseModel):
    market_regime: Literal[
        "trending_up", "trending_down", "ranging", "volatile", "uncertain"
    ] = "uncertain"
    risk_adjustment: float = Field(ge=0.1, le=2.0, default=1.0)
    pairs_to_watch: list[str] = Field(default_factory=list)
    pairs_to_avoid: list[str] = Field(default_factory=list)
    max_positions_override: int = Field(ge=1, le=5, default=3)
    reasoning: str = ""
    action_items: list[str] = Field(default_factory=list)


def parse_scan_result(raw: dict) -> ScanResult | None:
    if raw.get("parse_error"):
        logger.warning("Cannot parse scan result: model returned unparseable output")
        return None
    try:
        return ScanResult(**raw)
    except Exception:
        logger.exception("Failed to validate scan result: %s", raw)
        return None


def parse_trade_decision(raw: dict) -> TradeDecision | None:
    if raw.get("parse_error"):
        logger.warning("Cannot parse trade decision: model returned unparseable output")
        return None
    try:
        return TradeDecision(**raw)
    except Exception:
        logger.exception("Failed to validate trade decision: %s", raw)
        return None


def parse_portfolio_review(raw: dict) -> PortfolioReview | None:
    if raw.get("parse_error"):
        logger.warning("Cannot parse portfolio review: model returned unparseable output")
        return None
    try:
        return PortfolioReview(**raw)
    except Exception:
        logger.exception("Failed to validate portfolio review: %s", raw)
        return None
