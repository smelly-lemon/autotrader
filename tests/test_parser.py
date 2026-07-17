import pytest

from src.llm.parser import (
    ScanResult,
    TradeDecision,
    PortfolioReview,
    parse_scan_result,
    parse_trade_decision,
    parse_portfolio_review,
)


def test_parse_valid_scan():
    raw = {
        "signal": "opportunity",
        "direction": "long",
        "confidence": 0.85,
        "reasoning": "EMA cross bullish, RSI confirming",
    }
    result = parse_scan_result(raw)
    assert result is not None
    assert result.signal == "opportunity"
    assert result.direction == "long"
    assert result.confidence == 0.85


def test_parse_scan_clamps_confidence():
    raw = {"signal": "nothing", "direction": "none", "confidence": 1.5, "reasoning": ""}
    result = parse_scan_result(raw)
    assert result is not None
    assert result.confidence == 1.0


def test_parse_scan_rejects_garbage():
    result = parse_scan_result({"parse_error": True, "raw": "not json"})
    assert result is None


def test_parse_scan_rejects_invalid_signal():
    raw = {"signal": "maybe", "direction": "none", "confidence": 0.5}
    result = parse_scan_result(raw)
    assert result is None


def test_parse_valid_trade_decision():
    raw = {
        "action": "buy",
        "confidence": 0.75,
        "size_pct": 0.10,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "reasoning": "Strong confluence",
        "risk_notes": "None",
    }
    result = parse_trade_decision(raw)
    assert result is not None
    assert result.action == "buy"
    assert result.size_pct == 0.10


def test_parse_trade_decision_hold():
    raw = {"action": "hold", "confidence": 0.3, "reasoning": "No clear signal"}
    result = parse_trade_decision(raw)
    assert result is not None
    assert result.action == "hold"


def test_parse_valid_portfolio_review():
    raw = {
        "market_regime": "trending_up",
        "risk_adjustment": 1.2,
        "pairs_to_watch": ["BTC/USD", "ETH/USD"],
        "pairs_to_avoid": ["SOL/USD"],
        "max_positions_override": 3,
        "reasoning": "Bull market confirmed",
        "action_items": ["Increase BTC exposure"],
    }
    result = parse_portfolio_review(raw)
    assert result is not None
    assert result.market_regime == "trending_up"
    assert result.risk_adjustment == 1.2
    assert "BTC/USD" in result.pairs_to_watch


def test_parse_portfolio_review_clamps_risk():
    raw = {
        "market_regime": "volatile",
        "risk_adjustment": 5.0,
        "reasoning": "test",
    }
    result = parse_portfolio_review(raw)
    assert result is None  # 5.0 exceeds max of 2.0
