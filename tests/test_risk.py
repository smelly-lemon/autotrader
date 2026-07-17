import tempfile
from pathlib import Path

import pytest

from src.config import RiskConfig
from src.data.store import TradeStore
from src.llm.parser import TradeDecision
from src.risk.limits import HARD_LIMITS
from src.risk.manager import RiskManager


@pytest.fixture
def store(tmp_path):
    return TradeStore(db_path=tmp_path / "test.db")


@pytest.fixture
def risk(store):
    config = RiskConfig()
    return RiskManager(config, store)


def test_hold_is_rejected(risk):
    decision = TradeDecision(action="hold", confidence=0.5)
    verdict = risk.evaluate(decision, "BTC/USD", 50000, 1000, 1000)
    assert not verdict.approved
    assert "hold" in verdict.veto_reasons


def test_buy_within_limits(risk):
    decision = TradeDecision(
        action="buy", confidence=0.8, size_pct=0.10,
        stop_loss_pct=0.03, take_profit_pct=0.06,
    )
    verdict = risk.evaluate(decision, "BTC/USD", 50000, 1000, 1000)
    assert verdict.approved
    assert verdict.adjusted_size_pct > 0


def test_position_size_capped(risk):
    decision = TradeDecision(
        action="buy", confidence=0.9, size_pct=0.50,
        stop_loss_pct=0.03, take_profit_pct=0.06,
    )
    verdict = risk.evaluate(decision, "BTC/USD", 50000, 1000, 1000)
    assert verdict.approved
    assert verdict.adjusted_size_pct <= HARD_LIMITS.ABSOLUTE_MAX_POSITION_PCT


def test_max_open_positions(risk, store):
    for i in range(3):
        store.log_trade(f"PAIR{i}/USD", "buy", 100, 1.0)

    decision = TradeDecision(
        action="buy", confidence=0.8, size_pct=0.10,
        stop_loss_pct=0.03, take_profit_pct=0.06,
    )
    verdict = risk.evaluate(decision, "NEW/USD", 100, 1000, 1000)
    assert not verdict.approved
    assert any("Max open positions" in r for r in verdict.veto_reasons)


def test_duplicate_symbol_rejected(risk, store):
    store.log_trade("BTC/USD", "buy", 50000, 0.01)

    decision = TradeDecision(
        action="buy", confidence=0.8, size_pct=0.10,
        stop_loss_pct=0.03, take_profit_pct=0.06,
    )
    verdict = risk.evaluate(decision, "BTC/USD", 50000, 1000, 1000)
    assert not verdict.approved
    assert any("Already have" in r for r in verdict.veto_reasons)


def test_daily_drawdown_blocks(risk, store):
    trade_id = store.log_trade("BTC/USD", "buy", 50000, 0.01)
    store.close_trade(trade_id, 47000)

    decision = TradeDecision(
        action="buy", confidence=0.9, size_pct=0.10,
        stop_loss_pct=0.03, take_profit_pct=0.06,
    )
    verdict = risk.evaluate(decision, "ETH/USD", 3000, 100, 100)
    assert not verdict.approved
    assert any("drawdown" in r.lower() for r in verdict.veto_reasons)


def test_cooldown_after_loss(risk):
    risk.record_loss()

    decision = TradeDecision(
        action="buy", confidence=0.9, size_pct=0.10,
        stop_loss_pct=0.03, take_profit_pct=0.06,
    )
    verdict = risk.evaluate(decision, "BTC/USD", 50000, 1000, 1000)
    assert not verdict.approved
    assert any("cooldown" in r.lower() for r in verdict.veto_reasons)


def test_order_too_small(risk):
    decision = TradeDecision(
        action="buy", confidence=0.8, size_pct=0.001,
        stop_loss_pct=0.03, take_profit_pct=0.06,
    )
    verdict = risk.evaluate(decision, "BTC/USD", 50000, 1000, 0.50)
    assert not verdict.approved
    assert any("too small" in r.lower() for r in verdict.veto_reasons)


def test_risk_multiplier(risk):
    risk.set_risk_multiplier(0.5)

    decision = TradeDecision(
        action="buy", confidence=0.8, size_pct=0.15,
        stop_loss_pct=0.03, take_profit_pct=0.06,
    )
    verdict = risk.evaluate(decision, "BTC/USD", 50000, 1000, 1000)
    assert verdict.approved
    assert verdict.adjusted_size_pct <= 0.15 * 0.5 + 0.001
