import pytest

from src.data.store import TradeStore
from src.execution.paper import PaperExecutor


@pytest.fixture
def store(tmp_path):
    return TradeStore(db_path=tmp_path / "test.db")


@pytest.fixture
def executor(store):
    """Zero-cost executor for pure position/cash arithmetic tests."""
    return PaperExecutor(initial_balance=1000.0, store=store, fee_pct=0.0, slippage_pct=0.0)


@pytest.fixture
def real_cost_executor(store):
    """Executor with default (starter-tier) fees and slippage."""
    return PaperExecutor(initial_balance=1000.0, store=store)


@pytest.mark.asyncio
async def test_buy_reduces_cash(executor):
    result = await executor.execute_buy("BTC/USD", 100.0, 50000.0)
    assert result.success
    assert executor.cash == pytest.approx(900.0)
    assert "BTC/USD" in executor.positions


@pytest.mark.asyncio
async def test_buy_insufficient_funds(executor):
    result = await executor.execute_buy("BTC/USD", 2000.0, 50000.0)
    assert not result.success
    assert "Insufficient" in result.error


@pytest.mark.asyncio
async def test_sell_returns_cash(executor):
    await executor.execute_buy("BTC/USD", 100.0, 50000.0)
    pos = executor.positions["BTC/USD"]
    result = await executor.execute_sell("BTC/USD", pos.amount, 55000.0)
    assert result.success
    assert executor.cash > 900.0
    assert "BTC/USD" not in executor.positions


@pytest.mark.asyncio
async def test_sell_no_position(executor):
    result = await executor.execute_sell("BTC/USD", 1.0, 50000.0)
    assert not result.success
    assert "No open position" in result.error


@pytest.mark.asyncio
async def test_stop_loss_triggers(executor):
    await executor.execute_buy("BTC/USD", 100.0, 50000.0, stop_loss_pct=0.03)
    stop_price = 50000.0 * 0.96  # below 3% stop
    triggered = await executor.check_stop_losses({"BTC/USD": stop_price})
    assert len(triggered) == 1
    assert "BTC/USD" not in executor.positions


@pytest.mark.asyncio
async def test_take_profit_triggers(executor):
    await executor.execute_buy("BTC/USD", 100.0, 50000.0, take_profit_pct=0.06)
    tp_price = 50000.0 * 1.07  # above 6% take-profit
    triggered = await executor.check_stop_losses({"BTC/USD": tp_price})
    assert len(triggered) == 1
    assert "BTC/USD" not in executor.positions
    assert executor.cash > 1000.0  # made profit


@pytest.mark.asyncio
async def test_get_balance(executor):
    balance = await executor.get_balance()
    assert balance["cash"] == 1000.0
    assert balance["total_value"] == 1000.0
    assert balance["positions"] == {}


@pytest.mark.asyncio
async def test_total_value_includes_positions(executor):
    await executor.execute_buy("BTC/USD", 100.0, 50000.0)
    balance = await executor.get_balance()
    # 900 cash + 100 position (marked at entry) = 1000
    assert balance["total_value"] == pytest.approx(1000.0)

    await executor.check_stop_losses({"BTC/USD": 51000.0})  # updates marks, no trigger
    balance = await executor.get_balance()
    assert balance["total_value"] == pytest.approx(900.0 + 100.0 * 51000.0 / 50000.0)


@pytest.mark.asyncio
async def test_trade_logged_to_store(executor, store):
    await executor.execute_buy("BTC/USD", 100.0, 50000.0)
    trades = store.get_open_trades()
    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTC/USD"
    assert trades[0]["side"] == "buy"


@pytest.mark.asyncio
async def test_fees_and_slippage_reduce_roundtrip(real_cost_executor, store):
    """A flat round-trip must lose ~2x(fee+slippage), and stored pnl reflects it."""
    await real_cost_executor.execute_buy("BTC/USD", 100.0, 50000.0)
    pos = real_cost_executor.positions["BTC/USD"]
    await real_cost_executor.execute_sell("BTC/USD", pos.amount, 50000.0)

    loss = 1000.0 - real_cost_executor.cash
    expected = 100.0 * (0.012 + 0.0005) * 2
    assert loss == pytest.approx(expected, rel=0.05)

    trade = store.get_recent_trades(1)[0]
    assert trade["status"] == "closed"
    assert trade["pnl"] == pytest.approx(-loss, rel=0.05)


@pytest.mark.asyncio
async def test_restore_state_from_store(store):
    ex1 = PaperExecutor(initial_balance=1000.0, store=store, fee_pct=0.0, slippage_pct=0.0)
    await ex1.execute_buy("BTC/USD", 100.0, 50000.0)
    await ex1.execute_buy("ETH/USD", 50.0, 2000.0)
    pos = ex1.positions["BTC/USD"]
    await ex1.execute_sell("BTC/USD", pos.amount, 55000.0)  # +$10 realized

    ex2 = PaperExecutor(initial_balance=1000.0, store=store, fee_pct=0.0, slippage_pct=0.0)
    restored = ex2.restore_state()
    assert restored == 1
    assert "ETH/USD" in ex2.positions
    assert ex2.positions["ETH/USD"].amount == pytest.approx(ex1.positions["ETH/USD"].amount)
    assert ex2.cash == pytest.approx(ex1.cash)
