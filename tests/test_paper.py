import pytest
import pytest_asyncio

from src.data.store import TradeStore
from src.execution.paper import PaperExecutor


@pytest.fixture
def store(tmp_path):
    return TradeStore(db_path=tmp_path / "test.db")


@pytest.fixture
def executor(store):
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
async def test_trade_logged_to_store(executor, store):
    await executor.execute_buy("BTC/USD", 100.0, 50000.0)
    trades = store.get_open_trades()
    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTC/USD"
    assert trades[0]["side"] == "buy"
