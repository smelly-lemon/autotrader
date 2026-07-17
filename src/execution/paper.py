from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from src.data.store import TradeStore
from src.execution.executor import BaseExecutor, OrderResult

logger = logging.getLogger(__name__)


@dataclass
class PaperPosition:
    symbol: str
    amount: float
    entry_price: float
    trade_id: int
    stop_loss: float | None = None
    take_profit: float | None = None


class PaperExecutor(BaseExecutor):
    """Simulated trading engine that tracks positions and P&L without real orders.

    Models Coinbase taker fees and slippage so paper results approximate what a
    real account would pay. Costs are folded into the effective entry/exit
    prices logged to the store, so recorded pnl is net of fees + slippage.
    """

    def __init__(
        self,
        initial_balance: float,
        store: TradeStore,
        fee_pct: float = 0.012,       # Coinbase starter-tier taker (<$1K/mo volume)
        slippage_pct: float = 0.0005,
    ):
        self.cash = initial_balance
        self.positions: dict[str, PaperPosition] = {}
        self.store = store
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self._initial_balance = initial_balance
        self._last_prices: dict[str, float] = {}
        logger.info(
            "Paper executor initialized with $%.2f (fee %.2f%%, slippage %.3f%%)",
            initial_balance, fee_pct * 100, slippage_pct * 100,
        )

    def restore_state(self) -> int:
        """Rebuild positions and cash from the store after a restart.

        cash = initial balance + realized pnl - capital tied up in open trades.
        Returns the number of restored positions.
        """
        realized = self.store.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS p FROM trades WHERE status != 'open'"
        ).fetchone()["p"]

        open_cost = 0.0
        self.positions.clear()
        for t in self.store.get_open_trades():
            self.positions[t["symbol"]] = PaperPosition(
                symbol=t["symbol"],
                amount=t["amount"],
                entry_price=t["price"],
                trade_id=t["id"],
                stop_loss=t["stop_loss"],
                take_profit=t["take_profit"],
            )
            open_cost += t["cost"]

        self.cash = self._initial_balance + float(realized) - open_cost
        if self.positions or realized:
            logger.info(
                "Restored paper state: cash $%.2f, realized pnl $%+.2f, %d open positions",
                self.cash, realized, len(self.positions),
            )
        return len(self.positions)

    async def execute_buy(
        self,
        symbol: str,
        amount_usd: float,
        current_price: float,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
    ) -> OrderResult:
        if amount_usd > self.cash:
            return OrderResult(
                success=False,
                error=f"Insufficient cash: ${self.cash:.2f} < ${amount_usd:.2f}",
            )

        # Fill slips against us; fee inflates the effective entry price so that
        # stored pnl is automatically net of costs.
        fill_price = current_price * (1 + self.slippage_pct)
        effective_entry = fill_price * (1 + self.fee_pct)
        amount = amount_usd / effective_entry
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)

        trade_id = self.store.log_trade(
            symbol=symbol,
            side="buy",
            price=effective_entry,
            amount=amount,
            stop_loss=stop_loss,
            take_profit=take_profit,
            model_tier="analyzer",
        )

        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            amount=amount,
            entry_price=effective_entry,
            trade_id=trade_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.cash -= amount_usd
        order_id = f"paper-{uuid.uuid4().hex[:8]}"

        logger.info(
            "PAPER BUY: %s | %.6f @ $%.2f eff. ($%.2f incl. costs) | SL: $%.2f TP: $%.2f",
            symbol, amount, effective_entry, amount_usd, stop_loss, take_profit,
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side="buy",
            price=effective_entry,
            amount=amount,
            cost=amount_usd,
        )

    async def execute_sell(
        self,
        symbol: str,
        amount: float,
        current_price: float,
        status: str = "closed",
    ) -> OrderResult:
        pos = self.positions.get(symbol)
        if not pos:
            return OrderResult(
                success=False,
                error=f"No open position in {symbol}",
            )

        fill_price = current_price * (1 - self.slippage_pct)
        effective_exit = fill_price * (1 - self.fee_pct)

        sell_amount = min(amount, pos.amount)
        proceeds = sell_amount * effective_exit
        self.cash += proceeds

        self.store.close_trade(pos.trade_id, effective_exit, status=status)

        if sell_amount >= pos.amount:
            del self.positions[symbol]
        else:
            pos.amount -= sell_amount

        order_id = f"paper-{uuid.uuid4().hex[:8]}"
        pnl = (effective_exit - pos.entry_price) * sell_amount

        logger.info(
            "PAPER SELL: %s | %.6f @ $%.2f eff. ($%.2f) | net PnL: $%.2f",
            symbol, sell_amount, effective_exit, proceeds, pnl,
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side="sell",
            price=effective_exit,
            amount=sell_amount,
            cost=proceeds,
        )

    async def get_balance(self) -> dict:
        positions_dict = {}
        total_value = self.cash
        for sym, pos in self.positions.items():
            mark = self._last_prices.get(sym, pos.entry_price)
            total_value += pos.amount * mark
            positions_dict[sym] = {
                "amount": pos.amount,
                "entry_price": pos.entry_price,
                "mark_price": mark,
                "trade_id": pos.trade_id,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            }
        return {
            "cash": self.cash,
            "positions": positions_dict,
            "total_value": total_value,
            "initial_balance": self._initial_balance,
        }

    async def check_stop_losses(self, prices: dict[str, float]) -> list[OrderResult]:
        """Check all positions against current prices. Trigger stop-loss or take-profit."""
        triggered: list[OrderResult] = []
        self._last_prices.update(prices)

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            price = prices.get(symbol)
            if price is None:
                continue

            if pos.stop_loss and price <= pos.stop_loss:
                logger.warning("STOP-LOSS triggered for %s @ $%.2f (SL: $%.2f)", symbol, price, pos.stop_loss)
                result = await self.execute_sell(symbol, pos.amount, price, status="stopped")
                if result.success:
                    triggered.append(result)

            elif pos.take_profit and price >= pos.take_profit:
                logger.info("TAKE-PROFIT triggered for %s @ $%.2f (TP: $%.2f)", symbol, price, pos.take_profit)
                result = await self.execute_sell(symbol, pos.amount, price)
                if result.success:
                    triggered.append(result)

        return triggered
