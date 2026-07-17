from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

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
    """Simulated trading engine that tracks positions and P&L without real orders."""

    def __init__(self, initial_balance: float, store: TradeStore):
        self.cash = initial_balance
        self.positions: dict[str, PaperPosition] = {}
        self.store = store
        self._initial_balance = initial_balance
        logger.info("Paper executor initialized with $%.2f", initial_balance)

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

        amount = amount_usd / current_price
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)

        trade_id = self.store.log_trade(
            symbol=symbol,
            side="buy",
            price=current_price,
            amount=amount,
            stop_loss=stop_loss,
            take_profit=take_profit,
            model_tier="analyzer",
        )

        self.positions[symbol] = PaperPosition(
            symbol=symbol,
            amount=amount,
            entry_price=current_price,
            trade_id=trade_id,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.cash -= amount_usd
        order_id = f"paper-{uuid.uuid4().hex[:8]}"

        logger.info(
            "PAPER BUY: %s | %.6f @ $%.2f ($%.2f) | SL: $%.2f TP: $%.2f",
            symbol, amount, current_price, amount_usd, stop_loss, take_profit,
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side="buy",
            price=current_price,
            amount=amount,
            cost=amount_usd,
        )

    async def execute_sell(
        self,
        symbol: str,
        amount: float,
        current_price: float,
    ) -> OrderResult:
        pos = self.positions.get(symbol)
        if not pos:
            return OrderResult(
                success=False,
                error=f"No open position in {symbol}",
            )

        sell_amount = min(amount, pos.amount)
        proceeds = sell_amount * current_price
        self.cash += proceeds

        self.store.close_trade(pos.trade_id, current_price)

        if sell_amount >= pos.amount:
            del self.positions[symbol]
        else:
            pos.amount -= sell_amount

        order_id = f"paper-{uuid.uuid4().hex[:8]}"
        pnl = (current_price - pos.entry_price) * sell_amount

        logger.info(
            "PAPER SELL: %s | %.6f @ $%.2f ($%.2f) | PnL: $%.2f",
            symbol, sell_amount, current_price, proceeds, pnl,
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side="sell",
            price=current_price,
            amount=sell_amount,
            cost=proceeds,
        )

    async def get_balance(self) -> dict:
        positions_dict = {}
        unrealized_pnl = 0.0
        for sym, pos in self.positions.items():
            positions_dict[sym] = {
                "amount": pos.amount,
                "entry_price": pos.entry_price,
                "trade_id": pos.trade_id,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            }
        return {
            "cash": self.cash,
            "positions": positions_dict,
            "total_value": self.cash,  # updated by check_stop_losses with live prices
            "initial_balance": self._initial_balance,
        }

    async def check_stop_losses(self, prices: dict[str, float]) -> list[OrderResult]:
        """Check all positions against current prices. Trigger stop-loss or take-profit."""
        triggered: list[OrderResult] = []
        total_value = self.cash

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            price = prices.get(symbol)
            if price is None:
                total_value += pos.amount * pos.entry_price
                continue

            total_value += pos.amount * price

            if pos.stop_loss and price <= pos.stop_loss:
                logger.warning("STOP-LOSS triggered for %s @ $%.2f (SL: $%.2f)", symbol, price, pos.stop_loss)
                result = await self.execute_sell(symbol, pos.amount, price)
                if result.success:
                    self.store.close_trade(pos.trade_id, price, status="stopped")
                    triggered.append(result)

            elif pos.take_profit and price >= pos.take_profit:
                logger.info("TAKE-PROFIT triggered for %s @ $%.2f (TP: $%.2f)", symbol, price, pos.take_profit)
                result = await self.execute_sell(symbol, pos.amount, price)
                triggered.append(result)

        return triggered
