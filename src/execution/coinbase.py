from __future__ import annotations

import logging
from datetime import datetime, timezone

import ccxt.async_support as ccxt

from src.config import AppConfig
from src.data.store import TradeStore
from src.execution.executor import BaseExecutor, OrderResult
from src.execution.orders import ManagedOrder, OrderManager, OrderStatus

logger = logging.getLogger(__name__)


class CoinbaseExecutor(BaseExecutor):
    """Live trading executor for Coinbase Advanced Trade via ccxt."""

    def __init__(self, config: AppConfig, store: TradeStore):
        self.config = config
        self.store = store
        self.order_manager = OrderManager()
        self._exchange: ccxt.Exchange | None = None
        self._stop_loss_orders: dict[str, dict] = {}

    async def _get_exchange(self) -> ccxt.Exchange:
        if self._exchange is None:
            self._exchange = ccxt.coinbase({
                "apiKey": self.config.coinbase_api_key,
                "secret": self.config.coinbase_api_secret,
                "enableRateLimit": True,
            })
        return self._exchange

    async def execute_buy(
        self,
        symbol: str,
        amount_usd: float,
        current_price: float,
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
    ) -> OrderResult:
        exchange = await self._get_exchange()

        try:
            amount = amount_usd / current_price
            order = await exchange.create_market_buy_order(symbol, amount)

            managed = ManagedOrder(
                exchange_order_id=order["id"],
                symbol=symbol,
                side="buy",
                order_type="market",
                requested_amount=amount,
                requested_price=current_price,
            )
            self.order_manager.track(managed)
            self.order_manager.update_from_exchange(order)

            fill_price = order.get("average") or order.get("price") or current_price
            fill_amount = order.get("filled") or amount
            cost = fill_price * fill_amount

            # Calculate and log slippage
            slippage = self.order_manager.calculate_slippage(managed)
            if slippage is not None and abs(slippage) > 0.1:
                logger.warning("Slippage on %s buy: %.3f%%", symbol, slippage)

            trade_id = self.store.log_trade(
                symbol=symbol,
                side="buy",
                price=fill_price,
                amount=fill_amount,
                stop_loss=fill_price * (1 - stop_loss_pct),
                take_profit=fill_price * (1 + take_profit_pct),
                model_tier="analyzer",
                metadata={
                    "exchange_order_id": order["id"],
                    "slippage_pct": slippage,
                    "fee": managed.fee,
                },
            )

            # Track stop-loss for this position
            self._stop_loss_orders[symbol] = {
                "trade_id": trade_id,
                "entry_price": fill_price,
                "amount": fill_amount,
                "stop_loss": fill_price * (1 - stop_loss_pct),
                "take_profit": fill_price * (1 + take_profit_pct),
            }

            logger.info(
                "LIVE BUY: %s | %.6f @ $%.2f ($%.2f) | fee: $%.4f",
                symbol, fill_amount, fill_price, cost, managed.fee,
            )

            return OrderResult(
                success=True,
                order_id=order["id"],
                symbol=symbol,
                side="buy",
                price=fill_price,
                amount=fill_amount,
                cost=cost,
            )

        except Exception as e:
            logger.exception("Live buy failed for %s", symbol)
            return OrderResult(success=False, error=str(e))

    async def execute_sell(
        self,
        symbol: str,
        amount: float,
        current_price: float,
    ) -> OrderResult:
        exchange = await self._get_exchange()

        try:
            order = await exchange.create_market_sell_order(symbol, amount)

            managed = ManagedOrder(
                exchange_order_id=order["id"],
                symbol=symbol,
                side="sell",
                order_type="market",
                requested_amount=amount,
                requested_price=current_price,
            )
            self.order_manager.track(managed)
            self.order_manager.update_from_exchange(order)

            fill_price = order.get("average") or order.get("price") or current_price
            fill_amount = order.get("filled") or amount

            slippage = self.order_manager.calculate_slippage(managed)

            # Close the trade in the store
            sl_data = self._stop_loss_orders.pop(symbol, {})
            if sl_data.get("trade_id"):
                self.store.close_trade(sl_data["trade_id"], fill_price)

            logger.info(
                "LIVE SELL: %s | %.6f @ $%.2f | fee: $%.4f",
                symbol, fill_amount, fill_price, managed.fee,
            )

            return OrderResult(
                success=True,
                order_id=order["id"],
                symbol=symbol,
                side="sell",
                price=fill_price,
                amount=fill_amount,
                cost=fill_price * fill_amount,
            )

        except Exception as e:
            logger.exception("Live sell failed for %s", symbol)
            return OrderResult(success=False, error=str(e))

    async def get_balance(self) -> dict:
        exchange = await self._get_exchange()
        try:
            balance = await exchange.fetch_balance()
            total_usd = float(balance.get("total", {}).get("USD", 0) or 0)

            positions = {}
            for symbol, sl_data in self._stop_loss_orders.items():
                positions[symbol] = {
                    "amount": sl_data["amount"],
                    "entry_price": sl_data["entry_price"],
                    "trade_id": sl_data["trade_id"],
                    "stop_loss": sl_data["stop_loss"],
                    "take_profit": sl_data["take_profit"],
                }

            # Estimate total value including positions
            total_value = total_usd
            for sym, pos in positions.items():
                total_value += pos["amount"] * pos["entry_price"]

            return {
                "cash": total_usd,
                "positions": positions,
                "total_value": total_value,
                "initial_balance": 0,
            }
        except Exception:
            logger.exception("Failed to fetch balance")
            return {"cash": 0, "positions": {}, "total_value": 0, "initial_balance": 0}

    async def check_stop_losses(self, prices: dict[str, float]) -> list[OrderResult]:
        triggered: list[OrderResult] = []

        for symbol in list(self._stop_loss_orders.keys()):
            sl_data = self._stop_loss_orders[symbol]
            price = prices.get(symbol)
            if price is None:
                continue

            if price <= sl_data["stop_loss"]:
                logger.warning(
                    "STOP-LOSS triggered for %s @ $%.2f (SL: $%.2f)",
                    symbol, price, sl_data["stop_loss"],
                )
                result = await self.execute_sell(symbol, sl_data["amount"], price)
                if result.success:
                    self.store.close_trade(sl_data["trade_id"], price, status="stopped")
                    triggered.append(result)

            elif price >= sl_data["take_profit"]:
                logger.info(
                    "TAKE-PROFIT triggered for %s @ $%.2f (TP: $%.2f)",
                    symbol, price, sl_data["take_profit"],
                )
                result = await self.execute_sell(symbol, sl_data["amount"], price)
                if result.success:
                    triggered.append(result)

        return triggered

    async def close(self):
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
