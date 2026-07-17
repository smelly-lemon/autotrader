from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ManagedOrder:
    """Tracks an order through its lifecycle."""
    exchange_order_id: str
    symbol: str
    side: str
    order_type: str  # 'market' or 'limit'
    requested_amount: float
    requested_price: float | None
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: float = 0.0
    filled_price: float = 0.0
    fee: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""
    error: str = ""


class OrderManager:
    """Tracks and manages the lifecycle of exchange orders."""

    def __init__(self):
        self.orders: dict[str, ManagedOrder] = {}

    def track(self, order: ManagedOrder) -> ManagedOrder:
        self.orders[order.exchange_order_id] = order
        logger.info(
            "Tracking order %s: %s %s %s @ %s",
            order.exchange_order_id, order.side, order.requested_amount,
            order.symbol, order.requested_price or "market",
        )
        return order

    def update_from_exchange(self, exchange_order: dict) -> ManagedOrder | None:
        """Update a managed order from an exchange order response (ccxt format)."""
        oid = exchange_order.get("id", "")
        if oid not in self.orders:
            return None

        order = self.orders[oid]
        status_map = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
        }

        raw_status = exchange_order.get("status", "")
        order.status = status_map.get(raw_status, order.status)
        order.filled_amount = exchange_order.get("filled", order.filled_amount) or 0
        order.filled_price = exchange_order.get("average", order.filled_price) or 0

        fee_info = exchange_order.get("fee")
        if fee_info and fee_info.get("cost"):
            order.fee = float(fee_info["cost"])

        order.updated_at = datetime.now(timezone.utc).isoformat()

        if order.filled_amount > 0 and order.status != OrderStatus.FILLED:
            order.status = OrderStatus.PARTIALLY_FILLED

        return order

    def get_open_orders(self) -> list[ManagedOrder]:
        return [
            o for o in self.orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
        ]

    def get_order(self, order_id: str) -> ManagedOrder | None:
        return self.orders.get(order_id)

    def calculate_slippage(self, order: ManagedOrder) -> float | None:
        """Return slippage as a percentage. Positive = worse than expected."""
        if not order.requested_price or not order.filled_price:
            return None
        if order.side == "buy":
            return (order.filled_price - order.requested_price) / order.requested_price * 100
        else:
            return (order.requested_price - order.filled_price) / order.requested_price * 100
