from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    price: float = 0.0
    amount: float = 0.0
    cost: float = 0.0
    error: str = ""


class BaseExecutor(ABC):
    """Abstract trade executor -- implemented by paper and live engines."""

    @abstractmethod
    async def execute_buy(
        self,
        symbol: str,
        amount_usd: float,
        current_price: float,
    ) -> OrderResult:
        ...

    @abstractmethod
    async def execute_sell(
        self,
        symbol: str,
        amount: float,
        current_price: float,
    ) -> OrderResult:
        ...

    @abstractmethod
    async def get_balance(self) -> dict:
        """Return {'total_value': float, 'cash': float, 'positions': dict}"""
        ...

    @abstractmethod
    async def check_stop_losses(self, prices: dict[str, float]) -> list[OrderResult]:
        """Check all open positions against current prices for stop-loss triggers."""
        ...
