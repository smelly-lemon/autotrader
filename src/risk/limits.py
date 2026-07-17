from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardLimits:
    """Absolute limits that can never be overridden by any model output.

    These are safety rails -- even if config or a strategist model
    suggests looser limits, these ceilings are enforced.
    """
    ABSOLUTE_MAX_POSITION_PCT: float = 0.25
    ABSOLUTE_MAX_OPEN_POSITIONS: int = 5
    ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT: float = 0.10
    ABSOLUTE_MIN_STOP_LOSS_PCT: float = 0.005
    ABSOLUTE_MAX_SINGLE_TRADE_LOSS_PCT: float = 0.05
    MIN_ORDER_SIZE_USD: float = 1.0
    MIN_COOLDOWN_SECONDS: int = 60


HARD_LIMITS = HardLimits()
