from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.config import RiskConfig
from src.data.store import TradeStore
from src.llm.parser import TradeDecision
from src.risk.limits import HARD_LIMITS

logger = logging.getLogger(__name__)


@dataclass
class RiskVerdict:
    approved: bool
    original_decision: TradeDecision
    adjusted_size_pct: float = 0.0
    adjusted_stop_loss_pct: float = 0.03
    veto_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RiskManager:
    """Risk gate between LLM decisions and trade execution.

    This layer CANNOT be overridden by model output. It enforces
    hard limits on position sizing, drawdown, and cooldowns.
    """

    def __init__(self, config: RiskConfig, store: TradeStore):
        self.config = config
        self.store = store
        self._last_loss_time: float = 0
        self._risk_multiplier: float = 1.0

    def set_risk_multiplier(self, multiplier: float):
        clamped = max(0.1, min(2.0, multiplier))
        self._risk_multiplier = clamped
        logger.info("Risk multiplier set to %.2f", clamped)

    def evaluate(
        self,
        decision: TradeDecision,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        cash_available: float,
    ) -> RiskVerdict:
        if decision.action == "hold":
            return RiskVerdict(approved=False, original_decision=decision, veto_reasons=["hold"])

        veto_reasons: list[str] = []
        warnings: list[str] = []

        # 1. Check cooldown after loss
        if self._last_loss_time > 0:
            elapsed = time.time() - self._last_loss_time
            cooldown = max(self.config.cooldown_after_loss_seconds, HARD_LIMITS.MIN_COOLDOWN_SECONDS)
            if elapsed < cooldown:
                remaining = int(cooldown - elapsed)
                veto_reasons.append(f"Loss cooldown active: {remaining}s remaining")

        # 2. Check daily drawdown
        daily_pnl = self.store.get_daily_pnl()
        daily_drawdown_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 and daily_pnl < 0 else 0
        limit = min(self.config.daily_drawdown_limit_pct, HARD_LIMITS.ABSOLUTE_MAX_DAILY_DRAWDOWN_PCT)
        if daily_drawdown_pct >= limit:
            veto_reasons.append(
                f"Daily drawdown limit hit: {daily_drawdown_pct:.2%} >= {limit:.2%}"
            )

        # 3. Check max open positions
        open_trades = self.store.get_open_trades()
        max_positions = min(self.config.max_open_positions, HARD_LIMITS.ABSOLUTE_MAX_OPEN_POSITIONS)
        if len(open_trades) >= max_positions:
            veto_reasons.append(
                f"Max open positions reached: {len(open_trades)}/{max_positions}"
            )

        # 4. Check for existing position in same symbol
        symbol_positions = [t for t in open_trades if t["symbol"] == symbol]
        if symbol_positions and decision.action == "buy":
            veto_reasons.append(f"Already have open position in {symbol}")

        # 5. Size the position
        max_pct = min(
            decision.size_pct,
            self.config.max_position_pct * self._risk_multiplier,
            HARD_LIMITS.ABSOLUTE_MAX_POSITION_PCT,
        )
        trade_value = cash_available * max_pct

        if trade_value < HARD_LIMITS.MIN_ORDER_SIZE_USD:
            veto_reasons.append(f"Order too small: ${trade_value:.2f} < ${HARD_LIMITS.MIN_ORDER_SIZE_USD}")

        # 6. Check single trade loss limit
        stop_loss_pct = max(
            decision.stop_loss_pct,
            HARD_LIMITS.ABSOLUTE_MIN_STOP_LOSS_PCT,
        )
        potential_loss = trade_value * stop_loss_pct
        max_loss = portfolio_value * min(
            self.config.max_single_trade_loss_pct,
            HARD_LIMITS.ABSOLUTE_MAX_SINGLE_TRADE_LOSS_PCT,
        )
        if potential_loss > max_loss:
            old_size = max_pct
            max_pct = max_loss / (cash_available * stop_loss_pct) if cash_available * stop_loss_pct > 0 else 0
            max_pct = max(0, min(max_pct, HARD_LIMITS.ABSOLUTE_MAX_POSITION_PCT))
            warnings.append(
                f"Position sized down from {old_size:.2%} to {max_pct:.2%} "
                f"to respect max single-trade loss of ${max_loss:.2f}"
            )

        if veto_reasons:
            logger.warning("Trade VETOED for %s: %s", symbol, "; ".join(veto_reasons))
            return RiskVerdict(
                approved=False,
                original_decision=decision,
                veto_reasons=veto_reasons,
                warnings=warnings,
            )

        return RiskVerdict(
            approved=True,
            original_decision=decision,
            adjusted_size_pct=max_pct,
            adjusted_stop_loss_pct=stop_loss_pct,
            warnings=warnings,
        )

    def record_loss(self):
        self._last_loss_time = time.time()
        logger.info("Loss recorded, cooldown activated")
