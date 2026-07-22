from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DB_PATH = PROJECT_ROOT / "trades.db"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


class TradingConfig(BaseModel):
    pairs: list[str] = ["BTC/USD", "ETH/USD", "SOL/USD"]
    mode: Literal["paper", "live"] = "paper"
    initial_balance: float = 100.0
    # Simulated execution costs (paper mode). Default = Coinbase starter tier.
    taker_fee_pct: float = 0.012
    slippage_pct: float = 0.0005


class ScheduleConfig(BaseModel):
    signal_interval_seconds: int = 60
    candle_fetch_interval: int = 60
    retrain_hours: int = 24
    data_retention_days: int = 30


class MLConfig(BaseModel):
    target_horizon: int = 15
    confidence_threshold: float = 0.58
    position_size_pct: float = 0.10
    stop_loss_pct: float = 0.025
    take_profit_pct: float = 0.05
    max_holding_minutes: int = 120
    cv_folds: int = 5
    min_auc_to_trade: float = 0.52


class RiskConfig(BaseModel):
    max_position_pct: float = 0.15
    max_open_positions: int = 3
    daily_drawdown_limit_pct: float = 0.05
    stop_loss_pct: float = 0.025
    max_single_trade_loss_pct: float = 0.02
    cooldown_after_loss_seconds: int = 300


class ExchangeConfig(BaseModel):
    name: str = "coinbase"
    sandbox: bool = False
    rate_limit: bool = True


class AppConfig(BaseModel):
    trading: TradingConfig = Field(default_factory=TradingConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)

    coinbase_api_key_name: str = ""
    coinbase_api_private_key: str = ""


def _decode_pem(raw: str) -> str:
    """Handle \\n escapes in PEM keys from .env files."""
    if "\\n" in raw:
        return raw.replace("\\n", "\n")
    return raw


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Two CDP key formats are supported (both pass to ccxt as apiKey/secret):
    #   legacy: COINBASE_API_KEY_NAME (organizations/.../apiKeys/...) +
    #           COINBASE_API_PRIVATE_KEY (EC PEM, \n-escaped)
    #   new:    COINBASE_API_KEY_ID (UUID) +
    #           COINBASE_API_KEY_SECRET (base64 Ed25519)
    raw["coinbase_api_key_name"] = (
        os.getenv("COINBASE_API_KEY_NAME") or os.getenv("COINBASE_API_KEY_ID") or "")
    pk = os.getenv("COINBASE_API_PRIVATE_KEY") or os.getenv("COINBASE_API_KEY_SECRET") or ""
    raw["coinbase_api_private_key"] = _decode_pem(pk) if pk else ""

    return AppConfig(**raw)
