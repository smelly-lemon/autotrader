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


class ModelConfig(BaseModel):
    name: str
    temperature: float = 0.2
    context_window: int = 8192


class ModelsConfig(BaseModel):
    scanner: ModelConfig
    analyzer: ModelConfig
    strategist: ModelConfig


class TradingConfig(BaseModel):
    pairs: list[str] = ["BTC/USD", "ETH/USD", "SOL/USD"]
    mode: Literal["paper", "live"] = "paper"
    initial_balance: float = 1000.0


class ScheduleConfig(BaseModel):
    scan_interval_seconds: int = 120
    deep_analysis_cooldown: int = 300
    portfolio_review_hours: int = 4
    data_retention_days: int = 30


class RiskConfig(BaseModel):
    max_position_pct: float = 0.15
    max_open_positions: int = 3
    daily_drawdown_limit_pct: float = 0.05
    stop_loss_pct: float = 0.03
    max_single_trade_loss_pct: float = 0.02
    cooldown_after_loss_seconds: int = 900


class ExchangeConfig(BaseModel):
    name: str = "coinbase"
    sandbox: bool = False
    rate_limit: bool = True


class ScannerConfig(BaseModel):
    confidence_threshold: float = 0.65


class AppConfig(BaseModel):
    trading: TradingConfig = Field(default_factory=TradingConfig)
    models: ModelsConfig
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)

    # Secrets from environment
    coinbase_api_key: str = ""
    coinbase_api_secret: str = ""
    ollama_base_url: str = "http://localhost:11434"


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    raw["coinbase_api_key"] = os.getenv("COINBASE_API_KEY", "")
    raw["coinbase_api_secret"] = os.getenv("COINBASE_API_SECRET", "")
    raw["ollama_base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return AppConfig(**raw)
