from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import DB_PATH


class TradeStore:
    """SQLite-backed storage for trades, candles, and model decisions."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._ensure_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_tables(self):
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,          -- 'buy' or 'sell'
                price REAL NOT NULL,
                amount REAL NOT NULL,
                cost REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT DEFAULT 'open',  -- 'open', 'closed', 'stopped'
                pnl REAL,
                close_price REAL,
                close_timestamp TEXT,
                model_tier TEXT,             -- 'scanner', 'analyzer', 'strategist'
                reasoning TEXT,
                metadata TEXT                -- JSON blob for extra data
            );

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                model_tier TEXT NOT NULL,
                model_name TEXT,
                action TEXT NOT NULL,        -- 'buy', 'sell', 'hold', 'escalate'
                confidence REAL,
                reasoning TEXT,
                raw_output TEXT,
                indicators TEXT,             -- JSON of indicator snapshot
                was_executed INTEGER DEFAULT 0,
                risk_vetoed INTEGER DEFAULT 0,
                veto_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                positions TEXT NOT NULL,     -- JSON
                daily_pnl REAL,
                daily_pnl_pct REAL
            );

            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(timestamp);
        """)
        c.commit()

    def log_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        model_tier: str = "",
        reasoning: str = "",
        metadata: dict | None = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            """INSERT INTO trades
               (timestamp, symbol, side, price, amount, cost, stop_loss, take_profit,
                model_tier, reasoning, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, symbol, side, price, amount, price * amount,
             stop_loss, take_profit, model_tier, reasoning,
             json.dumps(metadata) if metadata else None),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def close_trade(
        self,
        trade_id: int,
        close_price: float,
        status: str = "closed",
    ):
        now = datetime.now(timezone.utc).isoformat()
        row = self.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if not row:
            return
        entry_price = row["price"]
        side = row["side"]
        amount = row["amount"]
        if side == "buy":
            pnl = (close_price - entry_price) * amount
        else:
            pnl = (entry_price - close_price) * amount

        self.conn.execute(
            """UPDATE trades SET status=?, pnl=?, close_price=?, close_timestamp=?
               WHERE id=?""",
            (status, pnl, close_price, now, trade_id),
        )
        self.conn.commit()

    def log_decision(
        self,
        symbol: str,
        model_tier: str,
        model_name: str,
        action: str,
        confidence: float,
        reasoning: str,
        raw_output: str,
        indicators: dict | None = None,
        was_executed: bool = False,
        risk_vetoed: bool = False,
        veto_reason: str = "",
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            """INSERT INTO decisions
               (timestamp, symbol, model_tier, model_name, action, confidence,
                reasoning, raw_output, indicators, was_executed, risk_vetoed, veto_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, symbol, model_tier, model_name, action, confidence,
             reasoning, raw_output,
             json.dumps(indicators) if indicators else None,
             int(was_executed), int(risk_vetoed), veto_reason),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def log_portfolio_snapshot(
        self,
        total_value: float,
        cash: float,
        positions: dict,
        daily_pnl: float = 0,
        daily_pnl_pct: float = 0,
    ):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO portfolio_snapshots
               (timestamp, total_value, cash, positions, daily_pnl, daily_pnl_pct)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now, total_value, cash, json.dumps(positions), daily_pnl, daily_pnl_pct),
        )
        self.conn.commit()

    def get_open_trades(self, symbol: str | None = None) -> list[dict]:
        if symbol:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE status='open' AND symbol=? ORDER BY timestamp",
                (symbol,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY timestamp"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_trades(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_decisions(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_pnl(self) -> float:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE close_timestamp LIKE ?",
            (f"{today}%",),
        ).fetchone()
        return float(row["total"])  # type: ignore[index]

    def get_portfolio_snapshots(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
