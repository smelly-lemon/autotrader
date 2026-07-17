from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config import DB_PATH, load_config
from src.data.store import TradeStore

TEMPLATE_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="Auto-Trader Dashboard")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

_store: TradeStore | None = None


def get_store() -> TradeStore:
    global _store
    if _store is None:
        _store = TradeStore()
    return _store


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    store = get_store()
    config = load_config()

    open_trades = store.get_open_trades()
    recent_trades = store.get_recent_trades(limit=30)
    recent_decisions = store.get_recent_decisions(limit=30)
    daily_pnl = store.get_daily_pnl()
    snapshots = store.get_portfolio_snapshots(limit=50)

    # Compute summary stats
    closed_trades = [t for t in recent_trades if t["status"] != "open"]
    wins = [t for t in closed_trades if (t.get("pnl") or 0) > 0]
    losses = [t for t in closed_trades if (t.get("pnl") or 0) < 0]
    total_pnl = sum(t.get("pnl") or 0 for t in closed_trades)
    win_rate = len(wins) / len(closed_trades) * 100 if closed_trades else 0

    latest_snapshot = snapshots[0] if snapshots else None
    portfolio_value = latest_snapshot["total_value"] if latest_snapshot else config.trading.initial_balance

    # Chart data
    snapshot_labels = []
    snapshot_values = []
    for s in reversed(snapshots):
        snapshot_labels.append(s["timestamp"][:16])
        snapshot_values.append(round(s["total_value"], 2))

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "config": config,
        "open_trades": open_trades,
        "recent_trades": recent_trades,
        "recent_decisions": recent_decisions,
        "daily_pnl": daily_pnl,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "wins": len(wins),
        "losses": len(losses),
        "portfolio_value": portfolio_value,
        "snapshot_labels": json.dumps(snapshot_labels),
        "snapshot_values": json.dumps(snapshot_values),
    })


@app.get("/api/trades")
async def api_trades(limit: int = 50):
    return get_store().get_recent_trades(limit)


@app.get("/api/decisions")
async def api_decisions(limit: int = 50):
    return get_store().get_recent_decisions(limit)


@app.get("/api/portfolio")
async def api_portfolio():
    store = get_store()
    return {
        "daily_pnl": store.get_daily_pnl(),
        "open_trades": store.get_open_trades(),
        "snapshots": store.get_portfolio_snapshots(limit=10),
    }


def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
