#!/usr/bin/env python3
"""Two-sleeve paper trading loop — the forward test of the validated strategies.

Sleeve A, "crash" (hourly): buy 3-sigma 24h crashes on the 66 high-amplitude
  USD alts from the wide swing screen, hold 72h, max 5 concurrent positions,
  10% of equity each. When more signals fire than slots, rank by the
  LightGBM crash-triage model (retrained weekly from cached history);
  fallback ranking = deepest crash first.
  Evidence: research/swing_screen_wide_results.json (+199bps/trade net,
  17/24 months, borderline cluster significance).

Sleeve B, "trend" (daily): Donchian 55d/20d on daily closes for 8 majors,
  exit on 20d low or 120d cap, max 4 concurrent, 12.5% of equity each.
  Implemented as TARGET-STATE RECONCILIATION: each daily cycle replays the
  rule on full history and trades toward the replayed state, so restarts,
  downtime, and first-start initialization are all self-healing.
  Evidence: research/strategy_battery_results.json (+1933bps/trade net
  over 10y, both halves positive; regime-dependent, currently cold).

Execution model: paper only (hard assert). Maker fee 0.6%/side (verified
account tier, post-only limit assumption) + 0.05% slippage per side via
PaperExecutor. Fills at live ticker price at decision time. Dedicated
ledger: paper_trades.db (fresh $100 experiment; old trades.db untouched).

State: everything derives from the ledger + rule replay. Crash exits are
due at entry + 72h (stored in trade metadata). Every signal (taken or
skipped) is logged to the decisions table for the forward-vs-backtest
comparison.

Run:  python -u scripts/paper_trader.py           # long-running loop
      python -u scripts/paper_trader.py --once    # one cycle, then exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal as os_signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.config import load_config  # noqa: E402
from src.data.store import TradeStore  # noqa: E402
from src.execution.paper import PaperExecutor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("paper_trader")

# ---------------------------------------------------------------- constants

H1_DIR = PROJECT_ROOT / "data" / "external" / "h1"
D1_DIR = PROJECT_ROOT / "data" / "external" / "d1"
WIDE_RESULTS = PROJECT_ROOT / "research" / "swing_screen_wide_results.json"
INV_RESULTS = PROJECT_ROOT / "research" / "swing_screen_results.json"
MODEL_DIR = PROJECT_ROOT / "models"
RANKER_PATH = MODEL_DIR / "crash_ranker.txt"
RANKER_META = MODEL_DIR / "crash_ranker_meta.json"
STATUS_PATH = PROJECT_ROOT / "logs" / "paper_status.json"
PAPER_DB = PROJECT_ROOT / "paper_trades.db"

# Sleeve A (validated rule — do not tune)
CRASH_Z = -3.0
CRASH_HOLD_H = 72
CRASH_SLOTS = 5
CRASH_POS_PCT = 0.10
SIGMA_WINDOW_H = 30 * 24
SIGMA_MIN_H = 10 * 24

# Sleeve B (validated rule — do not tune)
TREND_MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD",
                "DOGE/USD", "LINK/USD", "LTC/USD", "BCH/USD"]
DONCHIAN_ENTRY_D = 55
DONCHIAN_EXIT_D = 20
TREND_CAP_D = 120
TREND_SLOTS = 4
TREND_POS_PCT = 0.125

RANKER_MAX_AGE_D = 7


# ---------------------------------------------------------------- pure logic

def crash_z_last_bar(close: pd.Series) -> float | None:
    """z = r24h / sigma24h on the last closed hourly bar (study-identical)."""
    if len(close) < SIGMA_MIN_H + 25:
        return None
    logc = np.log(close)
    r24 = logc.diff(24)
    sigma = r24.rolling(SIGMA_WINDOW_H, min_periods=SIGMA_MIN_H).std()
    z = r24.iloc[-1] / sigma.iloc[-1] if sigma.iloc[-1] and np.isfinite(sigma.iloc[-1]) else None
    return float(z) if z is not None and np.isfinite(z) else None


def donchian_state(closes: pd.Series,
                   entry_n: int = DONCHIAN_ENTRY_D,
                   exit_n: int = DONCHIAN_EXIT_D,
                   cap_days: int = TREND_CAP_D) -> tuple[bool, pd.Timestamp | None]:
    """Replay the Donchian rule over daily closes; return (in_position,
    entry_date). Semantics identical to the strategy battery: enter when
    close makes a new entry_n-day high (prev close didn't), exit when close
    makes a new exit_n-day low or the position is cap_days old."""
    c = closes.dropna()
    if len(c) < entry_n + 2:
        return False, None
    hi = c.rolling(entry_n).max()
    lo = c.rolling(exit_n).min()
    in_pos, entry_i = False, None
    for i in range(1, len(c)):
        if not in_pos:
            if (np.isfinite(hi.iloc[i]) and np.isfinite(hi.iloc[i - 1])
                    and c.iloc[i] >= hi.iloc[i] and c.iloc[i - 1] < hi.iloc[i - 1]):
                in_pos, entry_i = True, i
        else:
            aged = (c.index[i] - c.index[entry_i]).days >= cap_days
            if c.iloc[i] <= lo.iloc[i] or aged:
                in_pos, entry_i = False, None
    return in_pos, (c.index[entry_i] if in_pos else None)


# ---------------------------------------------------------------- trader

class PaperTrader:
    def __init__(self):
        config = load_config()
        assert config.trading.mode == "paper", "paper_trader refuses to run live"
        self.store = TradeStore(PAPER_DB)
        self.executor = PaperExecutor(
            config.trading.initial_balance,
            self.store,
            fee_pct=config.trading.maker_fee_pct,
            slippage_pct=config.trading.slippage_pct,
        )
        self.executor.restore_state()

        wide = json.loads(WIDE_RESULTS.read_text())
        self.crash_universe: list[str] = wide["phase2_selected"]
        self.spreads: dict[str, float] = wide["phase2_screen"]["spreads_bps"]
        self.inv = {r["pair"]: r for r in
                    json.loads(INV_RESULTS.read_text())["phase1_inventory"]}

        self.h1: dict[str, pd.DataFrame] = {}
        self.d1: dict[str, pd.DataFrame] = {}
        self._exchange = None
        self._ranker = None
        self._stop = False
        self.last_hourly: str | None = None
        self.last_daily: str | None = None

    # ------------------------------------------------------------ data

    async def exchange(self):
        if self._exchange is None:
            import ccxt.async_support as ccxt_async
            self._exchange = ccxt_async.coinbase({"enableRateLimit": True})
        return self._exchange

    def _load_caches(self):
        for sym in self.crash_universe + ["BTC/USD"]:
            f = H1_DIR / f"{sym.replace('/', '_')}.parquet"
            if f.exists():
                self.h1[sym] = pd.read_parquet(f)
        for sym in TREND_MAJORS:
            f = D1_DIR / f"{sym.replace('/', '_')}.parquet"
            if f.exists():
                self.d1[sym] = pd.read_parquet(f)
        logger.info("caches loaded: %d hourly, %d daily", len(self.h1), len(self.d1))

    async def _refresh_symbol(self, sym: str, timeframe: str, sem: asyncio.Semaphore):
        cache = self.h1 if timeframe == "1h" else self.d1
        cdir = H1_DIR if timeframe == "1h" else D1_DIR
        step = pd.Timedelta(hours=1) if timeframe == "1h" else pd.Timedelta(days=1)
        cutoff = pd.Timestamp.now(tz="UTC").floor("h" if timeframe == "1h" else "D")
        ex = await self.exchange()
        rows = None
        async with sem:
            for attempt in (1, 2):
                try:
                    # 300 bars = 12.5 days (1h) / 300 days (1d): heals outages
                    rows = await ex.fetch_ohlcv(sym, timeframe, limit=300)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        logger.warning("fetch %s %s failed: %s",
                                       sym, timeframe, str(e)[:120])
                        return
                    await asyncio.sleep(2)
        if not rows:
            return
        new = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        new.index = pd.to_datetime(new["ts"], unit="ms", utc=True)
        new = new.drop(columns=["ts"])
        new = new[new.index < cutoff]  # drop the in-progress bar
        old = cache.get(sym)
        df = pd.concat([old, new]) if old is not None else new
        df = df[~df.index.duplicated(keep="last")].sort_index()
        cache[sym] = df
        cdir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cdir / f"{sym.replace('/', '_')}.parquet")

    async def refresh_hourly_data(self):
        sem = asyncio.Semaphore(6)
        await asyncio.gather(*(self._refresh_symbol(s, "1h", sem)
                               for s in self.crash_universe + ["BTC/USD"]))

    async def refresh_daily_data(self):
        sem = asyncio.Semaphore(6)
        await asyncio.gather(*(self._refresh_symbol(s, "1d", sem)
                               for s in TREND_MAJORS))

    async def live_price(self, sym: str) -> float | None:
        ex = await self.exchange()
        try:
            t = await ex.fetch_ticker(sym)
            p = t.get("last") or t.get("close")
            return float(p) if p else None
        except Exception as e:  # noqa: BLE001
            logger.warning("ticker %s failed: %s", sym, str(e)[:120])
            return None

    # ------------------------------------------------------------ sleeve registry

    def sleeve_positions(self) -> dict[str, dict]:
        """symbol -> {sleeve, meta, trade}  from the open trades ledger."""
        out = {}
        for t in self.store.get_open_trades():
            meta = json.loads(t["metadata"]) if t.get("metadata") else {}
            out[t["symbol"]] = {"sleeve": meta.get("sleeve", "crash"),
                                "meta": meta, "trade": t}
        return out

    # ------------------------------------------------------------ ranker

    def ensure_ranker(self):
        """Train/refresh the LightGBM crash-triage ranker from cached history."""
        try:
            if RANKER_PATH.exists() and RANKER_META.exists():
                meta = json.loads(RANKER_META.read_text())
                age = (datetime.now(timezone.utc)
                       - datetime.fromisoformat(meta["trained"])).days
                if age < RANKER_MAX_AGE_D:
                    if self._ranker is None:
                        import lightgbm as lgb
                        self._ranker = lgb.Booster(model_file=str(RANKER_PATH))
                        self._ranker_feats = meta["features"]
                    return
            import lightgbm as lgb
            from crash_model_study import FEATURES, LGB_PARAMS, build_events
            ev = build_events(CRASH_Z)
            # only events whose 72h outcome is known
            ev = ev[ev["entry_ts"] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=73)]
            if len(ev) < 200:
                logger.warning("ranker: only %d events, keeping fallback", len(ev))
                return
            lo, hi = ev["gross"].quantile([0.05, 0.95])
            mdl = lgb.LGBMRegressor(**LGB_PARAMS)
            mdl.fit(ev[FEATURES].astype(float), ev["gross"].clip(lo, hi))
            MODEL_DIR.mkdir(exist_ok=True)
            mdl.booster_.save_model(str(RANKER_PATH))
            RANKER_META.write_text(json.dumps(
                {"trained": datetime.now(timezone.utc).isoformat(),
                 "n_events": int(len(ev)), "features": FEATURES}))
            self._ranker = mdl.booster_
            self._ranker_feats = FEATURES
            logger.info("ranker trained on %d events", len(ev))
        except Exception:  # noqa: BLE001
            logger.exception("ranker training failed; deepest-crash fallback")

    def score_candidate(self, sym: str, z: float) -> float | None:
        """Predicted 72h gross return for a crash candidate (entry-time features)."""
        if self._ranker is None:
            return None
        df = self.h1.get(sym)
        btc = self.h1.get("BTC/USD")
        if df is None or btc is None:
            return None
        c, v = df["close"], df["volume"]
        logc = np.log(c)
        r24 = logc.diff(24)
        sigma24 = r24.rolling(SIGMA_WINDOW_H, min_periods=SIGMA_MIN_H).std()
        vol24 = v.rolling(24).sum()
        stat = self.inv.get(sym, {})
        btc_r24 = float(np.log(btc["close"]).diff(24).iloc[-1])
        feats = {
            "crash_z": z,
            "r24": float(r24.iloc[-1]),
            "r72": float(logc.diff(72).iloc[-1]),
            "r168": float(logc.diff(168).iloc[-1]),
            "sigma24": float(sigma24.iloc[-1]),
            "vol24_ratio": float(vol24.iloc[-1] /
                                 vol24.rolling(SIGMA_WINDOW_H, min_periods=SIGMA_MIN_H)
                                 .median().iloc[-1]),
            "dd30": float(c.iloc[-1] / c.rolling(SIGMA_WINDOW_H,
                                                 min_periods=SIGMA_MIN_H).max().iloc[-1] - 1),
            "btc_r24": btc_r24,
            "idio_r24": float(r24.iloc[-1]) - btc_r24,
            "amp3d": stat.get("mean_abs_r3d_pct", np.nan),
            "log_dollar_vol": np.log10(max(stat.get("median_dollar_vol", 1.0), 1.0)),
            "spread_bps": self.spreads.get(sym, 20.0),
            "listing_age_days": (df.index[-1] - df.index[0]).days,
            "log_price": float(np.log10(c.iloc[-1])) if c.iloc[-1] > 0 else np.nan,
        }
        try:
            x = pd.DataFrame([feats])[self._ranker_feats].astype(float)
            return float(self._ranker.predict(x)[0])
        except Exception:  # noqa: BLE001
            logger.exception("ranker scoring failed for %s", sym)
            return None

    # ------------------------------------------------------------ crash sleeve

    async def crash_cycle(self):
        positions = self.sleeve_positions()

        # 1. exits due
        now = datetime.now(timezone.utc)
        for sym, p in positions.items():
            if p["sleeve"] != "crash":
                continue
            exit_at = p["meta"].get("exit_at")
            if exit_at and now >= datetime.fromisoformat(exit_at):
                price = await self.live_price(sym)
                if price:
                    r = await self.executor.execute_sell(
                        sym, p["trade"]["amount"], price)
                    logger.info("crash exit %s: %s", sym,
                                "ok" if r.success else r.error)

        # 2. new signals on the last closed bar
        positions = self.sleeve_positions()
        crash_held = sum(1 for p in positions.values() if p["sleeve"] == "crash")
        candidates = []
        for sym in self.crash_universe:
            if sym in positions or sym not in self.h1:
                continue
            df = self.h1[sym]
            full = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
            close = df["close"].reindex(full)
            if (pd.Timestamp.now(tz="UTC") - close.index[-1]) > pd.Timedelta(hours=3):
                continue  # stale data — no signal
            z = crash_z_last_bar(close)
            if z is not None and z <= CRASH_Z:
                candidates.append((sym, z))

        if candidates:
            self.ensure_ranker()
            scored = [(sym, z, self.score_candidate(sym, z)) for sym, z in candidates]
            # rank: model prediction desc; fallback deepest crash first
            if all(s[2] is not None for s in scored):
                scored.sort(key=lambda s: -s[2])
            else:
                scored.sort(key=lambda s: s[1])
            slots = CRASH_SLOTS - crash_held
            balance = await self.executor.get_balance()
            for rank, (sym, z, pred) in enumerate(scored):
                take = rank < slots
                executed = False
                if take:
                    price = await self.live_price(sym)
                    usd = min(CRASH_POS_PCT * balance["total_value"],
                              self.executor.cash * 0.99)
                    if price and usd >= 1.0:
                        exit_at = (now + timedelta(hours=CRASH_HOLD_H)).isoformat()
                        r = await self.executor.execute_buy(
                            sym, usd, price,
                            stop_loss_pct=None, take_profit_pct=None,
                            metadata={"sleeve": "crash", "z": round(z, 2),
                                      "pred": pred, "exit_at": exit_at},
                            model_tier="crash")
                        executed = r.success
                        logger.info("crash entry %s z=%.2f pred=%s $%.2f: %s",
                                    sym, z, f"{pred:.4f}" if pred else "-",
                                    usd, "ok" if r.success else r.error)
                self.store.log_decision(
                    symbol=sym, model_tier="crash",
                    model_name="lgb_ranker" if pred is not None else "deepest_z",
                    action="buy" if take else "skip",
                    confidence=float(pred if pred is not None else z),
                    reasoning=f"z={z:.2f} rank={rank + 1}/{len(scored)} slots={slots}",
                    raw_output="", was_executed=executed)

    # ------------------------------------------------------------ trend sleeve

    async def trend_cycle(self):
        positions = self.sleeve_positions()
        target: dict[str, pd.Timestamp] = {}
        for sym in TREND_MAJORS:
            df = self.d1.get(sym)
            if df is None or not len(df):
                continue
            in_pos, entry_date = donchian_state(df["close"])
            if in_pos:
                target[sym] = entry_date

        held = {s for s, p in positions.items() if p["sleeve"] == "trend"}
        # capacity cap prefers the OLDEST trends: established positions are
        # never evicted by newer signals (matches per-pair rule independence
        # as closely as a slot cap allows)
        want = set(sorted(target, key=lambda s: target[s])[:TREND_SLOTS])

        # exits: held but no longer in target state
        for sym in held - want:
            price = await self.live_price(sym)
            if price:
                p = positions[sym]
                r = await self.executor.execute_sell(sym, p["trade"]["amount"], price)
                logger.info("trend exit %s: %s", sym, "ok" if r.success else r.error)
                self.store.log_decision(
                    symbol=sym, model_tier="trend", model_name="donchian_55_20",
                    action="sell", confidence=0.0,
                    reasoning="left target state (20d low / 120d cap)",
                    raw_output="", was_executed=r.success)

        # entries: in target state but not held (includes first-start replay init)
        balance = await self.executor.get_balance()
        for sym in want - held:
            if sym in positions:  # safety: symbol held by another sleeve
                continue
            price = await self.live_price(sym)
            usd = min(TREND_POS_PCT * balance["total_value"],
                      self.executor.cash * 0.99)
            if not price or usd < 1.0:
                continue
            r = await self.executor.execute_buy(
                sym, usd, price, stop_loss_pct=None, take_profit_pct=None,
                metadata={"sleeve": "trend",
                          "replay_entry": str(target[sym].date())},
                model_tier="trend")
            logger.info("trend entry %s (signal date %s) $%.2f: %s",
                        sym, target[sym].date(), usd,
                        "ok" if r.success else r.error)
            self.store.log_decision(
                symbol=sym, model_tier="trend", model_name="donchian_55_20",
                action="buy", confidence=0.0,
                reasoning=f"target state since {target[sym].date()}",
                raw_output="", was_executed=r.success)

    # ------------------------------------------------------------ status

    async def write_status(self):
        marks = {}
        for sym in list(self.executor.positions):
            df = self.h1.get(sym)
            if df is None:
                df = self.d1.get(sym)
            if df is not None and len(df):
                marks[sym] = float(df["close"].iloc[-1])
        await self.executor.check_stop_losses(marks)  # updates marks; SL/TP disabled
        balance = await self.executor.get_balance()
        positions = self.sleeve_positions()
        realized = self.store.conn.execute(
            "SELECT COALESCE(SUM(pnl),0) AS p FROM trades WHERE status != 'open'"
        ).fetchone()["p"]
        status = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "total_value": round(balance["total_value"], 2),
            "cash": round(balance["cash"], 2),
            "initial_balance": balance["initial_balance"],
            "realized_pnl": round(float(realized), 2),
            "n_positions": len(balance["positions"]),
            "positions": {
                s: {"sleeve": positions.get(s, {}).get("sleeve", "?"),
                    "entry": round(p["entry_price"], 6),
                    "mark": round(p["mark_price"], 6),
                    "meta": positions.get(s, {}).get("meta", {})}
                for s, p in balance["positions"].items()},
            "last_hourly": self.last_hourly,
            "last_daily": self.last_daily,
        }
        STATUS_PATH.parent.mkdir(exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2))
        self.store.log_portfolio_snapshot(
            balance["total_value"], balance["cash"], balance["positions"])
        logger.info("equity $%.2f (cash $%.2f, %d positions)",
                    balance["total_value"], balance["cash"],
                    len(balance["positions"]))

    # ------------------------------------------------------------ loop

    async def hourly(self):
        await self.refresh_hourly_data()
        await self.crash_cycle()
        self.last_hourly = datetime.now(timezone.utc).isoformat()
        await self.write_status()

    async def daily(self):
        await self.refresh_daily_data()
        await self.trend_cycle()
        self.last_daily = datetime.now(timezone.utc).isoformat()
        await self.write_status()

    async def run(self, once: bool = False):
        self._load_caches()
        self.ensure_ranker()
        await self.daily()
        await self.hourly()
        if once:
            await self.close()
            return
        while not self._stop:
            now = pd.Timestamp.now(tz="UTC")
            next_hourly = now.floor("h") + pd.Timedelta(seconds=90)
            if next_hourly <= now:
                next_hourly += pd.Timedelta(hours=1)
            next_daily = now.floor("D") + pd.Timedelta(minutes=6)
            if next_daily <= now:
                next_daily += pd.Timedelta(days=1)
            nxt = min(next_hourly, next_daily)
            # sleep in short chunks so SIGTERM lands promptly
            while not self._stop and pd.Timestamp.now(tz="UTC") < nxt:
                remaining = (nxt - pd.Timestamp.now(tz="UTC")).total_seconds()
                await asyncio.sleep(min(max(remaining, 0.5), 60))
            if self._stop:
                break
            try:
                if nxt == next_daily:
                    await self.daily()
                else:
                    await self.hourly()
            except Exception:  # noqa: BLE001 — the loop must survive anything
                logger.exception("cycle failed; continuing")
        await self.close()

    async def close(self):
        if self._exchange is not None:
            await self._exchange.close()
        self.store.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="run one daily + hourly cycle, then exit")
    args = ap.parse_args()

    trader = PaperTrader()

    def _stop(*_):
        trader._stop = True

    for sig in (os_signal.SIGTERM, os_signal.SIGINT):
        os_signal.signal(sig, _stop)

    asyncio.run(trader.run(once=args.once))


if __name__ == "__main__":
    main()
