#!/usr/bin/env python3
"""Venue-wide swing inventory + strategy screen on high-amplitude pairs.

QUESTION (empirical): are there pairs on Coinbase whose swings are large
enough that capturing a realistic fraction clears the fee wall
(maker starter: 120 bps round trip; taker starter: 240 bps + spread)?

PHASE 1 — swing inventory (all active USD-quoted spot markets):
  ~600 days of daily candles per pair. Per pair: sigma and mean|r| of
  1d/3d/7d returns, p90 |r_3d|, median daily high-low range, median daily
  dollar volume, and the BREAKEVEN CAPTURE FRACTION = 120bps / mean|r_H|
  (what share of a typical swing you must capture just to pay maker fees).

PHASE 2 — strategy screen (rules fixed before looking at Phase 1 output):
  Candidates: history >= 350 days AND median daily dollar volume >= $250K,
  ranked by mean|r_3d|; take the top 12. Fetch ~2 years of 1h candles.
  Families (same construction as scripts/short_term_study.py, hourly bars):
    momentum:        r_L >= +k * sigma_L  -> buy, hold H
    mean reversion:  r_L <= -k * sigma_L  -> buy, hold H
    (L, H) in {(24h, 72h), (72h, 168h)};  k in {2.0, 3.0}
    sigma_L = trailing 30-day std of overlapping L-bar log returns
              (min 10 days), signal at close, enter next bar open,
              long-only, sequential (non-overlapping) trades per pair.
  Costs: maker_starter = 120 bps RT (no spread); taker_starter = 240 bps RT
  + live top-of-book spread (floored at 5 bps for alts).
  PASS: pooled mean net at maker > 0, bootstrap 95% CI (of the mean, net
  maker) excluding zero, and >= 60% of traded months positive.

Output: research/swing_screen_results.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "research" / "swing_screen_results.json"

DAY_MS = 86_400_000
INVENTORY_DAYS = 600
SCREEN_DAYS = 730
CACHE_DIR = PROJECT_ROOT / "data" / "external" / "h1"
DUMP_TRADES = False  # set by --dump-trades

MIN_HISTORY_DAYS = 350
MIN_MEDIAN_DOLLAR_VOL = 250_000
N_SELECT = 12

GRID = [(24, 72), (72, 168)]      # (L hours, H hours)
K_VALUES = [2.0, 3.0]
SIGMA_WINDOW_H = 30 * 24
SIGMA_MIN_PERIODS_H = 10 * 24

MAKER_RT = 0.0120
TAKER_RT = 0.0240

RNG = np.random.default_rng(42)
N_BOOT = 10_000


def fetch_paginated(ex, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now_ms = ex.milliseconds()
    rows, cur, guard = [], since_ms, 0
    while cur < now_ms and guard < 200:
        guard += 1
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=cur, limit=300)
        except Exception as e:  # noqa: BLE001 — skip transient/permission errors
            time.sleep(1)
            batch = []
        if not batch:
            cur += 300 * tf_ms
            continue
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= cur:
            break
        cur = nxt
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop(columns=["ts"])


# ------------------------------------------------------------------ phase 1

def inventory(ex) -> list[dict]:
    markets = ex.load_markets()
    symbols = sorted(
        m["symbol"] for m in markets.values()
        if m.get("quote") == "USD" and m.get("spot", True) and m.get("active", True))
    print(f"Phase 1: {len(symbols)} active USD spot markets", flush=True)

    since = ex.milliseconds() - INVENTORY_DAYS * DAY_MS
    out = []
    for i, sym in enumerate(symbols):
        df = fetch_paginated(ex, sym, "1d", since)
        if len(df) < 30:
            continue
        c = df["close"]
        logc = np.log(c)
        stats = {"pair": sym, "days": int(len(df)),
                 "median_dollar_vol": float((df["volume"] * c).median()),
                 "median_daily_range_pct": float(
                     ((df["high"] - df["low"]) / c).median() * 100)}
        for h, tag in [(1, "1d"), (3, "3d"), (7, "7d")]:
            r = logc.diff(h).dropna()
            if len(r) < 20:
                continue
            mean_abs = float(r.abs().mean())
            stats[f"mean_abs_r{tag}_pct"] = round(mean_abs * 100, 2)
            stats[f"p90_abs_r{tag}_pct"] = round(float(r.abs().quantile(0.9)) * 100, 2)
            stats[f"breakeven_capture_maker_{tag}"] = round(MAKER_RT / mean_abs, 3)
        out.append(stats)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(symbols)}", flush=True)
    return out


# ------------------------------------------------------------------ phase 2

def sequential_trades(sig: np.ndarray, opens: np.ndarray, hold: int,
                      idx: pd.DatetimeIndex) -> list[dict]:
    trades, n, nxt = [], len(sig), 0
    for t in np.flatnonzero(sig):
        if t < nxt:
            continue
        e, x = t + 1, t + 1 + hold
        if x >= n:
            break
        p_in, p_out = opens[e], opens[x]
        if not (np.isfinite(p_in) and np.isfinite(p_out)) or p_in <= 0:
            continue
        trades.append({"entry_ts": str(idx[e]), "month": str(idx[e])[:7],
                       "gross": p_out / p_in - 1.0})
        nxt = x
    return trades


def screen(ex, selected: list[str]) -> dict:
    print(f"Phase 2: fetching ~{SCREEN_DAYS}d of 1h candles for {len(selected)} pairs",
          flush=True)
    since = ex.milliseconds() - SCREEN_DAYS * DAY_MS
    candles, spreads = {}, {}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sym in selected:
        cache = CACHE_DIR / f"{sym.replace('/', '_')}.parquet"
        df = pd.DataFrame()
        if cache.exists():
            cached = pd.read_parquet(cache)
            if len(cached) and (pd.Timestamp.now(tz="UTC") - cached.index[-1]
                                ) < pd.Timedelta(days=3):
                df = cached
        if df.empty:
            df = fetch_paginated(ex, sym, "1h", since)
            if len(df):
                df.to_parquet(cache)
        if len(df) < MIN_HISTORY_DAYS * 24 // 2:
            print(f"  {sym}: only {len(df)} bars, skipping", flush=True)
            continue
        full = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
        candles[sym] = df.reindex(full)
        try:
            t = ex.fetch_ticker(sym)
            bid, ask = t.get("bid") or 0, t.get("ask") or 0
            spreads[sym] = max((ask - bid) / ((ask + bid) / 2), 0.0005) \
                if bid and ask else 0.0020
        except Exception:  # noqa: BLE001
            spreads[sym] = 0.0020
        print(f"  {sym}: {df['close'].notna().sum():,} bars "
              f"({df.index[0].date()} -> {df.index[-1].date()}), "
              f"spread {spreads[sym] * 1e4:.0f}bps", flush=True)

    results = []
    for (L, H) in GRID:
        for k in K_VALUES:
            for fam in ("mom", "mr"):
                all_trades = []
                per_pair = {}
                for sym, df in candles.items():
                    logc = np.log(df["close"].to_numpy())
                    r_L = pd.Series(logc).diff(L)
                    sigma = r_L.rolling(SIGMA_WINDOW_H,
                                        min_periods=SIGMA_MIN_PERIODS_H).std()
                    s_np, r_np = sigma.to_numpy(), r_L.to_numpy()
                    valid = np.isfinite(s_np) & (s_np > 0) & np.isfinite(r_np)
                    sig = valid & ((r_np >= k * s_np) if fam == "mom"
                                   else (r_np <= -k * s_np))
                    trades = sequential_trades(sig, df["open"].to_numpy(), H, df.index)
                    for tr in trades:
                        tr["pair"] = sym
                        tr["net_maker"] = tr["gross"] - MAKER_RT
                        tr["net_taker"] = tr["gross"] - TAKER_RT - spreads[sym]
                    all_trades.extend(trades)
                    if trades:
                        g = np.array([t["gross"] for t in trades])
                        per_pair[sym] = {"n": len(g),
                                         "mean_gross_bps": round(float(g.mean()) * 1e4, 1)}
                results.append(summarize(fam, L, H, k, all_trades, per_pair))
    return {"spreads_bps": {s: round(v * 1e4, 1) for s, v in spreads.items()},
            "configs": results}


def summarize(fam: str, L: int, H: int, k: float,
              trades: list[dict], per_pair: dict) -> dict:
    out = {"family": fam, "L_h": L, "H_h": H, "k": k,
           "n_trades": len(trades), "per_pair": per_pair}
    if not trades:
        return out
    gross = np.array([t["gross"] for t in trades])
    net_maker = np.array([t["net_maker"] for t in trades])
    net_taker = np.array([t["net_taker"] for t in trades])
    months = pd.Series([t["month"] for t in trades])
    monthly = pd.Series(net_maker).groupby(months.values).mean()
    boots = RNG.choice(net_maker, size=(N_BOOT, len(net_maker)), replace=True).mean(axis=1)
    ci = [float(np.percentile(boots, q)) for q in (2.5, 97.5)]
    frac_pos = float((monthly > 0).mean())
    out.update({
        "mean_gross_bps": round(float(gross.mean()) * 1e4, 1),
        "mean_net_maker_bps": round(float(net_maker.mean()) * 1e4, 1),
        "mean_net_taker_bps": round(float(net_taker.mean()) * 1e4, 1),
        "net_maker_ci95_bps": [round(c * 1e4, 1) for c in ci],
        "win_rate_gross": round(float((gross > 0).mean()), 3),
        "months_traded": int(len(monthly)),
        "months_positive_net_maker": int((monthly > 0).sum()),
        "total_net_maker_pct": round(float(net_maker.sum()) * 100, 1),
        "pass": bool(net_maker.mean() > 0 and ci[0] > 0 and frac_pos >= 0.6
                     and len(trades) >= 100),
    })
    if DUMP_TRADES:
        out["trades"] = [{"pair": t.get("pair", ""), "entry_ts": t["entry_ts"],
                          "gross": round(t["gross"], 6)} for t in trades]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-amp-pct", type=float, default=None,
                    help="Select ALL eligible pairs with mean|3d| >= this pct "
                         "(instead of the default top-12 by amplitude). Used for "
                         "the widened confirmation batch; rules are otherwise "
                         "identical to the original screen.")
    ap.add_argument("--reuse-inventory", type=Path, default=None,
                    help="Reuse phase1_inventory from a previous results JSON.")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--dump-trades", action="store_true",
                    help="Include raw per-trade records in the output JSON "
                         "(needed for cluster-robust statistics).")
    args = ap.parse_args()
    global DUMP_TRADES
    DUMP_TRADES = args.dump_trades

    ex = ccxt.coinbase({"enableRateLimit": True})
    if args.reuse_inventory and args.reuse_inventory.exists():
        inv = json.loads(args.reuse_inventory.read_text())["phase1_inventory"]
        print(f"Reusing inventory ({len(inv)} pairs) from {args.reuse_inventory}")
    else:
        inv = inventory(ex)

    eligible = [r for r in inv
                if r["days"] >= MIN_HISTORY_DAYS
                and r["median_dollar_vol"] >= MIN_MEDIAN_DOLLAR_VOL
                and "mean_abs_r3d_pct" in r]
    eligible.sort(key=lambda r: r["mean_abs_r3d_pct"], reverse=True)
    if args.min_amp_pct is not None:
        chosen = [r for r in eligible if r["mean_abs_r3d_pct"] >= args.min_amp_pct]
    else:
        chosen = eligible[:N_SELECT]
    selected = [r["pair"] for r in chosen]
    print(f"\nSelected {len(selected)} pairs "
          f"(vol>=${MIN_MEDIAN_DOLLAR_VOL / 1e3:.0f}K/d, hist>={MIN_HISTORY_DAYS}d):")
    for r in chosen:
        print(f"  {r['pair']:<14} mean|3d| {r['mean_abs_r3d_pct']:.2f}%  "
              f"breakeven capture {r['breakeven_capture_maker_3d']:.0%}  "
              f"${r['median_dollar_vol'] / 1e6:.1f}M/d")

    phase2 = screen(ex, selected)

    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "design": "pre-registered; see module docstring",
              "selection": (f"mean|3d| >= {args.min_amp_pct}%" if args.min_amp_pct
                            else f"top {N_SELECT} by mean|3d|"),
              "phase1_inventory": inv,
              "phase2_selected": selected,
              "phase2_screen": phase2}
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}\n")

    print("=== Phase 2 configs (pooled, net at maker 120bps RT) ===")
    for c in phase2["configs"]:
        if c["n_trades"] == 0:
            continue
        print(f"{c['family']:>3} L={c['L_h']:>3}h H={c['H_h']:>3}h k={c['k']}: "
              f"n={c['n_trades']:>4}  gross={c['mean_gross_bps']:>8.1f}bps  "
              f"net(maker)={c['mean_net_maker_bps']:>8.1f}bps "
              f"CI={c['net_maker_ci95_bps']}  "
              f"months+={c['months_positive_net_maker']}/{c['months_traded']}  "
              f"PASS={c['pass']}")


if __name__ == "__main__":
    sys.exit(main())
