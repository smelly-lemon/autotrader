#!/usr/bin/env python3
"""Short-term structure study: what sub-daily patterns exist in our data,
and what fee floor would each survive?

PRE-REGISTERED DESIGN (fixed before running — no tuning on results):

Data
----
1m OHLCV candles (SQLite `trades.db`), Nov 5 2025 -> Jul 17 2026 (~366K min/pair).
Signal pairs (8, USD-quoted): BTC, ETH, SOL, XRP, DOGE, LINK, UNI, SHIB.
BTC/USDC and ETH/USDC are used only for the parity family (F4).

Families (F1/F2 are strategies; F3/F4 are diagnostics)
------------------------------------------------------
All strategies: long-only, one position per pair, no pyramiding.
Signal computed at close of minute t -> enter at open[t+1], exit at open[t+1+H].

F1 mean reversion:  L-minute log return <= -k * sigma_L  -> buy, hold H minutes.
F2 momentum:        L-minute log return >= +k * sigma_L  -> buy, hold H minutes.
    sigma_L = trailing 7-day std of overlapping L-minute log returns
              (min 2 days of history before any signal).
    Grid: (L, H) in {(15, 60), (60, 240)};  k in {2.0, 3.0}.
    -> 4 configs per family, fixed in advance.

F3 hour-of-day seasonality (diagnostic): mean 1h return by UTC hour,
    with per-month sign consistency.

F4 USD/USDC parity (diagnostic): basis_t = close_USD / close_USDC - 1 per
    minute for BTC and ETH. Distribution, exceedance frequencies
    (5/10/20/50 bps), and episode persistence for |basis| > 10 bps.

Costs (round trip)
------------------
- spread_only:    per-pair median top-of-book spread (ticker parquet,
                  floored at 1 bp) — the zero-fee-world lower bound.
- maker_starter:  2 x 0.60% (limit fills, no spread paid) = 120 bps.
- taker_starter:  2 x 1.20% + spread = 240 bps + spread.

Pass criteria (fixed in advance)
--------------------------------
PRIMARY (tradable on this account today): any F1/F2 config, pooled across
pairs, >= 200 trades, mean per-trade net at maker_starter > 0 with bootstrap
95% CI excluding zero, AND >= 60% of calendar months positive.
SECONDARY (an edge exists at all): same test net of spread_only costs.
Everything else is diagnostics, reported but not claimed.
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "trades.db"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "research" / "short_term_study_results.json"

SIGNAL_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD",
                "DOGE/USD", "LINK/USD", "UNI/USD", "SHIB/USD"]
PARITY_PAIRS = [("BTC/USD", "BTC/USDC"), ("ETH/USD", "ETH/USDC")]

GRID = [(15, 60), (60, 240)]          # (L lookback minutes, H hold minutes)
K_VALUES = [2.0, 3.0]
SIGMA_WINDOW_MIN = 7 * 1440           # 7 days of minutes
SIGMA_MIN_PERIODS = 2 * 1440          # 2 days before first signal

MAKER_RT = 2 * 0.0060                 # starter-tier maker, round trip
TAKER_RT = 2 * 0.0120                 # starter-tier taker, round trip

RNG = np.random.default_rng(42)
N_BOOT = 10_000


# ---------------------------------------------------------------- data loading

def load_candles(symbol: str) -> pd.DataFrame:
    """Load 1m candles onto a continuous minute grid (gaps stay NaN)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT timestamp, open, high, low, close, volume FROM candles "
        "WHERE symbol = ? AND timeframe = '1m' ORDER BY timestamp",
        conn, params=(symbol,))
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    full = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
    return df.reindex(full)


def load_spreads() -> dict[str, float]:
    """Per-pair median top-of-book spread from ticker parquet, floored at 1bp."""
    spreads: dict[str, float] = {}
    for pair in SIGNAL_PAIRS:
        key = f"ticker_{pair.replace('/', '_')}"
        files = sorted(RAW_DIR.glob(f"{key}*.parquet"))
        if not files:
            spreads[pair] = 0.0010  # conservative fallback: 10bp
            continue
        frames = [pd.read_parquet(f, columns=["best_bid", "best_ask"]) for f in files]
        df = pd.concat(frames)
        df = df[(df["best_bid"] > 0) & (df["best_ask"] > df["best_bid"])]
        mid = (df["best_ask"] + df["best_bid"]) / 2
        measured = float(((df["best_ask"] - df["best_bid"]) / mid).median())
        spreads[pair] = max(measured, 0.0001)
    return spreads


# ---------------------------------------------------------------- strategies

def sequential_trades(signal: np.ndarray, opens: np.ndarray, hold: int,
                      timestamps: pd.DatetimeIndex) -> list[dict]:
    """Non-overlapping long trades: signal at t -> open[t+1] .. open[t+1+hold]."""
    trades = []
    n = len(signal)
    idx = np.flatnonzero(signal)
    next_free = 0
    for t in idx:
        if t < next_free:
            continue
        e, x = t + 1, t + 1 + hold
        if x >= n:
            break
        p_in, p_out = opens[e], opens[x]
        if not (np.isfinite(p_in) and np.isfinite(p_out)) or p_in <= 0:
            continue
        trades.append({
            "entry_ts": str(timestamps[e]),
            "month": str(timestamps[e])[:7],
            "gross": p_out / p_in - 1.0,
        })
        next_free = x
    return trades


def run_family(candles: dict[str, pd.DataFrame], spreads: dict[str, float],
               family: str) -> list[dict]:
    """Run one family (mr | mom) over the pre-registered grid."""
    results = []
    for (L, H) in GRID:
        for k in K_VALUES:
            all_trades: list[dict] = []
            per_pair = {}
            for pair, df in candles.items():
                logc = np.log(df["close"].to_numpy())
                r_L = pd.Series(logc).diff(L)
                sigma = r_L.rolling(SIGMA_WINDOW_MIN,
                                    min_periods=SIGMA_MIN_PERIODS).std()
                sigma_np, r_np = sigma.to_numpy(), r_L.to_numpy()
                valid = np.isfinite(sigma_np) & (sigma_np > 0) & np.isfinite(r_np)
                if family == "mr":
                    sig = valid & (r_np <= -k * sigma_np)
                else:
                    sig = valid & (r_np >= k * sigma_np)
                trades = sequential_trades(sig, df["open"].to_numpy(), H, df.index)
                for tr in trades:
                    tr["pair"] = pair
                    tr["net_spread"] = tr["gross"] - spreads[pair]
                    tr["net_maker"] = tr["gross"] - MAKER_RT
                    tr["net_taker"] = tr["gross"] - TAKER_RT - spreads[pair]
                all_trades.extend(trades)
                if trades:
                    g = np.array([t["gross"] for t in trades])
                    per_pair[pair] = {"n": len(g),
                                      "mean_gross_bps": round(float(g.mean()) * 1e4, 2)}
            results.append(summarize_config(family, L, H, k, all_trades, per_pair))
    return results


def summarize_config(family: str, L: int, H: int, k: float,
                     trades: list[dict], per_pair: dict) -> dict:
    out = {"family": family, "L_min": L, "H_min": H, "k": k,
           "n_trades": len(trades), "per_pair": per_pair}
    if not trades:
        return out
    gross = np.array([t["gross"] for t in trades])
    months = pd.Series([t["month"] for t in trades])

    boots = RNG.choice(gross, size=(N_BOOT, len(gross)), replace=True).mean(axis=1)
    ci = np.percentile(boots, [2.5, 97.5])

    monthly = pd.Series(gross).groupby(months.values).mean()
    out.update({
        "mean_gross_bps": round(float(gross.mean()) * 1e4, 2),
        "median_gross_bps": round(float(np.median(gross)) * 1e4, 2),
        "gross_ci95_bps": [round(float(c) * 1e4, 2) for c in ci],
        "win_rate": round(float((gross > 0).mean()), 3),
        "months_positive": f"{int((monthly > 0).sum())}/{len(monthly)}",
        "monthly_mean_gross_bps": {m: round(v * 1e4, 1) for m, v in monthly.items()},
        "mean_net_bps": {
            "spread_only": round(float(np.mean([t["net_spread"] for t in trades])) * 1e4, 2),
            "maker_starter": round(float(np.mean([t["net_maker"] for t in trades])) * 1e4, 2),
            "taker_starter": round(float(np.mean([t["net_taker"] for t in trades])) * 1e4, 2),
        },
        "total_net_pct": {
            "spread_only": round(float(np.sum([t["net_spread"] for t in trades])) * 100, 1),
            "maker_starter": round(float(np.sum([t["net_maker"] for t in trades])) * 100, 1),
            "taker_starter": round(float(np.sum([t["net_taker"] for t in trades])) * 100, 1),
        },
    })
    # Pre-registered pass checks
    net_spread = np.array([t["net_spread"] for t in trades])
    net_maker = np.array([t["net_maker"] for t in trades])
    frac_pos = (monthly > 0).mean()
    for name, arr in [("primary_maker", net_maker), ("secondary_spread", net_spread)]:
        b = RNG.choice(arr, size=(N_BOOT, len(arr)), replace=True).mean(axis=1)
        lo = float(np.percentile(b, 2.5))
        out[f"pass_{name}"] = bool(
            len(arr) >= 200 and arr.mean() > 0 and lo > 0 and frac_pos >= 0.6)
    return out


# ---------------------------------------------------------------- diagnostics

def hour_of_day(candles: dict[str, pd.DataFrame]) -> dict:
    """F3: mean 1h return by UTC hour, pooled across pairs, monthly consistency."""
    rows = []
    for pair, df in candles.items():
        h = df["close"].resample("1h").last().ffill(limit=2)
        r = h.pct_change()
        rows.append(pd.DataFrame({"ret": r, "hour": r.index.hour,
                                  "month": r.index.strftime("%Y-%m")}))
    allr = pd.concat(rows).dropna()
    by_hour = allr.groupby("hour")["ret"].agg(["mean", "count"])
    by_hour_month = allr.groupby(["hour", "month"])["ret"].mean().unstack()
    consistency = (by_hour_month > 0).mean(axis=1)
    return {
        "mean_ret_bps_by_utc_hour": {int(h): round(v * 1e4, 2)
                                     for h, v in by_hour["mean"].items()},
        "frac_months_positive_by_hour": {int(h): round(float(v), 2)
                                         for h, v in consistency.items()},
        "note": "pooled across 8 USD pairs; diagnostic only, no cost model",
    }


def usdc_parity() -> dict:
    """F4: minute-close basis between USD and USDC listings."""
    out = {}
    for usd_sym, usdc_sym in PARITY_PAIRS:
        a = load_candles(usd_sym)["close"]
        b = load_candles(usdc_sym)["close"]
        both = pd.DataFrame({"usd": a, "usdc": b}).dropna()
        basis = (both["usd"] / both["usdc"] - 1.0)
        bps = basis * 1e4
        exceed = {f"gt_{t}bp": round(float((bps.abs() > t).mean()), 5)
                  for t in (5, 10, 20, 50)}
        # persistence of >10bp episodes
        mask = (bps.abs() > 10).to_numpy()
        runs, cur = [], 0
        for m in mask:
            if m:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        out[usd_sym.split("/")[0]] = {
            "n_minutes": int(len(bps)),
            "mean_bps": round(float(bps.mean()), 3),
            "std_bps": round(float(bps.std()), 3),
            "median_abs_bps": round(float(bps.abs().median()), 3),
            "p99_abs_bps": round(float(bps.abs().quantile(0.99)), 2),
            "max_abs_bps": round(float(bps.abs().max()), 1),
            "frac_minutes_exceeding": exceed,
            "episodes_gt10bp": len(runs),
            "median_episode_minutes": float(np.median(runs)) if runs else 0.0,
        }
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    print("Loading spreads...", flush=True)
    spreads = load_spreads()
    for p, s in spreads.items():
        print(f"  {p}: {s * 1e4:.1f} bps")

    print("Loading candles...", flush=True)
    candles = {p: load_candles(p) for p in SIGNAL_PAIRS}
    for p, df in candles.items():
        print(f"  {p}: {df['close'].notna().sum():,} bars "
              f"({df.index[0].date()} -> {df.index[-1].date()})")

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "design": "pre-registered; see module docstring",
        "spreads_bps": {p: round(s * 1e4, 1) for p, s in spreads.items()},
        "cost_scenarios_rt_bps": {"maker_starter": MAKER_RT * 1e4,
                                  "taker_starter_plus_spread": TAKER_RT * 1e4},
    }

    print("F1 mean reversion...", flush=True)
    report["F1_mean_reversion"] = run_family(candles, spreads, "mr")
    print("F2 momentum...", flush=True)
    report["F2_momentum"] = run_family(candles, spreads, "mom")
    print("F3 hour-of-day...", flush=True)
    report["F3_hour_of_day"] = hour_of_day(candles)
    print("F4 USD/USDC parity...", flush=True)
    report["F4_usdc_parity"] = usdc_parity()

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_PATH}")

    # Console verdict
    print("\n=== F1/F2 pooled configs ===")
    for fam_key in ("F1_mean_reversion", "F2_momentum"):
        for c in report[fam_key]:
            if c["n_trades"] == 0:
                continue
            print(f"{fam_key[:2]} L={c['L_min']:>3} H={c['H_min']:>3} k={c['k']}: "
                  f"n={c['n_trades']:>5}  gross={c['mean_gross_bps']:>7.2f}bps "
                  f"CI={c['gross_ci95_bps']}  months+={c['months_positive']}  "
                  f"net(maker)={c['mean_net_bps']['maker_starter']:>8.2f}bps  "
                  f"PASS1={c['pass_primary_maker']} PASS2={c['pass_secondary_spread']}")


if __name__ == "__main__":
    sys.exit(main())
