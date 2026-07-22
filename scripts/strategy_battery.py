#!/usr/bin/env python3
"""FINAL exhaustive strategy battery across all fee-viable, long-only,
single-venue mechanism families. PRE-REGISTERED: configs fixed before
running, everything reported, no tuning on results.

Universes
---------
A: 66 high-amplitude USD alts, ~2y hourly candles (data/external/h1/).
B: 8 majors, daily candles back to 2016 where listed (fetched+cached).

Costs: 120 bps maker round trip (verified account tier). Rule families are
parameter-fixed (no fitting), so the full sample is out-of-fitting; the
remaining risk is family-selection, handled by the pass bar below. ML
families are walk-forward by quarter.

Families (all long-only, sequential non-overlapping per pair)
-------------------------------------------------------------
A-universe (hourly, entry next bar open):
  F1  Donchian breakout 20d high -> exit 10d low or 21d cap
  F2  Donchian breakout 55d high -> exit 20d low or 42d cap
  F3  Trend pullback: close > 30d MA and r72h <= -1.5 sigma72 -> hold 72h
  F4  Vol-compression breakout: 10d vol in bottom 20% of trailing 90d
      and r24 >= +1 sigma24 -> hold 120h
  F5  New-listing momentum: age < 60d and r168 > 0 -> hold 168h
  F6  7d crash reversal: r168 <= -2 sigma168 -> hold 168h
  F7  BTC lead-lag catch-up: BTC r24 >= +2 sigma(BTC) and pair r24 <
      half BTC's -> hold 72h
  F8  Volume anomaly: 24h vol >= 3x 30d median and |r24| < 1 sigma -> hold 72h
  F9  Red streak: 5 consecutive daily closes down -> hold 72h
  F10 Cross-sectional momentum: weekly rotate into top-5 by trailing 7d
      return, hold 168h (full round-trip cost charged per week)
  F11 Cross-sectional reversal: same, bottom-5
  BENCH 24h crash rule (z<=-3, hold 72h) for reference
B-universe (daily, entry next day open):
  F12a Donchian 55d/20d   F12b Donchian 20d/10d   F12c 200d MA trend
       (buy cross above, exit cross below, cap 120d)
ML (A-universe, 6-hourly panel, walk-forward quarterly from 2025Q1,
purge 216h; realistic 5-slot execution taking only predictions > fee):
  F13 LightGBM pooled regression on ~20 features, hold 72h
  F14 Ridge linear, same features/loop
Diagnostic: day-of-week mean 24h returns (no pass claim).

PASS BAR (fixed): n>=100, pooled mean net > 0, weekly-cluster 95% CI > 0,
>=60% of traded months positive, AND both sample halves positive.
With ~14 families, ~1 borderline pass is expected by chance; treat any
pass without margin (CI lower bound > +50bps) as provisional.

Output: research/strategy_battery_results.json
"""

import json
import sys
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
H1_DIR = PROJECT_ROOT / "data" / "external" / "h1"
D1_DIR = PROJECT_ROOT / "data" / "external" / "d1"
WIDE_RESULTS = PROJECT_ROOT / "research" / "swing_screen_wide_results.json"
INV_RESULTS = PROJECT_ROOT / "research" / "swing_screen_results.json"
OUT_PATH = PROJECT_ROOT / "research" / "strategy_battery_results.json"

MAKER_RT = 0.0120
MAJORS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD",
          "DOGE/USD", "LINK/USD", "LTC/USD", "BCH/USD"]
RNG = np.random.default_rng(23)
N_BOOT = 10_000

D = 24  # hours per day


# ------------------------------------------------------------------ loading

def load_alts() -> dict[str, pd.DataFrame]:
    pairs = json.loads(WIDE_RESULTS.read_text())["phase2_selected"]
    out = {}
    for sym in pairs:
        f = H1_DIR / f"{sym.replace('/', '_')}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        full = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
        out[sym] = df.reindex(full)
    return out


def load_btc_hourly() -> pd.DataFrame:
    df = pd.read_parquet(H1_DIR / "BTC_USD.parquet")
    full = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
    return df.reindex(full)


def fetch_majors_daily() -> dict[str, pd.DataFrame]:
    """Daily candles back to 2016 (or listing), cached."""
    D1_DIR.mkdir(parents=True, exist_ok=True)
    ex = ccxt.coinbase({"enableRateLimit": True})
    out = {}
    for sym in MAJORS:
        cache = D1_DIR / f"{sym.replace('/', '_')}.parquet"
        if cache.exists():
            out[sym] = pd.read_parquet(cache)
            continue
        since = ex.parse8601("2016-01-01T00:00:00Z")
        rows, cur, guard = [], since, 0
        now = ex.milliseconds()
        while cur < now and guard < 60:
            guard += 1
            try:
                batch = ex.fetch_ohlcv(sym, "1d", since=cur, limit=300)
            except Exception:
                time.sleep(1)
                batch = []
            if not batch:
                cur += 300 * 86_400_000
                continue
            rows.extend(batch)
            nxt = batch[-1][0] + 86_400_000
            if nxt <= cur:
                break
            cur = nxt
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("ts").sort_values("ts")
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns=["ts"])
        df.to_parquet(cache)
        out[sym] = df
        print(f"  fetched {sym}: {len(df)}d from {df.index[0].date()}")
    return out


# ------------------------------------------------------------------ engine

def trades_from_mask(df: pd.DataFrame, entry: np.ndarray, hold: int,
                     exit_mask: np.ndarray | None = None,
                     pair: str = "") -> list[dict]:
    """Sequential long trades. Signal at bar t -> enter open[t+1].
    Exit at open[x+1] where x = first bar >= entry with exit_mask, else
    entry_bar + hold. Returns gross open-to-open."""
    o = df["open"].to_numpy()
    idx = df.index
    n = len(o)
    trades, nxt = [], 0
    for t in np.flatnonzero(entry):
        if t < nxt:
            continue
        e = t + 1
        x_cap = e + hold
        if e >= n:
            break
        x = x_cap
        if exit_mask is not None:
            hits = np.flatnonzero(exit_mask[e:min(x_cap, n - 1)])
            if len(hits):
                x = e + hits[0] + 1
        if x >= n:
            break
        if not (np.isfinite(o[e]) and np.isfinite(o[x])) or o[e] <= 0:
            continue
        trades.append({"pair": pair, "entry_ts": idx[e],
                       "gross": o[x] / o[e] - 1.0,
                       "hold_bars": int(x - e)})
        nxt = x
    return trades


def evaluate(name: str, trades: list[dict], cluster: str = "W") -> dict:
    out = {"family": name, "n": len(trades)}
    if len(trades) < 10:
        out["note"] = "too few trades"
        return out
    t = pd.DataFrame(trades)
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], utc=True)
    t["net"] = t["gross"] - MAKER_RT
    key = t["entry_ts"].dt.strftime("%G-W%V" if cluster == "W" else "%Y-%m")
    t["month"] = t["entry_ts"].dt.strftime("%Y-%m")

    groups = [g.to_numpy() for _, g in t.groupby(key.values)["net"]]
    k = len(groups)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.integers(0, k, k)
        means[b] = np.concatenate([groups[i] for i in pick]).mean()
    ci = [float(np.percentile(means, q)) for q in (2.5, 97.5)]

    monthly = t.groupby("month")["net"].mean()
    mid = t["entry_ts"].min() + (t["entry_ts"].max() - t["entry_ts"].min()) / 2
    h1 = t[t["entry_ts"] <= mid]["net"]
    h2 = t[t["entry_ts"] > mid]["net"]
    frac_pos = float((monthly > 0).mean())

    out.update({
        "mean_gross_bps": round(float(t["gross"].mean()) * 1e4, 1),
        "mean_net_bps": round(float(t["net"].mean()) * 1e4, 1),
        "net_ci95_bps": [round(c * 1e4, 1) for c in ci],
        "p_leq0": round(float((means <= 0).mean()), 4),
        "months_pos": f"{int((monthly > 0).sum())}/{len(monthly)}",
        "half1_net_bps": round(float(h1.mean()) * 1e4, 1) if len(h1) else None,
        "half2_net_bps": round(float(h2.mean()) * 1e4, 1) if len(h2) else None,
        "median_hold_bars": float(t["hold_bars"].median()) if "hold_bars" in t else None,
        "total_net_pct": round(float(t["net"].sum()) * 100, 1),
        "n_pairs": int(t["pair"].nunique()),
        "pass": bool(len(t) >= 100 and t["net"].mean() > 0 and ci[0] > 0
                     and frac_pos >= 0.6
                     and len(h1) and len(h2)
                     and h1.mean() > 0 and h2.mean() > 0),
    })
    return out


# ------------------------------------------------------------------ families

def family_signals(alts: dict[str, pd.DataFrame],
                   btc: pd.DataFrame) -> list[dict]:
    inv = {r["pair"]: r for r in json.loads(INV_RESULTS.read_text())["phase1_inventory"]}
    btc_logc = np.log(btc["close"])
    btc_r24 = btc_logc.diff(24)
    btc_sig24 = btc_r24.rolling(30 * D, min_periods=10 * D).std()

    fams: dict[str, list[dict]] = {k: [] for k in
        ["F1_donchian_20_10", "F2_donchian_55_20", "F3_trend_pullback",
         "F4_vol_squeeze_breakout", "F5_new_listing_momo", "F6_7d_crash",
         "F7_btc_leadlag", "F8_volume_anomaly", "F9_red_streak",
         "BENCH_24h_crash_rule"]}

    for sym, df in alts.items():
        c, v = df["close"], df["volume"]
        logc = np.log(c)
        r24, r72, r168 = logc.diff(24), logc.diff(72), logc.diff(168)
        s24 = r24.rolling(30 * D, min_periods=10 * D).std()
        s72 = r72.rolling(30 * D, min_periods=10 * D).std()
        s168 = r168.rolling(30 * D, min_periods=10 * D).std()
        z24 = (r24 / s24).to_numpy()

        hi20, lo10 = c.rolling(20 * D).max(), c.rolling(10 * D).min()
        hi55, lo20 = c.rolling(55 * D).max(), c.rolling(20 * D).min()
        ma30d = c.rolling(30 * D, min_periods=10 * D).mean()

        r1h = logc.diff()
        vol10 = r1h.rolling(10 * D).std()
        vol10_rank = vol10.rolling(90 * D, min_periods=30 * D) \
                          .rank(pct=True)

        vol24 = v.rolling(24).sum()
        vratio = vol24 / vol24.rolling(30 * D, min_periods=10 * D).median()

        cd = c.resample("1D").last()
        red = (cd.diff() < 0).rolling(5).sum() == 5
        red_h = red.reindex(df.index, method="ffill").fillna(False)
        red_edge = red_h & ~red_h.shift(1, fill_value=False)

        b24 = btc_r24.reindex(df.index)
        bs24 = btc_sig24.reindex(df.index)

        age_days = (df.index - df.index[0]).days

        def m(x):  # to numpy with nan->False
            arr = np.asarray(x, dtype=float)
            return np.isfinite(arr) & (arr > 0)

        e1 = m((c >= hi20).astype(float) * (c.shift(1) < hi20.shift(1)).astype(float))
        x1 = m((c <= lo10).astype(float))
        fams["F1_donchian_20_10"].extend(
            trades_from_mask(df, e1, 21 * D, x1, sym))

        e2 = m((c >= hi55).astype(float) * (c.shift(1) < hi55.shift(1)).astype(float))
        x2 = m((c <= lo20).astype(float))
        fams["F2_donchian_55_20"].extend(
            trades_from_mask(df, e2, 42 * D, x2, sym))

        e3 = m((c > ma30d).astype(float)
               * (r72 <= -1.5 * s72).astype(float))
        fams["F3_trend_pullback"].extend(trades_from_mask(df, e3, 72, None, sym))

        e4 = m((vol10_rank < 0.2).astype(float) * (r24 >= s24).astype(float))
        fams["F4_vol_squeeze_breakout"].extend(
            trades_from_mask(df, e4, 120, None, sym))

        e5 = m((r168 > 0).astype(float)) & (age_days < 60) & (age_days >= 8)
        fams["F5_new_listing_momo"].extend(trades_from_mask(df, e5, 168, None, sym))

        e6 = m((r168 <= -2 * s168).astype(float))
        fams["F6_7d_crash"].extend(trades_from_mask(df, e6, 168, None, sym))

        e7 = m((b24 >= 2 * bs24).astype(float) * (r24 < b24 / 2).astype(float))
        fams["F7_btc_leadlag"].extend(trades_from_mask(df, e7, 72, None, sym))

        e8 = m((vratio >= 3).astype(float) * (r24.abs() < s24).astype(float))
        fams["F8_volume_anomaly"].extend(trades_from_mask(df, e8, 72, None, sym))

        fams["F9_red_streak"].extend(
            trades_from_mask(df, red_edge.to_numpy(), 72, None, sym))

        e_b = np.isfinite(z24) & (z24 <= -3)
        fams["BENCH_24h_crash_rule"].extend(trades_from_mask(df, e_b, 72, None, sym))

    return [evaluate(k, v) for k, v in fams.items()]


def cross_sectional(alts: dict[str, pd.DataFrame]) -> list[dict]:
    closes = pd.DataFrame({s: df["close"] for s, df in alts.items()})
    opens = pd.DataFrame({s: df["open"] for s, df in alts.items()})
    r168 = np.log(closes).diff(168)
    mondays = [ts for ts in closes.index
               if ts.weekday() == 0 and ts.hour == 0]
    top_tr, bot_tr = [], []
    for ts in mondays:
        loc = closes.index.get_loc(ts)
        if loc + 1 + 168 >= len(closes) or loc < 168:
            continue
        row = r168.iloc[loc].dropna()
        if len(row) < 10:
            continue
        e, x = loc + 1, loc + 1 + 168
        for sel, bucket in ((row.nlargest(5).index, top_tr),
                            (row.nsmallest(5).index, bot_tr)):
            for sym in sel:
                p_in, p_out = opens[sym].iloc[e], opens[sym].iloc[x]
                if np.isfinite(p_in) and np.isfinite(p_out) and p_in > 0:
                    bucket.append({"pair": sym, "entry_ts": closes.index[e],
                                   "gross": p_out / p_in - 1.0, "hold_bars": 168})
    return [evaluate("F10_xs_momentum_top5", top_tr),
            evaluate("F11_xs_reversal_bottom5", bot_tr)]


def majors_daily(majors: dict[str, pd.DataFrame]) -> list[dict]:
    f12a, f12b, f12c = [], [], []
    for sym, df in majors.items():
        c = df["close"]
        hi55, lo20 = c.rolling(55).max(), c.rolling(20).min()
        hi20, lo10 = c.rolling(20).max(), c.rolling(10).min()
        ma200 = c.rolling(200).mean()

        def m(x):
            arr = np.asarray(x, dtype=float)
            return np.isfinite(arr) & (arr > 0)

        e_a = m((c >= hi55).astype(float) * (c.shift(1) < hi55.shift(1)).astype(float))
        f12a.extend(trades_from_mask(df, e_a, 120, m((c <= lo20).astype(float)), sym))
        e_b = m((c >= hi20).astype(float) * (c.shift(1) < hi20.shift(1)).astype(float))
        f12b.extend(trades_from_mask(df, e_b, 60, m((c <= lo10).astype(float)), sym))
        above = (c > ma200)
        e_c = m(above.astype(float) * (~above.shift(1, fill_value=False)).astype(float))
        f12c.extend(trades_from_mask(df, e_c, 120,
                                     m((~above).astype(float)), sym))
    return [evaluate("F12a_majors_donchian_55_20", f12a, cluster="M"),
            evaluate("F12b_majors_donchian_20_10", f12b, cluster="M"),
            evaluate("F12c_majors_200dma_trend", f12c, cluster="M")]


# ------------------------------------------------------------------ pooled ML

def build_panel(alts: dict[str, pd.DataFrame], btc: pd.DataFrame) -> pd.DataFrame:
    inv = {r["pair"]: r for r in json.loads(INV_RESULTS.read_text())["phase1_inventory"]}
    spreads = json.loads(WIDE_RESULTS.read_text())["phase2_screen"]["spreads_bps"]
    btc_logc = np.log(btc["close"])
    frames = []
    for sym, df in alts.items():
        c, v, o = df["close"], df["volume"], df["open"].to_numpy()
        logc = np.log(c)
        feat = pd.DataFrame(index=df.index)
        for h in (6, 24, 72, 168, 504):
            feat[f"r{h}"] = logc.diff(h)
        feat["sigma24"] = feat["r24"].rolling(30 * D, min_periods=10 * D).std()
        feat["sigma168"] = feat["r168"].rolling(30 * D, min_periods=10 * D).std()
        vol24 = v.rolling(24).sum()
        feat["vratio"] = vol24 / vol24.rolling(30 * D, min_periods=10 * D).median()
        feat["dd30"] = c / c.rolling(30 * D, min_periods=10 * D).max() - 1
        feat["ma30_dist"] = c / c.rolling(30 * D, min_periods=10 * D).mean() - 1
        b = btc_logc.reindex(df.index)
        feat["btc_r24"] = b.diff(24)
        feat["btc_r168"] = b.diff(168)
        feat["idio24"] = feat["r24"] - feat["btc_r24"]
        feat["age_d"] = (df.index - df.index[0]).days
        feat["amp3d"] = inv.get(sym, {}).get("mean_abs_r3d_pct", np.nan)
        feat["spread"] = spreads.get(sym, 20.0)
        feat["dow"] = df.index.dayofweek
        feat["hour"] = df.index.hour
        # label: 72h forward open-to-open from next bar
        n = len(df)
        fwd = np.full(n, np.nan)
        e = np.arange(n - 73)
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd[: n - 73] = o[e + 73] / o[e + 1] - 1.0
        feat["fwd72"] = fwd
        feat["pair"] = sym
        feat["ts"] = df.index
        frames.append(feat.iloc[::6])  # 6-hourly subsample
    panel = pd.concat(frames).dropna(subset=["fwd72", "r24", "sigma24"])
    panel["quarter"] = panel["ts"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    return panel.reset_index(drop=True)


def ml_family(panel: pd.DataFrame, kind: str) -> dict:
    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    feats = [c for c in panel.columns
             if c not in ("fwd72", "pair", "ts", "quarter")]
    quarters = sorted(q for q in panel["quarter"].unique() if q >= "2025Q1")
    panel = panel.copy()
    panel["pred"] = np.nan
    for q in quarters:
        q_start = pd.Period(q, freq="Q").start_time.tz_localize("UTC")
        cutoff = q_start - pd.Timedelta(hours=216)
        tr = panel[panel["ts"] < cutoff]
        te_idx = panel.index[panel["quarter"] == q]
        if len(tr) < 5000 or not len(te_idx):
            continue
        lo, hi = tr["fwd72"].quantile([0.05, 0.95])
        y = tr["fwd72"].clip(lo, hi)
        if kind == "lgb":
            mdl = lgb.LGBMRegressor(
                objective="regression", n_estimators=300, learning_rate=0.03,
                num_leaves=15, max_depth=4, min_child_samples=200,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                verbosity=-1, seed=7)
            mdl.fit(tr[feats], y)
            panel.loc[te_idx, "pred"] = mdl.predict(panel.loc[te_idx, feats])
        else:
            med = tr[feats].median()
            Xtr = tr[feats].fillna(med)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
            mdl = Ridge(alpha=1.0)
            mdl.fit((Xtr - mu) / sd, y)
            Xte = panel.loc[te_idx, feats].fillna(med)
            panel.loc[te_idx, "pred"] = mdl.predict((Xte - mu) / sd)

    oos = panel[panel["pred"].notna()].sort_values("ts")
    # realistic 5-slot loop: at each 6h step take fee-clearing top preds
    busy_until: dict[str, pd.Timestamp] = {}
    open_slots = 5
    active: list[tuple[pd.Timestamp, str]] = []
    trades = []
    for ts, grp in oos.groupby("ts"):
        active = [(end, p) for end, p in active if end > ts]
        cands = grp[grp["pred"] >= MAKER_RT].sort_values("pred", ascending=False)
        for _, row in cands.iterrows():
            if len(active) >= open_slots:
                break
            p = row["pair"]
            if any(pp == p for _, pp in active):
                continue
            trades.append({"pair": p, "entry_ts": ts, "gross": row["fwd72"],
                           "hold_bars": 72})
            active.append((ts + pd.Timedelta(hours=72), p))
    return evaluate(f"F13_pooled_{kind}", trades)


def day_of_week(alts: dict[str, pd.DataFrame]) -> dict:
    rows = []
    for sym, df in alts.items():
        cd = df["close"].resample("1D").last()
        r = cd.pct_change()
        rows.append(pd.DataFrame({"ret": r, "dow": r.index.dayofweek}))
    allr = pd.concat(rows).dropna()
    by = allr.groupby("dow")["ret"].mean() * 1e4
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {"mean_daily_ret_bps_by_dow": {names[i]: round(float(v), 1)
                                          for i, v in by.items()},
            "note": "diagnostic only"}


def main() -> None:
    print("Loading universes...", flush=True)
    alts = load_alts()
    btc = load_btc_hourly()
    majors = fetch_majors_daily()
    print(f"  {len(alts)} alts hourly, {len(majors)} majors daily", flush=True)

    results = []
    print("Rule families (alts)...", flush=True)
    results += family_signals(alts, btc)
    print("Cross-sectional...", flush=True)
    results += cross_sectional(alts)
    print("Majors daily trend...", flush=True)
    results += majors_daily(majors)
    print("Pooled ML panel...", flush=True)
    panel = build_panel(alts, btc)
    print(f"  panel: {len(panel):,} rows", flush=True)
    for kind in ("lgb", "ridge"):
        print(f"  ML {kind}...", flush=True)
        results.append(ml_family(panel, kind))
    dow = day_of_week(alts)

    report = {"design": "pre-registered; see docstring",
              "results": results, "day_of_week": dow}
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}\n")

    print(f"{'family':<28} {'n':>5} {'gross':>8} {'net':>8} "
          f"{'CI95':>20} {'mo+':>6} {'h1':>7} {'h2':>7} pass")
    for r in results:
        if "mean_net_bps" not in r:
            print(f"{r['family']:<28} {r['n']:>5}  (too few trades)")
            continue
        print(f"{r['family']:<28} {r['n']:>5} {r['mean_gross_bps']:>8.1f} "
              f"{r['mean_net_bps']:>8.1f} {str(r['net_ci95_bps']):>20} "
              f"{r['months_pos']:>6} {r['half1_net_bps']:>7.1f} "
              f"{r['half2_net_bps']:>7.1f} {r['pass']}")
    print("\nday-of-week:", dow["mean_daily_ret_bps_by_dow"])


if __name__ == "__main__":
    sys.exit(main())
