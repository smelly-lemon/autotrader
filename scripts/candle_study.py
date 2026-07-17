#!/usr/bin/env python3
"""Fee-survivability study: can anything on 4h/1d candle bars beat real fees?

PRE-REGISTERED DESIGN (fixed before running — do not tune on results):
- Data: 1m OHLCV candles (SQLite), all 12 pairs, Nov 5 2025 -> present,
  resampled to 4h and 1d bars.
- Model: pooled LightGBM classifier, fixed hyperparameters, no tuning.
- Configs: {4h bars: 6, 12 bar horizons} x {1d bars: 2, 5 bar horizons},
  long-only, fixed-horizon exits, one position per pair.
- Validation: expanding-window OOS. Test months Feb..Jul 2026, train on
  everything before the window minus a purge gap of `horizon` bars.
- Costs: measured per-pair spread + fee scenarios (starter taker 1.2%/side,
  starter maker 0.6%/side, silver taker 0.4%/side).
- PRIMARY pass criterion: at starter-taker fees, threshold 0.55, pooled
  across all pairs and windows: total net > 0 AND bootstrap 95% CI of mean
  per-trade net return excludes zero. Everything else is diagnostics.
- Baseline: equal-weight buy-and-hold over the same OOS period minus one
  round-trip of costs.

Usage:
    python scripts/candle_study.py                  # full study
    python scripts/candle_study.py --pairs BTC/USD --windows 1   # smoke test
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.influx_client import PRODUCT_IDS
from src.data.store import TradeStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "candle_study"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

BAR_CONFIGS: list[tuple[str, list[int]]] = [("4h", [6, 12]), ("1D", [2, 5])]
THRESHOLDS = [0.55, 0.60]  # 0.55 is the pre-registered primary
PRIMARY = {"bar": "4h+1d", "threshold": 0.55, "fees": "starter_taker"}

FEE_PER_SIDE = {
    "starter_taker": 0.012,
    "starter_maker": 0.006,
    "silver_taker": 0.004,
}
DEFAULT_SPREAD = 0.0010  # fallback relative spread if no ticker data for a pair

TEST_WINDOWS = [
    ("2026-02-01", "2026-03-01"),
    ("2026-03-01", "2026-04-01"),
    ("2026-04-01", "2026-05-01"),
    ("2026-05-01", "2026-06-01"),
    ("2026-06-01", "2026-07-01"),
    ("2026-07-01", "2026-07-17"),
]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "verbose": -1,
    "seed": 42,
}
NUM_ROUNDS = 200  # fixed, no early stopping — nothing to peek at


def measure_spreads(pairs: list[str]) -> dict[str, float]:
    """Median relative bid-ask spread per pair from live-collected ticker shards."""
    spreads: dict[str, float] = {}
    for pair in pairs:
        safe = pair.replace("/", "_")
        files = sorted(glob.glob(str(RAW_DIR / f"ticker_{safe}.*.parquet")))
        if not files:
            spreads[pair] = DEFAULT_SPREAD
            continue
        df = pd.concat(pd.read_parquet(f) for f in files[-3:])
        df = df[(df["best_bid"] > 0) & (df["best_ask"] > 0)]
        if df.empty:
            spreads[pair] = DEFAULT_SPREAD
            continue
        mid = (df["best_ask"] + df["best_bid"]) / 2
        measured = float(((df["best_ask"] - df["best_bid"]) / mid).median())
        # Floor at 1bp: REST top-of-book snapshots understate realized taker cost.
        spreads[pair] = max(measured, 0.0001)
    return spreads


def resample_bars(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    bars = df_1m.resample(rule).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).dropna(subset=["close"])
    return bars


def build_features(bars: pd.DataFrame, btc_bars: pd.DataFrame | None) -> pd.DataFrame:
    c = bars["close"]
    r1 = np.log(c / c.shift(1))
    f = pd.DataFrame(index=bars.index)
    for n in (1, 2, 3, 6, 12, 24):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    f["vol_6"] = r1.rolling(6).std()
    f["vol_24"] = r1.rolling(24).std()
    f["vol_ratio"] = f["vol_6"] / f["vol_24"]

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = c.ewm(span=12).mean()
    ema48 = c.ewm(span=48).mean()
    f["c_ema12"] = c / ema12 - 1
    f["c_ema48"] = c / ema48 - 1
    f["ema_cross"] = ema12 / ema48 - 1

    rng = (bars["high"] - bars["low"]) / c
    f["range_1"] = rng
    f["range_6"] = rng.rolling(6).mean()

    lv = np.log1p(bars["volume"])
    f["vol_z"] = (lv - lv.rolling(24).mean()) / lv.rolling(24).std()

    f["hour"] = bars.index.hour
    f["dow"] = bars.index.dayofweek

    if btc_bars is not None:
        bc = btc_bars["close"].reindex(bars.index).ffill()
        for n in (1, 2, 6):
            f[f"btc_ret_{n}"] = np.log(bc / bc.shift(n))
    return f


def build_dataset(
    candles: dict[str, pd.DataFrame], rule: str, horizon: int,
) -> pd.DataFrame:
    """Pooled dataset: features + forward simple return over `horizon` bars."""
    btc_bars = resample_bars(candles["BTC/USD"], rule) if "BTC/USD" in candles else None
    rows = []
    for i, (pair, df_1m) in enumerate(sorted(candles.items())):
        bars = resample_bars(df_1m, rule)
        if len(bars) < 60:
            continue
        f = build_features(bars, btc_bars if pair != "BTC/USD" else None)
        c = bars["close"]
        f["fwd_ret"] = c.shift(-horizon) / c - 1
        f["y"] = (f["fwd_ret"] > 0).astype(int)
        f["pair"] = pair
        f["pair_id"] = i
        rows.append(f.dropna(subset=["fwd_ret", "ret_24"]))
    data = pd.concat(rows).sort_index()
    return data


FEATURE_EXCLUDE = {"fwd_ret", "y", "pair"}


def run_window(
    data: pd.DataFrame, train_end: pd.Timestamp, test_start: pd.Timestamp,
    test_end: pd.Timestamp, horizon: int, rule: str,
) -> pd.DataFrame | None:
    """Train up to train_end (purged), return test rows with predictions."""
    bar_delta = pd.Timedelta(rule if rule != "1D" else "24h")
    purge_cutoff = train_end - horizon * bar_delta

    feature_cols = [col for col in data.columns if col not in FEATURE_EXCLUDE]
    train = data[data.index < purge_cutoff]
    test = data[(data.index >= test_start) & (data.index < test_end)]
    if len(train) < 500 or test.empty:
        return None

    dtrain = lgb.Dataset(
        train[feature_cols], label=train["y"],
        categorical_feature=["pair_id", "hour", "dow"],
    )
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS)
    test = test.copy()
    test["proba"] = model.predict(test[feature_cols])
    return test


def simulate_trades(test: pd.DataFrame, threshold: float, horizon: int, rule: str) -> list[dict]:
    """Long-only, fixed-horizon exits, one open position per pair."""
    bar_delta = pd.Timedelta(rule if rule != "1D" else "24h")
    trades = []
    for pair, g in test.groupby("pair"):
        g = g.sort_index()
        busy_until = None
        for ts, row in g.iterrows():
            if busy_until is not None and ts < busy_until:
                continue
            if row["proba"] > threshold:
                trades.append({
                    "pair": pair, "ts": str(ts), "proba": float(row["proba"]),
                    "gross_ret": float(row["fwd_ret"]),
                })
                busy_until = ts + horizon * bar_delta
    return trades


def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def apply_costs(trades: list[dict], fee_per_side: float, spreads: dict[str, float]) -> np.ndarray:
    return np.array([
        t["gross_ret"] - 2 * fee_per_side - spreads.get(t["pair"], DEFAULT_SPREAD)
        for t in trades
    ])


def buy_and_hold_baseline(
    candles: dict[str, pd.DataFrame], start: str, end: str, spreads: dict[str, float],
) -> dict:
    rets = {}
    for pair, df in candles.items():
        window = df[(df.index >= start) & (df.index < end)]["close"]
        if len(window) < 100:
            continue
        gross = float(window.iloc[-1] / window.iloc[0] - 1)
        rets[pair] = gross - 2 * FEE_PER_SIDE["starter_taker"] - spreads.get(pair, DEFAULT_SPREAD)
    ew = float(np.mean(list(rets.values()))) if rets else float("nan")
    return {"per_pair_net": rets, "equal_weight_net": ew}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=None, help="Comma-separated subset (smoke tests)")
    parser.add_argument("--windows", type=int, default=len(TEST_WINDOWS),
                        help="Number of test windows to run (from the start)")
    args = parser.parse_args()

    pairs = ([p.replace("-", "/") for p in args.pairs.split(",")]
             if args.pairs else [p.replace("-", "/") for p in PRODUCT_IDS])
    windows = TEST_WINDOWS[: args.windows]

    store = TradeStore()
    candles: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        df = store.get_all_candles_df(pair, "1m")
        if len(df) > 10_000:
            candles[pair] = df
            logger.info("Loaded %s: %d 1m candles (%s -> %s)",
                        pair, len(df), str(df.index[0])[:10], str(df.index[-1])[:10])
        else:
            logger.warning("Skipping %s: only %d candles", pair, len(df))
    store.close()

    spreads = measure_spreads(list(candles))
    logger.info("Measured spreads (bps): %s",
                {p: round(s * 1e4, 1) for p, s in spreads.items()})

    report: dict = {
        "design": "see scripts/candle_study.py docstring (pre-registered)",
        "pairs": list(candles),
        "spreads": spreads,
        "windows": windows,
        "configs": {},
        "baseline": buy_and_hold_baseline(candles, windows[0][0], windows[-1][1], spreads),
    }

    for rule, horizons in BAR_CONFIGS:
        for horizon in horizons:
            data = build_dataset(candles, rule, horizon)
            all_trades: dict[float, list[dict]] = {t: [] for t in THRESHOLDS}
            window_rows: dict[float, list[dict]] = {t: [] for t in THRESHOLDS}

            for w_start, w_end in windows:
                test = run_window(
                    data, pd.Timestamp(w_start, tz="UTC"), pd.Timestamp(w_start, tz="UTC"),
                    pd.Timestamp(w_end, tz="UTC"), horizon, rule,
                )
                if test is None:
                    continue
                for thr in THRESHOLDS:
                    trades = simulate_trades(test, thr, horizon, rule)
                    all_trades[thr].extend(trades)
                    nets = apply_costs(trades, FEE_PER_SIDE["starter_taker"], spreads)
                    window_rows[thr].append({
                        "window": f"{w_start}->{w_end}", "n": len(trades),
                        "total_net_starter_taker": float(nets.sum()) if len(nets) else 0.0,
                    })

            cfg_key = f"{rule}_h{horizon}"
            report["configs"][cfg_key] = {}
            for thr in THRESHOLDS:
                trades = all_trades[thr]
                # Sign consistency across windows is the robust significance
                # check: trades within a window are cross-pair correlated, so
                # the per-trade bootstrap CI is optimistically narrow.
                wins = [w for w in window_rows[thr] if w["n"] > 0]
                entry = {"threshold": thr, "n_trades": len(trades),
                         "per_window_starter_taker": window_rows[thr],
                         "windows_positive": sum(
                             1 for w in wins if w["total_net_starter_taker"] > 0),
                         "windows_traded": len(wins),
                         "fees": {}}
                gross = np.array([t["gross_ret"] for t in trades])
                if len(trades):
                    entry["avg_gross"] = float(gross.mean())
                    entry["win_rate_gross"] = float((gross > 0).mean())
                for fee_name, fee in FEE_PER_SIDE.items():
                    nets = apply_costs(trades, fee, spreads)
                    if len(nets) == 0:
                        entry["fees"][fee_name] = {"n": 0}
                        continue
                    lo, hi = bootstrap_ci(nets)
                    entry["fees"][fee_name] = {
                        "total_net": float(nets.sum()),
                        "avg_net": float(nets.mean()),
                        "win_rate_net": float((nets > 0).mean()),
                        "ci95_mean_net": [lo, hi],
                        "significant_positive": bool(lo > 0),
                    }
                report["configs"][cfg_key][f"thr_{thr}"] = entry
                st = entry["fees"].get("starter_taker", {})
                logger.info(
                    "%s thr=%.2f: %d trades | avg gross %+.4f | starter-taker total %+.4f "
                    "avg %+.5f CI[%+.5f, %+.5f]",
                    cfg_key, thr, len(trades),
                    entry.get("avg_gross", float("nan")),
                    st.get("total_net", float("nan")), st.get("avg_net", float("nan")),
                    *st.get("ci95_mean_net", [float("nan")] * 2),
                )

    # Pre-registered primary verdict
    primary_sig = [
        cfg[f"thr_{PRIMARY['threshold']}"]["fees"][PRIMARY["fees"]].get("significant_positive", False)
        for cfg in report["configs"].values()
    ]
    report["primary_verdict"] = {
        "criterion": "starter-taker, thr 0.55, pooled: total_net>0 AND CI95(mean net)>0",
        "configs_passing": int(sum(primary_sig)),
        "configs_total": len(primary_sig),
        "pass": bool(any(primary_sig)),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "fee_survivability.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Baseline equal-weight B&H net over OOS period: %+.4f",
                report["baseline"]["equal_weight_net"])
    logger.info("PRIMARY VERDICT: %s (%d/%d configs pass)",
                "PASS" if report["primary_verdict"]["pass"] else "FAIL",
                report["primary_verdict"]["configs_passing"], len(primary_sig))
    logger.info("STUDY COMPLETE -> %s", out)


if __name__ == "__main__":
    main()
