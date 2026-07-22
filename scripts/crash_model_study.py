#!/usr/bin/env python3
"""Crash-triage model study: can learned conditioning beat the blunt rule?

Context: the wide swing screen found one passing rule (buy 3-sigma 24h
crashes on high-amplitude alts, hold 72h; +199bps/trade net of verified
maker fees, borderline cluster-robust). A live account has ~5 position
slots but up to 85 signals/month in crash weeks, so signals must be
ranked regardless — the question is whether a model ranks them better
than chance or than the arbitrary k=3 cutoff.

PRE-REGISTERED DESIGN (fixed before running):

Events: all >= 2-sigma 24h crashes (z = r_24h / sigma_24h <= -2) on the
66 high-amplitude pairs, non-overlapping per pair (as in the screen),
entry next bar open, label = 72h forward return (gross; net = gross -
120bps maker RT). Crash depth stays continuous so the model can learn
its own threshold.

Features (all strictly entry-time):
  crash_z, r24, r72, r168 (prior returns), sigma24 (vol regime),
  vol24_ratio (24h volume / 30d median), dd30 (distance from 30d high),
  btc_r24 (market-wide move), idio_r24 (pair minus BTC),
  pair statics: amp3d, log10 median dollar vol, spread_bps,
  listing_age_days, log10 price.

Model: LightGBM regressor, fixed conservative hyperparameters, labels
winsorized at 5th/95th pct of TRAIN set only. No tuning of any kind.

Validation: walk-forward by calendar quarter (test quarters 2025Q1 ->
2026Q3), train on all events ending >= 144h before the quarter starts,
minimum 150 train events else skip.

Judged OOS, pooled over test quarters, net of maker fees:
  PRIMARY: model top-half (by predicted return, within quarter) minus
    take-all: weekly-cluster bootstrap 95% CI of the difference > 0.
  SECONDARY (capacity): top-5-per-ISO-week by model vs first-5-per-week
    chronological (the no-skill baseline a capped account would use).
  DIAGNOSTIC: model top-half vs the blunt z<=-3 rule; feature gains.

Pair heterogeneity (descriptive, full sample): chi-square test of
between-pair variance of mean net vs sampling noise; empirical-Bayes
shrunk pair means; correlation of pair mean net with pair statics.

Output: research/crash_model_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "external" / "h1"
WIDE_RESULTS = PROJECT_ROOT / "research" / "swing_screen_wide_results.json"
INV_RESULTS = PROJECT_ROOT / "research" / "swing_screen_results.json"
OUT_PATH = PROJECT_ROOT / "research" / "crash_model_results.json"

MAKER_RT = 0.0120
Z_EVENT = -2.0
HOLD_H = 72
PURGE_H = 144
MIN_TRAIN_EVENTS = 150
RNG = np.random.default_rng(11)
N_BOOT = 20_000

LGB_PARAMS = dict(
    objective="regression", n_estimators=300, learning_rate=0.03,
    num_leaves=15, max_depth=4, min_child_samples=60, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1, seed=7)

FEATURES = ["crash_z", "r24", "r72", "r168", "sigma24", "vol24_ratio",
            "dd30", "btc_r24", "idio_r24", "amp3d", "log_dollar_vol",
            "spread_bps", "listing_age_days", "log_price"]


def build_events(z_event: float = Z_EVENT) -> pd.DataFrame:
    wide = json.loads(WIDE_RESULTS.read_text())
    pairs = wide["phase2_selected"]
    spreads = wide["phase2_screen"]["spreads_bps"]
    inv = {r["pair"]: r for r in json.loads(INV_RESULTS.read_text())["phase1_inventory"]}

    btc = pd.read_parquet(CACHE_DIR / "BTC_USD.parquet")
    btc_r24 = np.log(btc["close"]).diff(24)

    rows = []
    for sym in pairs:
        f = CACHE_DIR / f"{sym.replace('/', '_')}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        full = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
        df = df.reindex(full)
        c, o, v = df["close"], df["open"].to_numpy(), df["volume"]
        logc = np.log(c)

        r24 = logc.diff(24)
        r72 = logc.diff(72)
        r168 = logc.diff(168)
        sigma24 = r24.rolling(720, min_periods=240).std()
        z = r24 / sigma24
        vol24 = v.rolling(24).sum()
        vol_ratio = vol24 / vol24.rolling(720, min_periods=240).median()
        dd30 = c / c.rolling(720, min_periods=240).max() - 1.0
        b24 = btc_r24.reindex(full)

        stat = inv.get(sym, {})
        amp3d = stat.get("mean_abs_r3d_pct", np.nan)
        ldv = np.log10(max(stat.get("median_dollar_vol", 1.0), 1.0))
        spr = spreads.get(sym, 20.0)
        t0 = df.index[0]

        z_np = z.to_numpy()
        valid = np.isfinite(z_np)
        sig = valid & (z_np <= z_event)
        n = len(df)
        nxt = 0
        for t in np.flatnonzero(sig):
            if t < nxt:
                continue
            e, x = t + 1, t + 1 + HOLD_H
            if x >= n:
                break
            if not (np.isfinite(o[e]) and np.isfinite(o[x])) or o[e] <= 0:
                continue
            ts = df.index[e]
            rows.append({
                "pair": sym, "entry_ts": ts,
                "gross": o[x] / o[e] - 1.0,
                "crash_z": z_np[t], "r24": r24.iloc[t], "r72": r72.iloc[t],
                "r168": r168.iloc[t], "sigma24": sigma24.iloc[t],
                "vol24_ratio": vol_ratio.iloc[t], "dd30": dd30.iloc[t],
                "btc_r24": b24.iloc[t],
                "idio_r24": r24.iloc[t] - (b24.iloc[t] if np.isfinite(b24.iloc[t]) else 0),
                "amp3d": amp3d, "log_dollar_vol": ldv, "spread_bps": spr,
                "listing_age_days": (ts - t0).days,
                "log_price": np.log10(c.iloc[t]) if c.iloc[t] > 0 else np.nan,
            })
            nxt = x
    ev = pd.DataFrame(rows)
    ev["net"] = ev["gross"] - MAKER_RT
    ev["week"] = ev["entry_ts"].dt.strftime("%G-W%V")
    ev["quarter"] = ev["entry_ts"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    return ev.sort_values("entry_ts").reset_index(drop=True)


def cluster_ci(values: np.ndarray, weeks: np.ndarray,
               n_boot: int = N_BOOT) -> tuple[list[float], float]:
    """Weekly cluster bootstrap CI of the mean + P(mean<=0)."""
    groups = pd.Series(values).groupby(weeks).apply(lambda s: s.to_numpy())
    arrays = list(groups)
    k = len(arrays)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.integers(0, k, k)
        means[b] = np.concatenate([arrays[i] for i in pick]).mean()
    return ([round(float(np.percentile(means, q)) * 1e4, 1) for q in (2.5, 97.5)],
            round(float((means <= 0).mean()), 5))


def cluster_diff_ci(ev: pd.DataFrame, mask_a: np.ndarray,
                    mask_b: np.ndarray) -> tuple[list[float], float]:
    """Cluster bootstrap CI for mean(net|a) - mean(net|b), resampling weeks."""
    weeks = ev["week"].to_numpy()
    uniq = np.unique(weeks)
    idx_by_week = {w: np.flatnonzero(weeks == w) for w in uniq}
    net = ev["net"].to_numpy()
    k = len(uniq)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.integers(0, k, k)
        idx = np.concatenate([idx_by_week[uniq[i]] for i in pick])
        a, bb = net[idx][mask_a[idx]], net[idx][mask_b[idx]]
        diffs[b] = (a.mean() if len(a) else np.nan) - (bb.mean() if len(bb) else np.nan)
    diffs = diffs[np.isfinite(diffs)]
    return ([round(float(np.percentile(diffs, q)) * 1e4, 1) for q in (2.5, 97.5)],
            round(float((diffs <= 0).mean()), 5))


def walk_forward(ev: pd.DataFrame) -> pd.DataFrame:
    """Attach OOS predictions quarter by quarter; return events that got one."""
    quarters = sorted(q for q in ev["quarter"].unique() if q >= "2025Q1")
    ev = ev.copy()
    ev["pred"] = np.nan
    for q in quarters:
        q_start = pd.Period(q, freq="Q").start_time.tz_localize("UTC")
        cutoff = q_start - pd.Timedelta(hours=PURGE_H + HOLD_H)
        train = ev[ev["entry_ts"] < cutoff]
        test_idx = ev.index[ev["quarter"] == q]
        if len(train) < MIN_TRAIN_EVENTS or len(test_idx) == 0:
            continue
        lo, hi = train["gross"].quantile([0.05, 0.95])
        y = train["gross"].clip(lo, hi)
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(train[FEATURES], y)
        ev.loc[test_idx, "pred"] = model.predict(ev.loc[test_idx, FEATURES])
    return ev[ev["pred"].notna()].copy()


def evaluate(oos: pd.DataFrame) -> dict:
    out = {"n_oos_events": int(len(oos)),
           "quarters": sorted(oos["quarter"].unique().tolist())}
    net = oos["net"].to_numpy()
    weeks = oos["week"].to_numpy()

    # top-half within quarter
    top_half = np.zeros(len(oos), dtype=bool)
    for q in oos["quarter"].unique():
        m = (oos["quarter"] == q).to_numpy()
        cut = np.median(oos.loc[m, "pred"])
        top_half |= m & (oos["pred"].to_numpy() >= cut)

    take_all_ci, take_all_p = cluster_ci(net, weeks)
    th_ci, th_p = cluster_ci(net[top_half], weeks[top_half])
    diff_ci, diff_p = cluster_diff_ci(oos, top_half, np.ones(len(oos), dtype=bool))
    out["take_all"] = {"n": int(len(oos)), "mean_net_bps": round(net.mean() * 1e4, 1),
                       "ci95": take_all_ci, "p_leq0": take_all_p}
    out["model_top_half"] = {"n": int(top_half.sum()),
                             "mean_net_bps": round(net[top_half].mean() * 1e4, 1),
                             "ci95": th_ci, "p_leq0": th_p}
    out["PRIMARY_top_half_minus_all"] = {
        "diff_bps": round((net[top_half].mean() - net.mean()) * 1e4, 1),
        "ci95": diff_ci, "p_leq0": diff_p,
        "pass": bool(diff_ci[0] > 0)}

    # blunt rule z<=-3 subset for comparison
    z3 = (oos["crash_z"] <= -3).to_numpy()
    if z3.sum() > 20:
        z3_ci, z3_p = cluster_ci(net[z3], weeks[z3])
        out["blunt_z3_rule"] = {"n": int(z3.sum()),
                                "mean_net_bps": round(net[z3].mean() * 1e4, 1),
                                "ci95": z3_ci, "p_leq0": z3_p}

    # capacity: 5 per ISO week — model-ranked vs chronological
    def capped(select_by_pred: bool) -> np.ndarray:
        mask = np.zeros(len(oos), dtype=bool)
        for w in np.unique(weeks):
            m = np.flatnonzero(weeks == w)
            if select_by_pred:
                order = m[np.argsort(-oos["pred"].to_numpy()[m])]
            else:
                order = m[np.argsort(oos["entry_ts"].to_numpy()[m])]
            mask[order[:5]] = True
        return mask

    cap_model = capped(True)
    cap_chrono = capped(False)
    cm_ci, cm_p = cluster_ci(net[cap_model], weeks[cap_model])
    cc_ci, cc_p = cluster_ci(net[cap_chrono], weeks[cap_chrono])
    dd_ci, dd_p = cluster_diff_ci(oos, cap_model, cap_chrono)
    out["cap5_model"] = {"n": int(cap_model.sum()),
                         "mean_net_bps": round(net[cap_model].mean() * 1e4, 1),
                         "ci95": cm_ci, "p_leq0": cm_p}
    out["cap5_chrono"] = {"n": int(cap_chrono.sum()),
                          "mean_net_bps": round(net[cap_chrono].mean() * 1e4, 1),
                          "ci95": cc_ci, "p_leq0": cc_p}
    out["SECONDARY_cap5_model_minus_chrono"] = {
        "diff_bps": round((net[cap_model].mean() - net[cap_chrono].mean()) * 1e4, 1),
        "ci95": dd_ci, "p_leq0": dd_p, "pass": bool(dd_ci[0] > 0)}

    # per-quarter table
    rows = []
    for q in out["quarters"]:
        m = (oos["quarter"] == q).to_numpy()
        rows.append({"quarter": q, "n": int(m.sum()),
                     "take_all_bps": round(net[m].mean() * 1e4, 1),
                     "top_half_bps": round(net[m & top_half].mean() * 1e4, 1)
                     if (m & top_half).sum() else None})
    out["per_quarter"] = rows
    return out


def heterogeneity(ev: pd.DataFrame) -> dict:
    """Do pairs differ in mean net beyond sampling noise? (full sample)"""
    g = ev.groupby("pair")["net"]
    m, n = g.mean(), g.size()
    pooled_var = float(ev["net"].var())
    grand = float(ev["net"].mean())
    chi2 = float((n * (m - grand) ** 2 / pooled_var).sum())
    dof = len(m) - 1
    # chi2 p-value via survival function (no scipy: use gamma approx through numpy)
    # Wilson-Hilferty approximation
    zwh = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / np.sqrt(2 / (9 * dof))
    p_approx = float(0.5 * np.exp(-0.717 * zwh - 0.416 * zwh * zwh)) if zwh > 0 else 0.5
    # empirical Bayes shrinkage
    tau2 = max((chi2 - dof) * pooled_var / float(n.sum()), 0.0)
    shrunk = (m * tau2 + grand * (pooled_var / n) ) / (tau2 + pooled_var / n) \
        if tau2 > 0 else pd.Series(grand, index=m.index)
    top = shrunk.sort_values(ascending=False)
    stats_corr = {}
    per_pair = ev.groupby("pair").agg(
        mean_net=("net", "mean"), amp3d=("amp3d", "first"),
        ldv=("log_dollar_vol", "first"), spread=("spread_bps", "first"),
        age=("listing_age_days", "max"))
    for col in ("amp3d", "ldv", "spread", "age"):
        stats_corr[col] = round(float(per_pair["mean_net"].corr(per_pair[col])), 3)
    return {
        "n_pairs": int(len(m)),
        "chi2": round(chi2, 1), "dof": dof,
        "p_homogeneous_approx": round(p_approx, 4),
        "tau_bps": round(float(np.sqrt(tau2)) * 1e4, 1),
        "shrunk_top5_bps": {k: round(v * 1e4, 0) for k, v in top.head(5).items()},
        "shrunk_bottom5_bps": {k: round(v * 1e4, 0) for k, v in top.tail(5).items()},
        "corr_pair_mean_net_vs_statics": stats_corr,
        "note": "descriptive, full-sample; tau = cross-pair effect spread",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-event", type=float, default=Z_EVENT,
                    help="Event threshold (z <= this). -2 = triage population; "
                         "-3 = the passing rule's own population.")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    ev = build_events(args.z_event)
    print(f"events (z<={args.z_event}): {len(ev)} on {ev['pair'].nunique()} pairs, "
          f"{ev['entry_ts'].min().date()} -> {ev['entry_ts'].max().date()}")
    ev[FEATURES] = ev[FEATURES].astype(float)

    oos = walk_forward(ev)
    res = evaluate(oos)
    het = heterogeneity(ev)

    # feature importances from a final full-sample fit (diagnostic only)
    lo, hi = ev["gross"].quantile([0.05, 0.95])
    final = lgb.LGBMRegressor(**LGB_PARAMS).fit(ev[FEATURES], ev["gross"].clip(lo, hi))
    gains = dict(zip(FEATURES, final.booster_.feature_importance("gain")))
    total_gain = sum(gains.values()) or 1
    res["feature_gain_pct"] = {k: round(100 * v / total_gain, 1)
                               for k, v in sorted(gains.items(),
                                                  key=lambda kv: -kv[1])}

    report = {"design": "pre-registered; see docstring",
              "z_event": args.z_event,
              "n_events_total": int(len(ev)),
              "oos_evaluation": res,
              "pair_heterogeneity": het}
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}\n")

    print("=== OOS (pooled test quarters, net of 120bps maker RT) ===")
    for key in ("take_all", "model_top_half", "blunt_z3_rule",
                "cap5_chrono", "cap5_model"):
        if key in res:
            r = res[key]
            print(f"{key:>16}: n={r['n']:>5}  mean={r['mean_net_bps']:>7.1f}bps  "
                  f"CI={r['ci95']}  P(<=0)={r['p_leq0']}")
    print(f"\nPRIMARY  top-half − all:   {res['PRIMARY_top_half_minus_all']}")
    print(f"SECONDARY cap5 model−chrono: {res['SECONDARY_cap5_model_minus_chrono']}")
    print(f"\nheterogeneity: {json.dumps(het, indent=1)}")
    print(f"\nfeature gains: {res['feature_gain_pct']}")


if __name__ == "__main__":
    sys.exit(main())
