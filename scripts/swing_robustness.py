#!/usr/bin/env python3
"""Robustness checks on the passing cell of the wide swing screen
(mean reversion, L=24h, k=3, H=72h, net of 120bps maker round trip).

The trade-level bootstrap treats trades as independent, but crash entries
cluster in time (many alts signal in the same week) — so the honest test
resamples clusters. Checks, fixed in advance:

1. WEEKLY CLUSTER BOOTSTRAP: resample entry-ISO-weeks with replacement;
   95% CI of mean net-maker must exclude zero.
2. MONTH CONCENTRATION: leave-one-month-out worst-case mean; mean after
   dropping the 3 best months.
3. PAIR BREADTH: fraction of pairs with positive mean net; split of
   original 12 screen pairs vs the 54 added in the wide batch.
4. BETA BASELINE: unconditional mean 72h forward return across all bars
   of the same pairs (is this just long-alt drift?), same net-of-fee basis.
5. Exact bootstrap p-values, trade-level and cluster-level.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "research" / "swing_screen_wide_results.json"
ORIGINAL = PROJECT_ROOT / "research" / "swing_screen_results.json"
CACHE_DIR = PROJECT_ROOT / "data" / "external" / "h1"

MAKER_RT = 0.0120
RNG = np.random.default_rng(7)
N_BOOT = 20_000


def main() -> None:
    report = json.loads(RESULTS.read_text())
    cell = next(c for c in report["phase2_screen"]["configs"]
                if c["family"] == "mr" and c["L_h"] == 24 and c["k"] == 3.0)
    trades = pd.DataFrame(cell["trades"])
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["net"] = trades["gross"] - MAKER_RT
    trades["week"] = trades["entry_ts"].dt.strftime("%G-W%V")
    trades["month"] = trades["entry_ts"].dt.strftime("%Y-%m")
    n = len(trades)
    print(f"cell: mr L=24h k=3 H=72h — n={n}, mean net {trades['net'].mean()*1e4:+.0f}bps")

    out = {"n_trades": n, "mean_net_maker_bps": round(trades["net"].mean() * 1e4, 1)}

    # 1. weekly cluster bootstrap
    weeks = trades.groupby("week")["net"].apply(list)
    wk_arrays = [np.array(v) for v in weeks]
    n_weeks = len(wk_arrays)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = RNG.integers(0, n_weeks, n_weeks)
        sample = np.concatenate([wk_arrays[i] for i in pick])
        means[b] = sample.mean()
    ci = np.percentile(means, [2.5, 97.5])
    p_cluster = float((means <= 0).mean())
    out["cluster_bootstrap"] = {
        "n_weeks": n_weeks,
        "ci95_bps": [round(c * 1e4, 1) for c in ci],
        "p_leq_zero": round(p_cluster, 5)}
    print(f"1. weekly cluster bootstrap ({n_weeks} weeks): "
          f"CI [{ci[0]*1e4:+.0f}, {ci[1]*1e4:+.0f}]bps, P(mean<=0)={p_cluster:.4f}")

    # trade-level p for reference
    tmeans = RNG.choice(trades["net"].to_numpy(), size=(N_BOOT, n), replace=True).mean(axis=1)
    out["trade_bootstrap_p_leq_zero"] = round(float((tmeans <= 0).mean()), 5)

    # 2. month concentration
    by_month = trades.groupby("month")["net"]
    month_means = by_month.mean().sort_values(ascending=False)
    month_ns = by_month.size()
    loo = {}
    for m in month_means.index:
        rest = trades[trades["month"] != m]["net"]
        loo[m] = rest.mean()
    worst_m = min(loo, key=loo.get)
    top3 = list(month_means.index[:3])
    dropped3 = trades[~trades["month"].isin(top3)]["net"]
    out["month_concentration"] = {
        "n_months": int(len(month_means)),
        "best_months": {m: {"mean_net_bps": round(month_means[m] * 1e4, 0),
                            "n": int(month_ns[m])} for m in top3},
        "loo_worst_mean_net_bps": round(loo[worst_m] * 1e4, 1),
        "loo_worst_month_removed": worst_m,
        "mean_net_bps_after_dropping_3_best_months": round(dropped3.mean() * 1e4, 1),
        "n_after_dropping_3_best": int(len(dropped3))}
    print(f"2. LOO worst (drop {worst_m}): {loo[worst_m]*1e4:+.0f}bps; "
          f"drop 3 best months: {dropped3.mean()*1e4:+.0f}bps (n={len(dropped3)})")

    # 3. pair breadth + original/new split
    per_pair = trades.groupby("pair")["net"].agg(["mean", "count"])
    frac_pos = float((per_pair["mean"] > 0).mean())
    orig12 = set(json.loads(ORIGINAL.read_text())["phase2_selected"])
    t_orig = trades[trades["pair"].isin(orig12)]["net"]
    t_new = trades[~trades["pair"].isin(orig12)]["net"]
    out["pair_breadth"] = {
        "n_pairs": int(len(per_pair)),
        "frac_pairs_positive_net": round(frac_pos, 3),
        "orig12": {"n": int(len(t_orig)), "mean_net_bps": round(t_orig.mean() * 1e4, 1)},
        "new_pairs": {"n": int(len(t_new)), "mean_net_bps": round(t_new.mean() * 1e4, 1)}}
    print(f"3. pairs positive: {frac_pos:.0%} of {len(per_pair)}; "
          f"orig12 {t_orig.mean()*1e4:+.0f}bps (n={len(t_orig)}), "
          f"new54 {t_new.mean()*1e4:+.0f}bps (n={len(t_new)})")

    # 4. unconditional 72h forward return baseline on same pairs
    pairs = trades["pair"].unique()
    unc = []
    for sym in pairs:
        f = CACHE_DIR / f"{sym.replace('/', '_')}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        full = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
        o = df["open"].reindex(full).to_numpy()
        r = o[72:] / o[:-72] - 1.0
        unc.append(pd.Series(r).dropna())
    allr = pd.concat(unc)
    out["beta_baseline"] = {
        "mean_unconditional_72h_gross_bps": round(float(allr.mean()) * 1e4, 1),
        "mean_unconditional_72h_net_maker_bps": round(
            float(allr.mean() - MAKER_RT) * 1e4, 1),
        "n_bars": int(len(allr))}
    print(f"4. unconditional 72h drift on same pairs: {allr.mean()*1e4:+.1f}bps gross "
          f"({(allr.mean()-MAKER_RT)*1e4:+.0f}bps net) vs signal {trades['net'].mean()*1e4:+.0f}bps net")

    out_path = PROJECT_ROOT / "research" / "swing_robustness.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
