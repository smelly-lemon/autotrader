"""Strategy 2: Regime-Aware Swing Trading Model.

4h-horizon LightGBM with microstructure-enriched features, regime detection
via Hidden Markov Model (or volatility clustering), and adaptive thresholds.

Usage:
    python -m src.ml.swing --days 90 --eval
    python -m src.ml.swing --days 180 --save models/swing.txt
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.data.influx_client import HesiodInfluxClient
from src.ml.features import FeatureBuilder

logger = logging.getLogger(__name__)

TARGET_COL = "direction_1"  # 1 bar forward at 4h interval = 4h horizon
PURGE_BARS = 3  # 3 bars * 4h = 12h purge gap

DROP_COLS = [
    "product_id", "price", "price_open", "price_high", "price_low",
    "best_bid", "best_ask", "volume_24h", "vwap",
    "fwd_ret_1", "fwd_abs_ret_1", "direction_1",
    "fwd_ret_6", "fwd_abs_ret_6", "direction_6",
    "fwd_ret_15", "fwd_abs_ret_15", "direction_15",
    "fwd_ret_60", "fwd_abs_ret_60", "direction_60",
    "fwd_ret_240", "fwd_abs_ret_240", "direction_240",
]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "min_child_samples": 20,
    "lambda_l1": 0.05,
    "lambda_l2": 0.5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# Regime detection parameters
N_REGIMES = 3  # bull, bear, sideways


@dataclass
class RegimeState:
    regime: int  # 0=low-vol/sideways, 1=trending-up/bull, 2=trending-down/bear
    regime_proba: np.ndarray  # probabilities for each regime
    vol_percentile: float  # current volatility vs history


def detect_regimes_volatility(
    returns: pd.Series,
    vol_window: int = 30,
    n_regimes: int = N_REGIMES,
) -> pd.DataFrame:
    """Simple volatility-based regime detection.

    Uses realized volatility percentile ranking to classify into regimes:
    - Low vol + positive drift = bull
    - Low vol + negative drift = bear
    - High vol = volatile/transitional

    Falls back to this when HMM libraries aren't available.
    """
    vol = returns.rolling(vol_window, min_periods=5).std()
    drift = returns.rolling(vol_window, min_periods=5).mean()

    vol_rank = vol.rank(pct=True)

    regimes = pd.DataFrame(index=returns.index)
    regimes["realized_vol"] = vol
    regimes["vol_rank"] = vol_rank
    regimes["drift"] = drift

    # Classify regimes
    regimes["regime"] = 0  # default: sideways

    # Bull: low-to-medium vol + positive drift
    bull_mask = (vol_rank < 0.7) & (drift > 0)
    regimes.loc[bull_mask, "regime"] = 1

    # Bear: low-to-medium vol + negative drift
    bear_mask = (vol_rank < 0.7) & (drift < 0)
    regimes.loc[bear_mask, "regime"] = 2

    # High vol: regime 0 (sideways/transitional)

    return regimes


def detect_regimes_hmm(
    returns: pd.Series,
    n_regimes: int = N_REGIMES,
) -> pd.DataFrame:
    """HMM-based regime detection using hmmlearn."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.warning("hmmlearn not installed, falling back to volatility-based regime detection")
        return detect_regimes_volatility(returns, n_regimes=n_regimes)

    clean_returns = returns.dropna()
    if len(clean_returns) < 50:
        return detect_regimes_volatility(returns, n_regimes=n_regimes)

    X = clean_returns.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    model.fit(X)

    hidden_states = model.predict(X)
    state_proba = model.predict_proba(X)

    # Sort regimes by mean return so they have consistent meaning
    regime_means = {i: clean_returns[hidden_states == i].mean() for i in range(n_regimes)}
    sorted_regimes = sorted(regime_means.keys(), key=lambda k: regime_means[k])

    # Map: lowest mean = bear (2), middle = sideways (0), highest = bull (1)
    regime_map = {}
    if n_regimes == 3:
        regime_map = {sorted_regimes[0]: 2, sorted_regimes[1]: 0, sorted_regimes[2]: 1}
    else:
        for i, r in enumerate(sorted_regimes):
            regime_map[r] = i

    regimes = pd.DataFrame(index=clean_returns.index)
    regimes["regime"] = pd.Series(hidden_states, index=clean_returns.index).map(regime_map)
    regimes["realized_vol"] = clean_returns.rolling(30, min_periods=5).std()
    regimes["vol_rank"] = regimes["realized_vol"].rank(pct=True)
    regimes["drift"] = clean_returns.rolling(30, min_periods=5).mean()

    for i in range(n_regimes):
        regimes[f"regime_proba_{i}"] = state_proba[:, i]

    return regimes.reindex(returns.index)


def prepare_swing_features(
    df: pd.DataFrame,
    use_hmm: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features for the swing model, including regime features."""
    df = df.copy()
    df = df.dropna(subset=[TARGET_COL])

    y = df[TARGET_COL].astype(int)

    # Add regime features per product_id
    regime_frames = []
    for pid in df["product_id"].unique():
        mask = df["product_id"] == pid
        subset = df.loc[mask]
        if "ret_1" in subset.columns:
            returns = subset["ret_1"]
        else:
            returns = np.log(subset["price"]).diff()

        if use_hmm:
            regime_df = detect_regimes_hmm(returns)
        else:
            regime_df = detect_regimes_volatility(returns)

        regime_df.index = subset.index
        regime_frames.append(regime_df)

    if regime_frames:
        regime_combined = pd.concat(regime_frames)
        for col in regime_combined.columns:
            df[f"regime_{col}"] = regime_combined[col].values

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


def walk_forward_cv_swing(
    df: pd.DataFrame,
    n_folds: int = 5,
    use_hmm: bool = True,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> list[dict]:
    """Walk-forward CV for the swing model."""
    timestamps = df.index.unique().sort_values()
    n_timestamps = len(timestamps)
    test_size = n_timestamps // (n_folds + 1)

    results = []

    for fold_i in range(n_folds):
        test_start_idx = n_timestamps - (n_folds - fold_i) * test_size
        test_end_idx = test_start_idx + test_size
        train_end_idx = test_start_idx - PURGE_BARS

        if train_end_idx <= 0 or test_start_idx >= n_timestamps:
            continue

        train_end_ts = timestamps[min(train_end_idx, n_timestamps - 1)]
        test_start_ts = timestamps[min(test_start_idx, n_timestamps - 1)]
        test_end_ts = timestamps[min(test_end_idx - 1, n_timestamps - 1)]

        train_df = df[df.index <= train_end_ts]
        test_df = df[(df.index >= test_start_ts) & (df.index <= test_end_ts)]

        X_train, y_train = prepare_swing_features(train_df, use_hmm)
        X_test, y_test = prepare_swing_features(test_df, use_hmm)

        if len(X_train) < 50 or len(X_test) < 10:
            logger.warning("Fold %d: insufficient data, skipping", fold_i)
            continue

        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        logger.info(
            "Fold %d: train %s..%s (%d rows), test %s..%s (%d rows)",
            fold_i,
            str(train_df.index.min())[:10], str(train_end_ts)[:10], len(X_train),
            str(test_start_ts)[:10], str(test_end_ts)[:10], len(X_test),
        )

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model = lgb.train(
            LGBM_PARAMS, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        preds_proba = model.predict(X_test)
        preds_class = (preds_proba > 0.5).astype(int)

        acc = accuracy_score(y_test, preds_class)
        try:
            auc = roc_auc_score(y_test, preds_proba)
        except ValueError:
            auc = 0.5
        try:
            ll = log_loss(y_test, preds_proba, labels=[0, 1])
        except ValueError:
            ll = float("nan")

        importances = dict(zip(
            model.feature_name(),
            model.feature_importance(importance_type="gain").tolist(),
        ))

        result = {
            "fold": fold_i,
            "train_start": str(train_df.index.min())[:10],
            "train_end": str(train_end_ts)[:10],
            "test_start": str(test_start_ts)[:10],
            "test_end": str(test_end_ts)[:10],
            "accuracy": acc,
            "auc": auc,
            "logloss": ll,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feature_importances": importances,
        }
        results.append(result)

        logger.info("  -> accuracy=%.4f, AUC=%.4f, logloss=%.4f", acc, auc, ll)

    return results


def train_final_swing_model(
    df: pd.DataFrame,
    use_hmm: bool = True,
    num_boost_round: int = 500,
) -> tuple[lgb.Booster, list[str]]:
    """Train on all available data."""
    X, y = prepare_swing_features(df, use_hmm)
    logger.info("Training final swing model on %d samples, %d features", len(X), len(X.columns))

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=num_boost_round)
    return model, X.columns.tolist()


class SwingPredictor:
    """Inference wrapper for the trained swing model."""

    def __init__(self, model_path: str | Path):
        self._model = lgb.Booster(model_file=str(model_path))
        self._feature_names = self._model.feature_name()

    def predict(self, features: pd.DataFrame, use_hmm: bool = True) -> pd.DataFrame:
        """Score a feature DataFrame. Returns predictions with confidence and regime."""
        X, _ = prepare_swing_features(features, use_hmm)

        missing = [c for c in self._feature_names if c not in X.columns]
        for c in missing:
            X[c] = np.nan
        X = X[self._feature_names]

        proba = self._model.predict(X)

        result = pd.DataFrame(index=X.index)
        result["product_id"] = features["product_id"] if "product_id" in features.columns else "unknown"
        result["probability"] = proba
        result["confidence"] = np.abs(proba - 0.5) * 2
        result["signal"] = 0
        result.loc[proba > 0.55, "signal"] = 1
        result.loc[proba < 0.45, "signal"] = -1

        if "regime_regime" in features.columns:
            result["regime"] = features["regime_regime"].reindex(X.index)

        return result


def print_swing_cv_summary(results: list[dict]):
    """Print formatted swing model CV results."""
    if not results:
        print("No fold results to report.")
        return

    accs = [r["accuracy"] for r in results]
    aucs = [r["auc"] for r in results]
    lls = [r["logloss"] for r in results]

    print("\n" + "=" * 60)
    print("Swing Model Walk-Forward CV Summary")
    print("=" * 60)

    for r in results:
        print(f"  Fold {r['fold']}: train[{r['train_start']}..{r['train_end']}] "
              f"test[{r['test_start']}..{r['test_end']}] "
              f"acc={r['accuracy']:.4f} AUC={r['auc']:.4f} LL={r['logloss']:.4f} "
              f"(n_train={r['n_train']}, n_test={r['n_test']})")

    print(f"\n  Mean accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Mean AUC:      {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  Mean logloss:  {np.mean(lls):.4f} +/- {np.std(lls):.4f}")

    # Aggregate feature importances
    all_imp: dict[str, list[float]] = {}
    for r in results:
        for feat, imp in r["feature_importances"].items():
            all_imp.setdefault(feat, []).append(imp)

    avg_imp = {k: np.mean(v) for k, v in all_imp.items()}
    top_features = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:15]

    print("\n  Top 15 features by avg gain:")
    for feat, imp in top_features:
        print(f"    {feat:40s} {imp:12.1f}")

    print("=" * 60)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train regime-aware swing model")
    parser.add_argument("--days", type=int, default=90, help="Days of data to use")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--eval", action="store_true", help="Run walk-forward evaluation")
    parser.add_argument("--save", type=str, default=None, help="Save final model to path")
    parser.add_argument("--no-hmm", action="store_true", help="Use volatility-based regime detection instead of HMM")
    parser.add_argument("--from-parquet", type=str, default=None,
                        help="Load pre-extracted features from parquet file")
    parser.add_argument("--from-local", type=str, default=None,
                        help="Read from local parquet dir instead of InfluxDB (e.g. data/raw)")
    args = parser.parse_args()

    if args.from_parquet:
        logger.info("Loading pre-extracted features from %s...", args.from_parquet)
        df = pd.read_parquet(args.from_parquet)
    else:
        from datetime import datetime, timedelta, timezone
        stop = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")

        if args.from_local:
            from src.data.influx_client import LocalParquetClient
            client = LocalParquetClient(args.from_local)
        else:
            client = HesiodInfluxClient()

        logger.info("Building swing dataset [%s -> %s]...", start, stop)
        builder = FeatureBuilder(client)
        try:
            df = builder.build_swing_dataset(start, stop, "4h")
        finally:
            builder.close()

    if df.empty:
        logger.error("No data returned.")
        return

    logger.info("Dataset: %d rows, %d columns", len(df), len(df.columns))

    use_hmm = not args.no_hmm

    if args.eval:
        results = walk_forward_cv_swing(df, n_folds=args.folds, use_hmm=use_hmm)
        print_swing_cv_summary(results)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model, feature_names = train_final_swing_model(df, use_hmm=use_hmm)
        model.save_model(str(save_path))
        logger.info("Model saved to %s", save_path)

        meta_path = save_path.with_suffix(".features.txt")
        meta_path.write_text("\n".join(feature_names))
        logger.info("Feature names saved to %s", meta_path)


if __name__ == "__main__":
    main()
