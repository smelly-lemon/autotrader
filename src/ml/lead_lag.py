"""Strategy 1: BTC Lead-Lag Altcoin Trading Model.

LightGBM classifier that predicts alt-coin directional moves conditional
on BTC having moved significantly. Trained with walk-forward (purged)
time-series cross-validation.

Usage:
    # Train and evaluate
    python -m src.ml.lead_lag --days 30 --eval

    # Train final model and save
    python -m src.ml.lead_lag --days 180 --save models/lead_lag.txt
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)

from src.data.influx_client import HesiodInfluxClient
from src.ml.features import FeatureBuilder

logger = logging.getLogger(__name__)

TARGET_COL = "direction_15"
PURGE_BARS = 20  # bars to purge between train/test to prevent leakage

# Features to drop before training (identifiers, targets, raw prices)
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
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    accuracy: float
    auc: float
    logloss: float
    n_train: int
    n_test: int
    feature_importances: dict[str, float] = field(default_factory=dict)


def prepare_features(df: pd.DataFrame, btc_move_only: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the dataset and split into X, y.

    If btc_move_only=True, only keeps rows where BTC moved >0.3% in last 5 bars,
    which is the core thesis of the lead-lag strategy.
    """
    df = df.copy()

    if btc_move_only and "btc_moved" in df.columns:
        df = df[df["btc_moved"]].copy()

    df = df.dropna(subset=[TARGET_COL])

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Encode product_id if still present (shouldn't be after DROP_COLS)
    if "product_id" in X.columns:
        X["product_id"] = X["product_id"].astype("category")

    # Drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


def walk_forward_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    btc_move_only: bool = True,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> list[FoldResult]:
    """Walk-forward cross-validation with purged gaps.

    Splits data chronologically into expanding-window train sets and
    fixed-size test sets, with a purge gap between them to prevent
    look-ahead bias.
    """
    pairs = df["product_id"].unique()
    timestamps = df.index.unique().sort_values()
    n_timestamps = len(timestamps)

    # Each fold uses an expanding train window and a fixed test window
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

        X_train, y_train = prepare_features(train_df, btc_move_only)
        X_test, y_test = prepare_features(test_df, btc_move_only)

        if len(X_train) < 100 or len(X_test) < 20:
            logger.warning("Fold %d: insufficient data (train=%d, test=%d), skipping",
                           fold_i, len(X_train), len(X_test))
            continue

        # Align columns
        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        logger.info(
            "Fold %d: train %s..%s (%d rows), test %s..%s (%d rows)",
            fold_i,
            str(train_df.index.min())[:10],
            str(train_end_ts)[:10],
            len(X_train),
            str(test_start_ts)[:10],
            str(test_end_ts)[:10],
            len(X_test),
        )

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
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
        ll = log_loss(y_test, preds_proba)

        importances = dict(zip(
            model.feature_name(),
            model.feature_importance(importance_type="gain").tolist(),
        ))

        result = FoldResult(
            fold=fold_i,
            train_start=str(train_df.index.min())[:10],
            train_end=str(train_end_ts)[:10],
            test_start=str(test_start_ts)[:10],
            test_end=str(test_end_ts)[:10],
            accuracy=acc,
            auc=auc,
            logloss=ll,
            n_train=len(X_train),
            n_test=len(X_test),
            feature_importances=importances,
        )
        results.append(result)

        logger.info(
            "  -> accuracy=%.4f, AUC=%.4f, logloss=%.4f",
            acc, auc, ll,
        )

    return results


def train_final_model(
    df: pd.DataFrame,
    btc_move_only: bool = True,
    num_boost_round: int = 500,
) -> tuple[lgb.Booster, list[str]]:
    """Train on all available data and return the model + feature names."""
    X, y = prepare_features(df, btc_move_only)
    logger.info("Training final model on %d samples, %d features", len(X), len(X.columns))

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=num_boost_round,
    )
    return model, X.columns.tolist()


class LeadLagPredictor:
    """Inference wrapper for the trained lead-lag model."""

    def __init__(self, model_path: str | Path):
        self._model = lgb.Booster(model_file=str(model_path))
        self._feature_names = self._model.feature_name()

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score a feature DataFrame and return predictions with confidence.

        Returns a DataFrame with columns: product_id, probability, signal, confidence.
        signal: 1 (long), -1 (short), 0 (skip)
        """
        X = features.drop(
            columns=[c for c in DROP_COLS if c in features.columns], errors="ignore"
        )
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            X = X.drop(columns=non_numeric)
        X = X.replace([np.inf, -np.inf], np.nan)

        # Align with trained features
        missing = [c for c in self._feature_names if c not in X.columns]
        for c in missing:
            X[c] = np.nan
        X = X[self._feature_names]

        proba = self._model.predict(X)

        result = pd.DataFrame(index=features.index)
        result["product_id"] = features["product_id"] if "product_id" in features.columns else "unknown"
        result["probability"] = proba
        result["confidence"] = np.abs(proba - 0.5) * 2  # 0..1 scale
        result["signal"] = 0
        result.loc[proba > 0.55, "signal"] = 1  # long with margin
        result.loc[proba < 0.45, "signal"] = -1  # short with margin
        return result


def print_cv_summary(results: list[FoldResult]):
    """Print a formatted summary of walk-forward CV results."""
    if not results:
        print("No fold results to report.")
        return

    accs = [r.accuracy for r in results]
    aucs = [r.auc for r in results]
    lls = [r.logloss for r in results]

    print("\n" + "=" * 60)
    print("Walk-Forward Cross-Validation Summary")
    print("=" * 60)

    for r in results:
        print(f"  Fold {r.fold}: train[{r.train_start}..{r.train_end}] "
              f"test[{r.test_start}..{r.test_end}] "
              f"acc={r.accuracy:.4f} AUC={r.auc:.4f} LL={r.logloss:.4f} "
              f"(n_train={r.n_train}, n_test={r.n_test})")

    print(f"\n  Mean accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Mean AUC:      {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  Mean logloss:  {np.mean(lls):.4f} +/- {np.std(lls):.4f}")

    # Aggregate feature importances
    all_imp: dict[str, list[float]] = {}
    for r in results:
        for feat, imp in r.feature_importances.items():
            all_imp.setdefault(feat, []).append(imp)

    avg_imp = {k: np.mean(v) for k, v in all_imp.items()}
    top_features = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:15]

    print("\n  Top 15 features by avg gain:")
    for feat, imp in top_features:
        print(f"    {feat:40s} {imp:12.1f}")

    print("=" * 60)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train BTC lead-lag altcoin model")
    parser.add_argument("--days", type=int, default=30, help="Days of data to use")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--eval", action="store_true", help="Run walk-forward evaluation")
    parser.add_argument("--save", type=str, default=None, help="Save final model to path")
    parser.add_argument("--all-bars", action="store_true",
                        help="Train on all bars, not just BTC-move-conditioned ones")
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

        logger.info("Building lead-lag dataset [%s -> %s]...", start, stop)
        builder = FeatureBuilder(client)
        try:
            df = builder.build_lead_lag_dataset(start, stop, "1min")
        finally:
            builder.close()

    if df.empty:
        logger.error("No data returned. Check InfluxDB tunnel and date range.")
        return

    logger.info("Dataset: %d rows, %d columns", len(df), len(df.columns))

    btc_move_only = not args.all_bars

    if args.eval:
        results = walk_forward_cv(df, n_folds=args.folds, btc_move_only=btc_move_only)
        print_cv_summary(results)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model, feature_names = train_final_model(df, btc_move_only=btc_move_only)
        model.save_model(str(save_path))
        logger.info("Model saved to %s", save_path)

        meta_path = save_path.with_suffix(".features.txt")
        meta_path.write_text("\n".join(feature_names))
        logger.info("Feature names saved to %s", meta_path)


if __name__ == "__main__":
    main()
