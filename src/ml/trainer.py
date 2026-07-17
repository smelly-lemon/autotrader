"""LightGBM model training with walk-forward cross-validation.

Trains on OHLCV features from SQLite candles, evaluates with proper
temporal splits and purge gaps, saves model artifacts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.ml.ohlcv_features import PURGE_BARS

logger = logging.getLogger(__name__)

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
    train_rows: int
    test_rows: int
    accuracy: float
    auc: float
    logloss: float
    feature_importances: dict[str, float] = field(default_factory=dict)


def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> list[FoldResult]:
    """Walk-forward CV with purged gaps between train and test."""
    timestamps = X.index.unique().sort_values()
    n_ts = len(timestamps)
    test_size = n_ts // (n_folds + 1)
    results = []

    for fold_i in range(n_folds):
        test_start_idx = n_ts - (n_folds - fold_i) * test_size
        test_end_idx = test_start_idx + test_size
        train_end_idx = test_start_idx - PURGE_BARS

        if train_end_idx <= 0 or test_start_idx >= n_ts:
            continue

        train_end_ts = timestamps[min(train_end_idx, n_ts - 1)]
        test_start_ts = timestamps[min(test_start_idx, n_ts - 1)]
        test_end_ts = timestamps[min(test_end_idx - 1, n_ts - 1)]

        train_mask = X.index <= train_end_ts
        test_mask = (X.index >= test_start_ts) & (X.index <= test_end_ts)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(X_train) < 200 or len(X_test) < 50:
            logger.warning("Fold %d: insufficient data, skipping", fold_i)
            continue

        common = X_train.columns.intersection(X_test.columns)
        X_train, X_test = X_train[common], X_test[common]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)

        model = lgb.train(
            LGBM_PARAMS, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        proba = model.predict(X_test)
        preds = (proba > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = 0.5
        ll = log_loss(y_test, proba)

        importances = dict(zip(
            model.feature_name(),
            model.feature_importance(importance_type="gain").tolist(),
        ))

        results.append(FoldResult(
            fold=fold_i, train_rows=len(X_train), test_rows=len(X_test),
            accuracy=acc, auc=auc, logloss=ll,
            feature_importances=importances,
        ))
        logger.info("  Fold %d: acc=%.4f AUC=%.4f logloss=%.4f (train=%d test=%d)",
                     fold_i, acc, auc, ll, len(X_train), len(X_test))

    return results


def train_final_model(
    X: pd.DataFrame, y: pd.Series, num_boost_round: int = 500,
) -> lgb.Booster:
    """Train on all data and return the model."""
    logger.info("Training final model: %d rows, %d features", len(X), len(X.columns))
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(LGBM_PARAMS, dtrain, num_boost_round=num_boost_round)
    return model


def save_model(model: lgb.Booster, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    meta = path.with_suffix(".features.txt")
    meta.write_text("\n".join(model.feature_name()))
    logger.info("Model saved: %s (%d features)", path, len(model.feature_name()))


def load_model(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def print_cv_summary(results: list[FoldResult]):
    if not results:
        print("No results.")
        return

    accs = [r.accuracy for r in results]
    aucs = [r.auc for r in results]
    lls = [r.logloss for r in results]

    print("\n" + "=" * 60)
    print("Walk-Forward CV Summary")
    print("=" * 60)
    for r in results:
        print(f"  Fold {r.fold}: acc={r.accuracy:.4f} AUC={r.auc:.4f} "
              f"LL={r.logloss:.4f} (train={r.train_rows} test={r.test_rows})")

    print(f"\n  Mean accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Mean AUC:      {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  Mean logloss:  {np.mean(lls):.4f} +/- {np.std(lls):.4f}")

    all_imp: dict[str, list[float]] = {}
    for r in results:
        for feat, imp in r.feature_importances.items():
            all_imp.setdefault(feat, []).append(imp)

    avg_imp = {k: np.mean(v) for k, v in all_imp.items()}
    top = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)[:15]
    print("\n  Top 15 features:")
    for feat, imp in top:
        print(f"    {feat:35s} {imp:10.1f}")
    print("=" * 60)
