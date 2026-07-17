#!/usr/bin/env python3
"""Train and export the best model configuration for live trading.

Uses the winning config from model_search_v2 (4h bars, h30 horizon,
magnitude-filtered binary classification with advanced features)
and saves the model for use by MLLiveTrader.

Usage:
    python scripts/train_best_model.py --data-dir data/raw --out models/best_model.txt
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_search_v2 import build_dataset_v2, make_drop_cols, _load
from src.data.influx_client import LocalParquetClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out", default="models/best_model.txt")
    parser.add_argument("--start", default="2025-11-05")
    parser.add_argument("--stop", default=None)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    client = LocalParquetClient(args.data_dir)
    stop = args.stop or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("Building dataset...")
    ds = build_dataset_v2(client, "4h", args.start, stop, horizons=[30])
    if ds.empty:
        logger.error("Empty dataset")
        return

    logger.info("Dataset: %d rows, %d cols", len(ds), len(ds.columns))

    # Prepare
    work = ds.copy().reset_index()
    if work.columns[0] not in ds.columns:
        work.rename(columns={work.columns[0]: "timestamp"}, inplace=True)

    target_col = "direction_30"
    ret_col = "fwd_ret_30"
    work = work.dropna(subset=[target_col])
    if ret_col in work.columns:
        work = work[work[ret_col].abs() >= 0.002]

    y = work[target_col].astype(int)
    drop = make_drop_cols([30])
    all_drop = list(set(drop) & set(work.columns))
    X = work.drop(columns=all_drop, errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    logger.info("Training on %d samples, %d features", len(X), len(X.columns))

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "num_leaves": 127, "learning_rate": 0.05,
        "feature_fraction": 0.6, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 30, "lambda_l1": 0.1, "lambda_l2": 1.0,
        "verbose": -1, "n_jobs": -1, "seed": 42,
    }

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=500)
    model.save_model(args.out)

    # Feature importance
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=model.feature_name()
    ).sort_values(ascending=False)

    logger.info("\nTop 20 features by gain:")
    for feat, gain in importance.head(20).items():
        logger.info("  %-30s %.1f", feat, gain)

    logger.info("\nModel saved to %s", args.out)
    logger.info("Use with: python -m src.ml.live_trader --model %s", args.out)
    client.close()


if __name__ == "__main__":
    main()
