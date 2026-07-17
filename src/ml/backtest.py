"""Walk-forward backtesting engine with realistic Coinbase fee modeling.

Validates that ML strategies produce net-positive returns after accounting
for maker/taker fees at various volume tiers.

Usage:
    python -m src.ml.backtest --strategy lead-lag --days 30 --folds 5
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ml.features import FeatureBuilder
from src.ml.lead_lag import (
    LGBM_PARAMS,
    TARGET_COL,
    prepare_features,
    PURGE_BARS,
)

logger = logging.getLogger(__name__)

# Coinbase Advanced fee schedule (as of 2026)
# https://help.coinbase.com/en/advanced-trade/trading-and-funding/advanced-trade-fees
FEE_TIERS = {
    "starter":    {"maker": 0.0060, "taker": 0.0120},  # <$1K/month
    "bronze":     {"maker": 0.0040, "taker": 0.0060},  # $1K-$10K/month
    "silver":     {"maker": 0.0025, "taker": 0.0040},  # $10K-$50K/month
    "gold":       {"maker": 0.0015, "taker": 0.0025},  # $50K-$100K/month
    "platinum":   {"maker": 0.0010, "taker": 0.0018},  # $100K-$1M/month
}

DEFAULT_FEE_TIER = "silver"


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    product_id: str
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: float
    gross_return: float
    fee_cost: float
    net_return: float
    confidence: float
    holding_bars: int


@dataclass
class BacktestResult:
    strategy: str
    fee_tier: str
    start_date: str
    end_date: str
    n_trades: int
    win_rate: float
    avg_gross_return: float
    avg_net_return: float
    total_gross_return: float
    total_net_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_holding_bars: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


class BacktestEngine:
    """Simulates trading with realistic fee modeling."""

    def __init__(
        self,
        fee_tier: str = DEFAULT_FEE_TIER,
        confidence_threshold: float = 0.55,
        max_holding_bars: int = 60,
        initial_capital: float = 10_000.0,
        position_size_frac: float = 0.1,
    ):
        fees = FEE_TIERS[fee_tier]
        self.maker_fee = fees["maker"]
        self.taker_fee = fees["taker"]
        self.fee_tier = fee_tier
        self.confidence_threshold = confidence_threshold
        self.max_holding_bars = max_holding_bars
        self.initial_capital = initial_capital
        self.position_size_frac = position_size_frac

    def _fee_for_trade(self, is_maker: bool = False) -> float:
        return self.maker_fee if is_maker else self.taker_fee

    def run_backtest(
        self,
        predictions: pd.DataFrame,
        prices: pd.Series,
        product_ids: pd.Series | None = None,
    ) -> BacktestResult:
        """Run a backtest on model predictions.

        predictions must have columns: probability, signal, confidence
        prices: Series of asset prices aligned with predictions index
        """
        trades: list[Trade] = []
        equity = self.initial_capital
        equity_curve = [equity]
        equity_times = [predictions.index[0]] if len(predictions) > 0 else []

        position = None
        position_entry_time = None
        position_entry_price = None
        position_direction = None
        position_confidence = None
        position_product = None
        bars_held = 0

        for i, (ts, row) in enumerate(predictions.iterrows()):
            price = prices.get(ts, np.nan)
            if np.isnan(price) or price <= 0:
                continue

            # Check exit conditions for open position
            if position is not None:
                bars_held += 1
                should_exit = False

                # Time-based exit
                if bars_held >= self.max_holding_bars:
                    should_exit = True

                # Signal reversal exit
                if row["signal"] != 0 and row["signal"] != position_direction:
                    should_exit = True

                if should_exit:
                    exit_fee = self._fee_for_trade(is_maker=False)
                    entry_fee = self._fee_for_trade(is_maker=False)

                    if position_direction == 1:
                        gross_ret = (price - position_entry_price) / position_entry_price
                    else:
                        gross_ret = (position_entry_price - price) / position_entry_price

                    fee_cost = entry_fee + exit_fee
                    net_ret = gross_ret - fee_cost

                    trade_pnl = net_ret * equity * self.position_size_frac
                    equity += trade_pnl

                    trades.append(Trade(
                        entry_time=position_entry_time,
                        exit_time=ts,
                        product_id=position_product or "unknown",
                        direction=position_direction,
                        entry_price=position_entry_price,
                        exit_price=price,
                        gross_return=gross_ret,
                        fee_cost=fee_cost,
                        net_return=net_ret,
                        confidence=position_confidence,
                        holding_bars=bars_held,
                    ))

                    position = None
                    position_entry_time = None
                    position_entry_price = None
                    position_direction = None
                    position_confidence = None
                    position_product = None
                    bars_held = 0

            # Check entry conditions
            if position is None and row["signal"] != 0 and row["confidence"] >= (self.confidence_threshold - 0.5) * 2:
                position = True
                position_entry_time = ts
                position_entry_price = price
                position_direction = row["signal"]
                position_confidence = row["confidence"]
                if product_ids is not None:
                    position_product = product_ids.get(ts, "unknown")
                bars_held = 0

            equity_curve.append(equity)
            equity_times.append(ts)

        # Close any remaining position at the last price
        if position is not None:
            last_price = prices.iloc[-1]
            if position_direction == 1:
                gross_ret = (last_price - position_entry_price) / position_entry_price
            else:
                gross_ret = (position_entry_price - last_price) / position_entry_price

            fee_cost = self._fee_for_trade() * 2
            net_ret = gross_ret - fee_cost

            trades.append(Trade(
                entry_time=position_entry_time,
                exit_time=predictions.index[-1],
                product_id=position_product or "unknown",
                direction=position_direction,
                entry_price=position_entry_price,
                exit_price=last_price,
                gross_return=gross_ret,
                fee_cost=fee_cost,
                net_return=net_ret,
                confidence=position_confidence,
                holding_bars=bars_held,
            ))

        eq_series = pd.Series(equity_curve, index=equity_times[:len(equity_curve)])

        # Compute metrics
        if not trades:
            return BacktestResult(
                strategy="lead-lag", fee_tier=self.fee_tier,
                start_date=str(predictions.index[0])[:10] if len(predictions) > 0 else "",
                end_date=str(predictions.index[-1])[:10] if len(predictions) > 0 else "",
                n_trades=0, win_rate=0, avg_gross_return=0, avg_net_return=0,
                total_gross_return=0, total_net_return=0, max_drawdown=0,
                sharpe_ratio=0, profit_factor=0, avg_holding_bars=0,
                equity_curve=eq_series,
            )

        net_returns = [t.net_return for t in trades]
        gross_returns = [t.gross_return for t in trades]
        wins = [r for r in net_returns if r > 0]
        losses = [r for r in net_returns if r <= 0]

        # Max drawdown from equity curve
        peak = eq_series.expanding().max()
        drawdown = (eq_series - peak) / peak
        max_dd = drawdown.min()

        # Sharpe ratio (annualized, assuming 1-min bars)
        if len(net_returns) > 1:
            ret_std = np.std(net_returns)
            ret_mean = np.mean(net_returns)
            trades_per_year = len(trades) / max(1, (predictions.index[-1] - predictions.index[0]).days) * 365
            sharpe = (ret_mean / ret_std * np.sqrt(trades_per_year)) if ret_std > 0 else 0
        else:
            sharpe = 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        return BacktestResult(
            strategy="lead-lag",
            fee_tier=self.fee_tier,
            start_date=str(predictions.index[0])[:10],
            end_date=str(predictions.index[-1])[:10],
            n_trades=len(trades),
            win_rate=len(wins) / len(trades),
            avg_gross_return=np.mean(gross_returns),
            avg_net_return=np.mean(net_returns),
            total_gross_return=sum(gross_returns),
            total_net_return=sum(net_returns),
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_holding_bars=np.mean([t.holding_bars for t in trades]),
            trades=trades,
            equity_curve=eq_series,
        )


def walk_forward_backtest(
    df: pd.DataFrame,
    n_folds: int = 5,
    fee_tier: str = DEFAULT_FEE_TIER,
    btc_move_only: bool = True,
    confidence_threshold: float = 0.55,
    max_holding_bars: int = 60,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> list[BacktestResult]:
    """Walk-forward backtest: train model on fold, generate predictions, simulate trades."""
    timestamps = df.index.unique().sort_values()
    n_timestamps = len(timestamps)
    test_size = n_timestamps // (n_folds + 1)

    engine = BacktestEngine(
        fee_tier=fee_tier,
        confidence_threshold=confidence_threshold,
        max_holding_bars=max_holding_bars,
    )
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
            logger.warning("Fold %d: insufficient data, skipping", fold_i)
            continue

        common_cols = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        logger.info(
            "Fold %d: train %d rows, test %d rows",
            fold_i, len(X_train), len(X_test),
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

        predictions = pd.DataFrame(index=X_test.index)
        predictions["probability"] = preds_proba
        predictions["confidence"] = np.abs(preds_proba - 0.5) * 2
        predictions["signal"] = 0
        predictions.loc[preds_proba > confidence_threshold, "signal"] = 1
        predictions.loc[preds_proba < (1 - confidence_threshold), "signal"] = -1

        # Extract prices and product IDs from the test set
        test_filtered = test_df.copy()
        if btc_move_only and "btc_moved" in test_filtered.columns:
            test_filtered = test_filtered[test_filtered["btc_moved"]]
        test_filtered = test_filtered.dropna(subset=[TARGET_COL])

        prices = test_filtered["price"].reindex(X_test.index)
        product_ids = test_filtered["product_id"].reindex(X_test.index) if "product_id" in test_filtered.columns else None

        result = engine.run_backtest(predictions, prices, product_ids)
        result.strategy = f"lead-lag-fold-{fold_i}"
        results.append(result)

        logger.info(
            "  Fold %d results: %d trades, win_rate=%.2f%%, "
            "net_return=%.4f, sharpe=%.2f, profit_factor=%.2f",
            fold_i, result.n_trades, result.win_rate * 100,
            result.total_net_return, result.sharpe_ratio, result.profit_factor,
        )

    return results


def print_backtest_summary(results: list[BacktestResult]):
    """Print formatted backtest results across all folds."""
    if not results:
        print("No backtest results to report.")
        return

    print("\n" + "=" * 70)
    print("Walk-Forward Backtest Summary")
    print("=" * 70)
    print(f"  Fee tier: {results[0].fee_tier}")
    print()

    total_trades = 0
    all_net = []
    all_gross = []

    for r in results:
        print(f"  {r.strategy}: {r.start_date}..{r.end_date}")
        print(f"    Trades: {r.n_trades:>6d}  |  Win rate: {r.win_rate:>6.1%}  |  "
              f"Avg net return: {r.avg_net_return:>+8.4f}")
        print(f"    Gross total: {r.total_gross_return:>+8.4f}  |  "
              f"Net total: {r.total_net_return:>+8.4f}  |  "
              f"Max DD: {r.max_drawdown:>+7.2%}")
        print(f"    Sharpe: {r.sharpe_ratio:>6.2f}  |  "
              f"Profit factor: {r.profit_factor:>6.2f}  |  "
              f"Avg hold: {r.avg_holding_bars:>5.1f} bars")
        print()
        total_trades += r.n_trades
        all_net.append(r.total_net_return)
        all_gross.append(r.total_gross_return)

    print(f"  Aggregate:")
    print(f"    Total trades across folds: {total_trades}")
    print(f"    Mean fold gross return: {np.mean(all_gross):>+8.4f}")
    print(f"    Mean fold net return:   {np.mean(all_net):>+8.4f}")
    print(f"    Net positive folds:     {sum(1 for n in all_net if n > 0)}/{len(all_net)}")

    # Fee impact analysis
    gross_sum = sum(all_gross)
    net_sum = sum(all_net)
    fee_drag = gross_sum - net_sum
    print(f"\n  Fee impact analysis:")
    print(f"    Total gross: {gross_sum:>+8.4f}")
    print(f"    Total net:   {net_sum:>+8.4f}")
    print(f"    Fee drag:    {fee_drag:>+8.4f} ({fee_drag/max(abs(gross_sum), 1e-8)*100:.1f}% of gross)")
    print("=" * 70)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Walk-forward backtest with realistic fees")
    parser.add_argument("--strategy", choices=["lead-lag", "swing"], default="lead-lag")
    parser.add_argument("--days", type=int, default=30, help="Days of data to use")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--fee-tier", choices=list(FEE_TIERS.keys()), default=DEFAULT_FEE_TIER)
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--max-hold", type=int, default=60, help="Max holding period in bars")
    args = parser.parse_args()

    from datetime import datetime, timedelta, timezone
    stop = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")

    logger.info("Building dataset [%s -> %s]...", start, stop)
    builder = FeatureBuilder()
    try:
        if args.strategy == "lead-lag":
            df = builder.build_lead_lag_dataset(start, stop, "1min")
        else:
            df = builder.build_swing_dataset(start, stop, "4h")
    finally:
        builder.close()

    if df.empty:
        logger.error("No data returned.")
        return

    results = walk_forward_backtest(
        df,
        n_folds=args.folds,
        fee_tier=args.fee_tier,
        btc_move_only=(args.strategy == "lead-lag"),
        confidence_threshold=args.threshold,
        max_holding_bars=args.max_hold,
    )

    print_backtest_summary(results)


if __name__ == "__main__":
    main()
