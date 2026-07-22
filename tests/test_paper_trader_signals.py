"""Tests for the pure signal functions of the two-sleeve paper trader."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from paper_trader import crash_z_last_bar, donchian_state  # noqa: E402


def hourly_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")


def daily_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")


class TestCrashZ:
    def test_flat_series_no_signal(self):
        n = 800
        rng = np.random.default_rng(1)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))),
                          index=hourly_index(n))
        z = crash_z_last_bar(close)
        assert z is not None
        assert abs(z) < 3

    def test_crash_detected(self):
        n = 800
        rng = np.random.default_rng(2)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        prices[-24:] = prices[-25] * np.linspace(1.0, 0.70, 24)  # 30% 24h crash
        close = pd.Series(prices, index=hourly_index(n))
        z = crash_z_last_bar(close)
        assert z is not None and z <= -3

    def test_insufficient_history(self):
        close = pd.Series(np.linspace(100, 90, 100), index=hourly_index(100))
        assert crash_z_last_bar(close) is None

    def test_pump_not_signal(self):
        n = 800
        rng = np.random.default_rng(3)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        prices[-24:] = prices[-25] * np.linspace(1.0, 1.4, 24)  # pump, not crash
        close = pd.Series(prices, index=hourly_index(n))
        z = crash_z_last_bar(close)
        assert z is not None and z > 0


class TestDonchianState:
    def test_breakout_enters(self):
        n = 100
        prices = np.full(n, 100.0)
        prices[:60] += np.sin(np.arange(60)) * 2       # sideways
        prices[60:] = np.linspace(103, 140, 40)        # breakout + trend
        in_pos, entry = donchian_state(pd.Series(prices, index=daily_index(n)))
        assert in_pos
        assert entry is not None and entry >= daily_index(n)[59]

    def test_breakdown_exits(self):
        n = 140
        prices = np.concatenate([
            np.full(60, 100.0) + np.sin(np.arange(60)) * 2,
            np.linspace(103, 140, 40),     # trend up (enter)
            np.linspace(139, 90, 40),      # collapse through 20d low (exit)
        ])
        in_pos, entry = donchian_state(pd.Series(prices, index=daily_index(n)))
        assert not in_pos and entry is None

    def test_cap_exits_stale_position(self):
        # enters, then grinds up so slowly the 20d low is never touched,
        # for longer than the 120d cap -> replay must show flat (cap exit),
        # unless a re-entry cross happens later.
        n = 260
        prices = np.concatenate([
            np.full(60, 100.0) + np.sin(np.arange(60)) * 2,
            np.linspace(103, 110, 200),    # 200d slow grind, no 20d-low touch
        ])
        # a perfectly monotone grind re-enters on every bar after a cap exit
        # (each new close is a 55d high). Use a plateau after day 160 so no
        # new 55d-high cross occurs after the cap fires.
        prices[220:] = prices[219]
        in_pos, entry = donchian_state(pd.Series(prices, index=daily_index(n)))
        # position entered ~day 60 hits the 120d cap ~day 180; plateau after
        # day 220 prevents fresh crosses right at the end -> flat OR a
        # re-entry strictly after the cap date is acceptable; entered-at-60
        # must NOT still be open.
        if in_pos:
            assert (daily_index(n)[-1] - entry).days < 120
        else:
            assert entry is None

    def test_no_history_flat(self):
        in_pos, entry = donchian_state(pd.Series(dtype=float))
        assert not in_pos and entry is None


@pytest.mark.parametrize("seed", [7, 11, 13])
def test_donchian_never_crashes_on_random_walks(seed):
    rng = np.random.default_rng(seed)
    n = 400
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    in_pos, entry = donchian_state(pd.Series(prices, index=daily_index(n)))
    assert isinstance(in_pos, bool)
