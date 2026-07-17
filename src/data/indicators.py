from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a curated set of technical indicators on an OHLCV dataframe.

    Returns a new dataframe with indicator columns appended.  We pick indicators
    that give the LLM a well-rounded view: trend, momentum, volatility, and volume.
    """
    out = df.copy()

    # Trend
    out.ta.ema(length=9, append=True)
    out.ta.ema(length=21, append=True)
    out.ta.sma(length=50, append=True)
    out.ta.macd(append=True)

    # Momentum
    out.ta.rsi(length=14, append=True)
    out.ta.stochrsi(length=14, append=True)

    # Volatility
    out.ta.bbands(length=20, append=True)
    out.ta.atr(length=14, append=True)

    # Volume
    out.ta.obv(append=True)
    out.ta.vwap(append=True)

    return out


def summarize_indicators(df: pd.DataFrame) -> dict:
    """Summarize the latest indicator values into a dict for LLM consumption."""
    if df.empty:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    def _safe(col: str) -> float | None:
        for candidate in [col, col.upper(), col.lower()]:
            if candidate in latest.index:
                v = latest[candidate]
                return round(float(v), 6) if pd.notna(v) else None
        return None

    price = float(latest["close"])
    summary: dict = {
        "price": price,
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "volume": float(latest["volume"]),
        "price_change_pct": round((price - float(prev["close"])) / float(prev["close"]) * 100, 4),
    }

    indicator_keys = [
        "EMA_9", "EMA_21", "SMA_50",
        "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "RSI_14",
        "STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3",
        "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0",
        "ATRr_14",
        "OBV", "VWAP_D",
    ]
    for key in indicator_keys:
        val = _safe(key)
        if val is not None:
            summary[key] = val

    # Derived signals for the LLM
    ema9 = _safe("EMA_9")
    ema21 = _safe("EMA_21")
    if ema9 is not None and ema21 is not None:
        summary["ema_cross"] = "bullish" if ema9 > ema21 else "bearish"

    rsi = _safe("RSI_14")
    if rsi is not None:
        if rsi > 70:
            summary["rsi_signal"] = "overbought"
        elif rsi < 30:
            summary["rsi_signal"] = "oversold"
        else:
            summary["rsi_signal"] = "neutral"

    bbl = _safe("BBL_20_2.0")
    bbu = _safe("BBU_20_2.0")
    if bbl is not None and bbu is not None:
        if price < bbl:
            summary["bb_signal"] = "below_lower"
        elif price > bbu:
            summary["bb_signal"] = "above_upper"
        else:
            summary["bb_signal"] = "within_bands"

    return summary


def multi_timeframe_summary(
    collector,
    symbol: str,
    timeframes: list[str] | None = None,
) -> dict:
    """Placeholder for async multi-timeframe analysis. Call from async context."""
    raise NotImplementedError("Use async version: async_multi_timeframe_summary")


async def async_multi_timeframe_summary(
    collector,
    symbol: str,
    timeframes: list[str] | None = None,
) -> dict[str, dict]:
    timeframes = timeframes or ["5m", "15m", "1h", "4h"]
    summaries = {}
    for tf in timeframes:
        try:
            df = await collector.fetch_ohlcv(symbol, timeframe=tf, limit=100)
            df = compute_indicators(df)
            summaries[tf] = summarize_indicators(df)
        except Exception:
            summaries[tf] = {"error": "failed to fetch"}
    return summaries
