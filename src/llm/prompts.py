from __future__ import annotations

import json

SCANNER_SYSTEM = """You are a crypto market scanner. You analyze technical indicators and output a JSON object.
You must ONLY output valid JSON with these exact fields:
- "signal": one of "opportunity", "nothing"
- "direction": one of "long", "short", "none"
- "confidence": a float between 0.0 and 1.0
- "reasoning": a brief 1-2 sentence explanation

Do not output anything except the JSON object. No markdown, no code fences."""

ANALYZER_SYSTEM = """You are an expert crypto trading analyst. You receive multi-timeframe technical data
and portfolio context. You must decide whether to execute a trade.

You must ONLY output valid JSON with these exact fields:
- "action": one of "buy", "sell", "hold"
- "confidence": a float between 0.0 and 1.0
- "size_pct": what percentage of available cash to use (0.0 to 1.0), only for buy/sell
- "stop_loss_pct": suggested stop-loss distance as a decimal (e.g. 0.03 for 3%)
- "take_profit_pct": suggested take-profit distance as a decimal (e.g. 0.06 for 6%)
- "reasoning": a detailed explanation of your analysis and decision
- "risk_notes": any risk concerns about this trade

Base your decisions on the confluence of signals across timeframes.
Be conservative -- only trade when multiple indicators align.
Never chase momentum without confirmation. Prefer high-probability setups.
Do not output anything except the JSON object."""

STRATEGIST_SYSTEM = """You are a portfolio strategist reviewing overall crypto trading performance.
You analyze recent trades, P&L, market conditions, and current positions.

You must ONLY output valid JSON with these exact fields:
- "market_regime": one of "trending_up", "trending_down", "ranging", "volatile", "uncertain"
- "risk_adjustment": a multiplier for position sizing (0.5 = half size, 1.0 = normal, 1.5 = increased)
- "pairs_to_watch": list of trading pairs to prioritize
- "pairs_to_avoid": list of pairs to reduce exposure to
- "max_positions_override": suggested max open positions (1-5)
- "reasoning": detailed analysis of portfolio state and market conditions
- "action_items": list of specific recommendations

Do not output anything except the JSON object."""


def build_scanner_prompt(symbol: str, indicators: dict) -> str:
    return f"""Analyze {symbol} for trading opportunities.

Current technical indicators:
{json.dumps(indicators, indent=2)}

Based on these indicators, is there a trading opportunity? Consider:
1. Trend alignment (EMA cross, MACD)
2. Momentum (RSI, StochRSI)
3. Volatility (Bollinger Bands, ATR)
4. Volume confirmation (OBV, VWAP)

Output your assessment as JSON."""


def build_analyzer_prompt(
    symbol: str,
    multi_tf_data: dict[str, dict],
    portfolio: dict,
    recent_trades: list[dict],
    order_book: dict | None = None,
) -> str:
    parts = [
        f"## Trade Decision for {symbol}\n",
        "### Multi-Timeframe Analysis",
    ]
    for tf, data in multi_tf_data.items():
        parts.append(f"\n**{tf} timeframe:**")
        parts.append(json.dumps(data, indent=2))

    parts.append("\n### Portfolio State")
    parts.append(json.dumps(portfolio, indent=2))

    if recent_trades:
        parts.append("\n### Recent Trades (last 5)")
        for t in recent_trades[:5]:
            parts.append(f"- {t.get('symbol')} {t.get('side')} @ {t.get('price')} | PnL: {t.get('pnl', 'open')}")

    if order_book:
        parts.append("\n### Order Book Summary")
        parts.append(f"Spread: {order_book.get('spread', 'N/A')}")
        parts.append(f"Mid price: {order_book.get('mid_price', 'N/A')}")
        top_bids = order_book.get("bids", [])[:5]
        top_asks = order_book.get("asks", [])[:5]
        parts.append(f"Top bids: {top_bids}")
        parts.append(f"Top asks: {top_asks}")

    parts.append(
        "\n### Instructions\n"
        "Decide: buy, sell, or hold. Only trade if multiple signals align across timeframes. "
        "Be conservative with sizing. Consider the current portfolio exposure."
    )

    return "\n".join(parts)


def build_strategist_prompt(
    portfolio: dict,
    recent_trades: list[dict],
    daily_pnl: float,
    market_summaries: dict[str, dict],
) -> str:
    parts = [
        "## Portfolio Review\n",
        "### Current Portfolio",
        json.dumps(portfolio, indent=2),
        f"\n### Daily P&L: ${daily_pnl:.2f}",
        "\n### Recent Trade History",
    ]

    for t in recent_trades[:20]:
        status = "OPEN" if t.get("status") == "open" else f"PnL: ${t.get('pnl', 0):.2f}"
        parts.append(f"- {t.get('symbol')} {t.get('side')} @ {t.get('price')} [{status}]")

    parts.append("\n### Market Conditions by Pair")
    for pair, summary in market_summaries.items():
        parts.append(f"\n**{pair}:**")
        parts.append(json.dumps(summary, indent=2))

    parts.append(
        "\n### Instructions\n"
        "Review the portfolio and market conditions. Provide strategic guidance on:\n"
        "1. Overall market regime\n"
        "2. Whether to adjust risk (increase/decrease position sizes)\n"
        "3. Which pairs look most promising\n"
        "4. Any positions that should be closed\n"
    )

    return "\n".join(parts)
