# Auto-Trader

LLM-autonomous crypto trading system running 24/7 on local hardware with Ollama models.

## Architecture

Three-tier model strategy:

- **Tier 1 - Scanner** (`mistral:7b`): Fast scan every 2 minutes across all pairs
- **Tier 2 - Analyzer** (`qwen3.6:35b`): Deep multi-timeframe analysis on detected opportunities
- **Tier 3 - Strategist** (`gpt-oss:120b`): Portfolio review every 4 hours

A hard-coded risk manager sits between the LLM and execution and cannot be overridden by model output.

## Quick Start

```bash
# Create venv with Python 3.11+
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your Coinbase API keys (only needed for live trading)

# Make sure Ollama is running
ollama serve

# Start in paper trading mode (default)
python -m src.main
```

The dashboard will be available at http://localhost:8080.

## Configuration

Edit `config.yaml` to adjust:

- Trading pairs
- Model selection (any Ollama model)
- Risk parameters (position sizing, stop-loss, drawdown limits)
- Scan intervals
- Paper vs live mode

## Risk Management

Hard limits that cannot be overridden by any model:

| Limit | Default | Hard Max |
|-------|---------|----------|
| Position size | 15% | 25% |
| Open positions | 3 | 5 |
| Daily drawdown | 5% | 10% |
| Stop-loss | 3% | min 0.5% |
| Single trade loss | 2% | 5% |
| Loss cooldown | 15 min | min 60s |

## Going Live

1. Paper trade for at least 2 weeks
2. Review performance on the dashboard
3. Create Coinbase Advanced Trade API keys
4. Add keys to `.env`
5. Change `mode: live` in `config.yaml`
6. Start with minimum position sizes

## Tests

```bash
python -m pytest tests/ -v
```
