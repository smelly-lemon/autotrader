"""ML Signal Provider — bridges trained models with the existing LLM pipeline.

Generates structured ML signals from the lead-lag and swing models, formatted
for injection into the Scanner and Analyzer prompts.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "sideways", 1: "bull", 2: "bear"}


class MLSignalProvider:
    """Loads trained models and provides signals for prompt injection.

    Designed to be instantiated once and called per scan/analysis cycle.
    Models are loaded lazily on first use.
    """

    def __init__(
        self,
        lead_lag_model_path: str | Path | None = None,
        swing_model_path: str | Path | None = None,
    ):
        self._lead_lag_path = Path(lead_lag_model_path) if lead_lag_model_path else None
        self._swing_path = Path(swing_model_path) if swing_model_path else None
        self._lead_lag = None
        self._swing = None

    def _load_lead_lag(self):
        if self._lead_lag is None and self._lead_lag_path and self._lead_lag_path.exists():
            from src.ml.lead_lag import LeadLagPredictor
            self._lead_lag = LeadLagPredictor(self._lead_lag_path)
            logger.info("Loaded lead-lag model from %s", self._lead_lag_path)

    def _load_swing(self):
        if self._swing is None and self._swing_path and self._swing_path.exists():
            from src.ml.swing import SwingPredictor
            self._swing = SwingPredictor(self._swing_path)
            logger.info("Loaded swing model from %s", self._swing_path)

    def get_lead_lag_signal(self, features: pd.DataFrame) -> dict | None:
        """Generate lead-lag signal from pre-built features.

        Returns a dict suitable for prompt injection, or None if unavailable.
        """
        self._load_lead_lag()
        if self._lead_lag is None:
            return None

        try:
            preds = self._lead_lag.predict(features)
            if preds.empty:
                return None

            latest = preds.iloc[-1]
            return {
                "model": "lead_lag",
                "signal": _signal_label(int(latest["signal"])),
                "probability": round(float(latest["probability"]), 4),
                "confidence": round(float(latest["confidence"]), 4),
                "product_id": str(latest.get("product_id", "unknown")),
                "description": "BTC lead-lag model: predicts alt-coin moves conditional on BTC momentum",
            }
        except Exception:
            logger.exception("Lead-lag prediction failed")
            return None

    def get_swing_signal(self, features: pd.DataFrame) -> dict | None:
        """Generate swing signal from pre-built features.

        Returns a dict suitable for prompt injection, or None if unavailable.
        """
        self._load_swing()
        if self._swing is None:
            return None

        try:
            preds = self._swing.predict(features)
            if preds.empty:
                return None

            latest = preds.iloc[-1]
            regime = REGIME_LABELS.get(int(latest.get("regime", 0)), "unknown")

            return {
                "model": "swing",
                "signal": _signal_label(int(latest["signal"])),
                "probability": round(float(latest["probability"]), 4),
                "confidence": round(float(latest["confidence"]), 4),
                "regime": regime,
                "product_id": str(latest.get("product_id", "unknown")),
                "description": f"Regime-aware swing model (4h horizon, current regime: {regime})",
            }
        except Exception:
            logger.exception("Swing prediction failed")
            return None

    def format_signals_for_prompt(
        self,
        signals: list[dict],
    ) -> str:
        """Format a list of ML signals into a structured text block for LLM prompts."""
        if not signals:
            return ""

        lines = ["### ML Model Signals", ""]
        for sig in signals:
            if sig is None:
                continue
            direction = sig["signal"]
            conf = sig["confidence"]
            prob = sig["probability"]
            model = sig["model"]
            desc = sig.get("description", "")

            lines.append(f"**{model.upper()} Model** ({desc})")
            lines.append(f"  Signal: {direction} | Probability: {prob:.4f} | Confidence: {conf:.4f}")

            if "regime" in sig:
                lines.append(f"  Market regime: {sig['regime']}")

            # Add interpretive guidance
            if conf >= 0.8:
                lines.append("  Strength: STRONG — high model confidence")
            elif conf >= 0.4:
                lines.append("  Strength: MODERATE — consider with other signals")
            else:
                lines.append("  Strength: WEAK — treat as informational only")

            lines.append("")

        lines.append(
            "Note: ML signals are one input among many. They should confirm or challenge "
            "the technical analysis above, not override it. Disagreement between ML and "
            "technicals suggests caution."
        )

        return "\n".join(lines)


def _signal_label(signal: int) -> str:
    if signal == 1:
        return "LONG"
    elif signal == -1:
        return "SHORT"
    return "NEUTRAL"
