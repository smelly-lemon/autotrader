from __future__ import annotations

import json
import logging
import time

import httpx

from src.config import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for Ollama's HTTP API."""

    def __init__(self, config: AppConfig):
        self.base_url = config.ollama_base_url.rstrip("/")
        self.config = config
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=300)
        return self._http

    async def generate(
        self,
        prompt: str,
        model_config: ModelConfig,
        system: str = "",
        format_json: bool = True,
    ) -> dict:
        """Send a generate request to Ollama and return the parsed response."""
        payload: dict = {
            "model": model_config.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": model_config.temperature,
                "num_ctx": model_config.context_window,
            },
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        start = time.monotonic()
        logger.info("Ollama request: model=%s, prompt_len=%d", model_config.name, len(prompt))

        resp = await self.http.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        elapsed = time.monotonic() - start
        logger.info(
            "Ollama response: model=%s, tokens=%d, elapsed=%.1fs",
            model_config.name,
            data.get("eval_count", 0),
            elapsed,
        )

        return data

    async def generate_parsed(
        self,
        prompt: str,
        model_config: ModelConfig,
        system: str = "",
    ) -> dict:
        """Generate and parse the JSON response body."""
        data = await self.generate(prompt, model_config, system=system, format_json=True)
        text = data.get("response", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from model output: %s", text[:500])
            return {"raw": text, "parse_error": True}

    async def is_healthy(self) -> bool:
        try:
            resp = await self.http.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None
