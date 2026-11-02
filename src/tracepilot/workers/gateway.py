from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class CompatibleJSONGateway:
    """Explicitly configured chat-completions endpoint; usable with local or remote model servers."""

    def __init__(self, endpoint: str, model: str, *, api_key_env: str = "OPENAI_API_KEY",
                 timeout: float = 90) -> None:
        if urllib.parse.urlparse(endpoint).scheme not in {"http", "https"} or not model:
            raise ValueError("an HTTP(S) chat-completions endpoint and model are required")
        self.endpoint, self.model, self.api_key_env, self.timeout = endpoint, model, api_key_env, timeout

    def generate(self, instructions: str, context: dict[str, Any]) -> dict[str, Any]:
        request = {
            "model": self.model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": instructions},
                         {"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
        }
        headers = {"Content-Type": "application/json"}
        if os.environ.get(self.api_key_env):
            headers["Authorization"] = "Bearer " + os.environ[self.api_key_env]
        start = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(
                self.endpoint, data=json.dumps(request).encode(), headers=headers, method="POST"
            ), timeout=self.timeout) as response:
                raw = response.read(2_000_001)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"model endpoint returned HTTP {exc.code}") from exc
        if len(raw) > 2_000_000:
            raise ValueError("model response exceeds limit")
        response = json.loads(raw)
        payload = json.loads(response["choices"][0]["message"]["content"])
        if not isinstance(payload, dict):
            raise ValueError("model response must be a JSON object")
        return {
            "proposal": payload, "model": response.get("model", self.model),
            "usage": response.get("usage", {}), "elapsed_seconds": time.monotonic() - start,
            "prompt_version": "research-proposal-v1", "executed": False,
        }
