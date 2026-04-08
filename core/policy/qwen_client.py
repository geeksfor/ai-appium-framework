# core/policy/qwen_client.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class QwenResponse:
    content: str
    elapsed_ms: int
    raw: Dict[str, Any]


class QwenClient:
    """
    DashScope / Model Studio OpenAI-compatible Chat client.
    Docs: OpenAI compatible interface requires API key, BASE_URL, model name.  :contentReference[oaicite:5]{index=5}
    Note: Some OpenAI-compatible endpoints may not support 'developer' role. Use system+user only. :contentReference[oaicite:6]{index=6}
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 60,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        self.model = model or os.getenv("DASHSCOPE_MODEL", "qwen-vl-max")
        self.timeout_s = timeout_s

        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")

    def chat(
        self,
        system_prompt: str,
        user_payload: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        extra: Optional[Dict[str, Any]] = None,
    ) -> QwenResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if extra:
            body.update(extra)

        t0 = time.time()
        resp = requests.post(url, headers=headers, data=json.dumps(body, ensure_ascii=False), timeout=self.timeout_s)
        elapsed_ms = int((time.time() - t0) * 1000)

        # raise on non-2xx with useful text
        if resp.status_code // 100 != 2:
            raise RuntimeError(f"Qwen API HTTP {resp.status_code}: {resp.text}")

        raw = resp.json()
        content = (
            raw.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return QwenResponse(content=content or "", elapsed_ms=elapsed_ms, raw=raw)