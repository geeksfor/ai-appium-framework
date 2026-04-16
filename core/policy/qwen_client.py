from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
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

    增强点：
    1) timeout 支持环境变量 DASHSCOPE_TIMEOUT_S 覆盖；
    2) 支持 QWEN_DEBUG=1 打印关键请求信息；
    3) 支持 QWEN_DEBUG_DIR 落盘请求元信息，便于定位超时；
    4) 超时时会抛出更可读的错误，包含模型、base_url、耗时、payload 大小。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 60,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        ).rstrip("/")
        self.model = model or os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus")

        env_timeout = os.getenv("DASHSCOPE_TIMEOUT_S")
        if env_timeout is not None and str(env_timeout).strip():
            try:
                timeout_s = int(str(env_timeout).strip())
            except Exception:
                pass
        self.timeout_s = timeout_s

        self.debug = str(os.getenv("QWEN_DEBUG", "")).strip().lower() in {"1", "true", "yes"}
        self.debug_dir = str(os.getenv("QWEN_DEBUG_DIR", "")).strip()

        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")

    def _debug_print(self, *parts: Any) -> None:
        if self.debug:
            print("[QWEN DEBUG]", *parts)

    def _debug_dump(self, name: str, payload: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        try:
            d = Path(self.debug_dir)
            d.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            p = d / f"{ts}_{name}.json"
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

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

        request_json = json.dumps(body, ensure_ascii=False)
        request_bytes = len(request_json.encode("utf-8"))
        system_chars = len(system_prompt or "")
        user_chars = len(user_payload or "")

        meta = {
            "base_url": self.base_url,
            "url": url,
            "model": self.model,
            "timeout_s": self.timeout_s,
            "system_prompt_chars": system_chars,
            "user_payload_chars": user_chars,
            "request_bytes": request_bytes,
            "has_extra": bool(extra),
            "extra_keys": sorted(list(extra.keys())) if isinstance(extra, dict) else [],
        }
        self._debug_print("base_url =", self.base_url)
        self._debug_print("model =", self.model)
        self._debug_print("timeout_s =", self.timeout_s)
        self._debug_print("system_prompt_chars =", system_chars)
        self._debug_print("user_payload_chars =", user_chars)
        self._debug_print("request_bytes =", request_bytes)
        self._debug_dump("qwen_request_meta", meta)

        t0 = time.time()
        try:
            resp = requests.post(
                url,
                headers=headers,
                data=request_json,
                timeout=self.timeout_s,
            )
        except requests.exceptions.ReadTimeout as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            err = {
                **meta,
                "elapsed_ms": elapsed_ms,
                "error_type": "ReadTimeout",
                "error": str(e),
            }
            self._debug_print("failed_after_ms =", elapsed_ms)
            self._debug_print("exception =", repr(e))
            self._debug_dump("qwen_timeout", err)
            raise RuntimeError(
                f"Qwen API ReadTimeout after {elapsed_ms}ms; model={self.model}; "
                f"base_url={self.base_url}; timeout_s={self.timeout_s}; request_bytes={request_bytes}; "
                f"system_chars={system_chars}; user_chars={user_chars}; error={e}"
            ) from e
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            err = {
                **meta,
                "elapsed_ms": elapsed_ms,
                "error_type": type(e).__name__,
                "error": str(e),
            }
            self._debug_print("failed_after_ms =", elapsed_ms)
            self._debug_print("exception =", repr(e))
            self._debug_dump("qwen_exception", err)
            raise RuntimeError(
                f"Qwen API request failed after {elapsed_ms}ms; model={self.model}; "
                f"base_url={self.base_url}; timeout_s={self.timeout_s}; request_bytes={request_bytes}; "
                f"system_chars={system_chars}; user_chars={user_chars}; error={type(e).__name__}: {e}"
            ) from e

        elapsed_ms = int((time.time() - t0) * 1000)
        self._debug_print("request_elapsed_ms =", elapsed_ms)
        self._debug_print("status_code =", resp.status_code)

        # raise on non-2xx with useful text
        if resp.status_code // 100 != 2:
            err = {
                **meta,
                "elapsed_ms": elapsed_ms,
                "status_code": resp.status_code,
                "response_text_head": (resp.text or "")[:2000],
            }
            self._debug_dump("qwen_http_error", err)
            raise RuntimeError(
                f"Qwen API HTTP {resp.status_code}; model={self.model}; base_url={self.base_url}; "
                f"elapsed_ms={elapsed_ms}; request_bytes={request_bytes}; response={resp.text}"
            )

        raw = resp.json()
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")

        resp_meta = {
            **meta,
            "elapsed_ms": elapsed_ms,
            "status_code": resp.status_code,
            "response_content_chars": len(content or ""),
        }
        self._debug_dump("qwen_response_meta", resp_meta)

        return QwenResponse(content=content or "", elapsed_ms=elapsed_ms, raw=raw)
