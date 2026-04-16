from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import requests


def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    model = os.getenv("DASHSCOPE_MODEL", "qwen3.5-plus")
    timeout_s = int(os.getenv("DASHSCOPE_TIMEOUT_S", "20"))

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with exactly: ok"},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }

    print("[TEST] url =", url)
    print("[TEST] model =", model)
    print("[TEST] timeout_s =", timeout_s)
    print("[TEST] request_bytes =", len(json.dumps(body, ensure_ascii=False).encode("utf-8")))

    t0 = time.time()
    try:
        resp = requests.post(
            url,
            headers=headers,
            data=json.dumps(body, ensure_ascii=False),
            timeout=timeout_s,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        print("[TEST] elapsed_ms =", elapsed_ms)
        print("[TEST] status_code =", resp.status_code)
        print("[TEST] response_text =", resp.text[:1000])

        if resp.status_code // 100 != 2:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

        raw = resp.json()
        content = (
            raw.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        print("[TEST] parsed_content =", repr(content))

    except Exception as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        print("[TEST] failed_after_ms =", elapsed_ms)
        print("[TEST] exception =", repr(e))
        raise


if __name__ == "__main__":
    main()