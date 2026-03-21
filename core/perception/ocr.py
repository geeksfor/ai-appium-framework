# core/perception/ocr.py
from __future__ import annotations

import base64
import mimetypes
import os
import time
import json
import re
from dataclasses import dataclass
from typing import Optional, Protocol

import requests


@dataclass
class OCRResult:
    text: str
    provider: str
    model: str
    elapsed_ms: int
    raw: Optional[dict] = None
    error: Optional[str] = None


class OCRProvider(Protocol):
    def recognize(self, image_path: str) -> OCRResult: ...


def _encode_image_to_data_url(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # DashScope/OpenAI 兼容接口支持 data url 的 image_url 传图方式 :contentReference[oaicite:1]{index=1}
    return f"data:{mime};base64,{b64}"


class QwenVisionOCRProvider:
    """
    用 Qwen-VL/Qwen-OCR 走 OpenAI-compatible Vision 接口做 OCR。
    - 推荐 model: qwen-vl-ocr（专用 OCR）:contentReference[oaicite:2]{index=2}
    - base_url（新加坡）默认：https://dashscope-intl.aliyuncs.com/compatible-mode/v1 :contentReference[oaicite:3]{index=3}
    API Key: 环境变量 DASHSCOPE_API_KEY :contentReference[oaicite:4]{index=4}
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-vl-ocr-latest",
        timeout_s: int = 60,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = os.getenv("DASHSCOPE_MODEL", model)
        self.timeout_s = timeout_s

    def recognize(self, image_path: str) -> OCRResult:
        t0 = time.time()
        if not self.api_key:
            return OCRResult(
                text="",
                provider="qwen",
                model=self.model,
                elapsed_ms=0,
                error="DASHSCOPE_API_KEY not set",
            )

        data_url = _encode_image_to_data_url(image_path)

        # OpenAI Chat Completions compatible endpoint
        url = f"{self.base_url}/chat/completions"

        # OCR prompt：让输出尽量“纯文本 + 保持阅读顺序”
        prompt = (
            "请你对这张图片进行文字识别（OCR）。"
            "要求：1) 输出图片中所有可见文字；2) 必须按从上到下、从左到右顺序；"
            "3) 只输出纯文本，不要解释。"
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
            resp.raise_for_status()
            raw = resp.json()
            text = (
                raw.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            elapsed_ms = int((time.time() - t0) * 1000)
            return OCRResult(
                text=text,
                provider="qwen",
                model=self.model,
                elapsed_ms=elapsed_ms,
                raw=raw,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            return OCRResult(
                text="",
                provider="qwen",
                model=self.model,
                elapsed_ms=elapsed_ms,
                error=f"{type(e).__name__}: {e}",
            )


class PageSourceTextProvider:
    """
    兜底方案：没有 OCR 时，从 Appium page_source 里提取可见 text。
    注意：微信小程序/Canvas 很多内容不在可访问性树里，这个仅作为替代。
    """

    def __init__(self):
        pass

    def recognize_from_page_source(self, page_source_xml: str) -> OCRResult:
        t0 = time.time()
        # Android UiAutomator2 常见：text="xxx" 或 content-desc="xxx"
        texts = re.findall(r'text="([^"]+)"', page_source_xml or "")
        descs = re.findall(r'content-desc="([^"]+)"', page_source_xml or "")
        merged = [t.strip() for t in (texts + descs) if t and t.strip()]
        # 去重但保持顺序
        seen = set()
        out = []
        for t in merged:
            if t not in seen:
                seen.add(t)
                out.append(t)

        elapsed_ms = int((time.time() - t0) * 1000)
        return OCRResult(
            text="\n".join(out),
            provider="pagesource",
            model="uiautomator2",
            elapsed_ms=elapsed_ms,
        )