# core/perception/ocr.py
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import requests


@dataclass
class OCRResult:
    text: str
    provider: str
    model: str
    elapsed_ms: int
    raw: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class OCRBoxesResult:
    text: str
    boxes: List[Dict[str, Any]]  # [{"text","x1","y1","x2","y2"}]
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
    return f"data:{mime};base64,{b64}"


def _extract_json_text(s: str) -> str:
    """
    容错提取 JSON：
    - 支持 ```json ... ```
    - 支持前后夹杂说明文字
    """
    s = (s or "").strip()

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if fence:
        return fence.group(1).strip()

    obj = re.search(r"(\{.*\})", s, re.DOTALL)
    if obj:
        return obj.group(1).strip()

    return s


class QwenVisionOCRProvider:
    """
    通用 Qwen 视觉 OCR Provider：
    - recognize(): 纯文本 OCR
    - recognize_with_prompt(): 用自定义 prompt 让模型输出指定格式

    环境变量：
    - DASHSCOPE_API_KEY
    - DASHSCOPE_BASE_URL
    - DASHSCOPE_MODEL
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 60,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        self.model = model or os.getenv("DASHSCOPE_MODEL", "qwen-vl-ocr-latest")
        self.timeout_s = timeout_s

    def _post_chat(self, image_path: str, prompt: str) -> OCRResult:
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
        url = f"{self.base_url}/chat/completions"

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
            resp = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False),
                timeout=self.timeout_s,
            )
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

    def recognize_with_prompt(self, image_path: str, prompt: str) -> OCRResult:
        """
        自定义 prompt 版本。
        你后面做 bbox OCR、结构化 OCR 都走这个。
        """
        return self._post_chat(image_path=image_path, prompt=prompt)

    def recognize(self, image_path: str) -> OCRResult:
        """
        纯文本 OCR。
        """
        prompt = (
            "请你对这张图片进行文字识别（OCR）。"
            "要求：1) 输出图片中所有可见文字；"
            "2) 尽量按从上到下、从左到右顺序；"
            "3) 只输出纯文本，不要解释。"
        )
        return self._post_chat(image_path=image_path, prompt=prompt)


class QwenOCRBoxesProvider:
    """
    视觉 OCR + bbox Provider。
    使用自定义 prompt 输出：
    {
      "text": "...",
      "boxes": [{"text":"...", "x1":0.1, "y1":0.2, "x2":0.3, "y2":0.4}]
    }
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        prompt_path: str = "assets/prompts/ocr_boxes_v1.txt",
        timeout_s: int = 60,
    ):
        self.client = QwenVisionOCRProvider(
            api_key=api_key,
            base_url=base_url,
            model=model or os.getenv("DASHSCOPE_BBOX_MODEL", "qwen3.5-plus"),
            timeout_s=timeout_s,
        )
        self.prompt_path = prompt_path

    def recognize_with_boxes(self, image_path: str) -> OCRBoxesResult:
        t0 = time.time()

        try:
            prompt = Path(self.prompt_path).read_text(encoding="utf-8")
        except Exception as e:
            return OCRBoxesResult(
                text="",
                boxes=[],
                provider="qwen",
                model=self.client.model,
                elapsed_ms=0,
                error=f"prompt load failed: {type(e).__name__}: {e}",
            )

        r = self.client.recognize_with_prompt(image_path=image_path, prompt=prompt)
        if r.error:
            return OCRBoxesResult(
                text="",
                boxes=[],
                provider=r.provider,
                model=r.model,
                elapsed_ms=r.elapsed_ms,
                raw=r.raw,
                error=r.error,
            )

        try:
            jtxt = _extract_json_text(r.text)
            obj = json.loads(jtxt)

            text = str(obj.get("text", "") or "").strip()
            boxes_raw = obj.get("boxes", [])

            boxes: List[Dict[str, Any]] = []
            if isinstance(boxes_raw, list):
                for b in boxes_raw:
                    if not isinstance(b, dict):
                        continue
                    txt = str(b.get("text", "") or "").strip()
                    if not txt:
                        continue
                    try:
                        x1 = max(0.0, min(1.0, float(b["x1"])))
                        y1 = max(0.0, min(1.0, float(b["y1"])))
                        x2 = max(0.0, min(1.0, float(b["x2"])))
                        y2 = max(0.0, min(1.0, float(b["y2"])))
                    except Exception:
                        continue

                    if x2 < x1 or y2 < y1:
                        continue

                    boxes.append(
                        {
                            "text": txt,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )

            return OCRBoxesResult(
                text=text,
                boxes=boxes,
                provider=r.provider,
                model=r.model,
                elapsed_ms=r.elapsed_ms,
                raw=r.raw,
            )
        except Exception as e:
            return OCRBoxesResult(
                text="",
                boxes=[],
                provider=r.provider,
                model=r.model,
                elapsed_ms=r.elapsed_ms,
                raw=r.raw,
                error=f"bbox json parse failed: {type(e).__name__}: {e}",
            )


class PageSourceTextProvider:
    """
    兜底方案：没有 OCR 时，从 Appium page_source 里提取可见 text。
    """

    def recognize_from_page_source(self, page_source_xml: str) -> OCRResult:
        t0 = time.time()

        texts = re.findall(r'text="([^"]+)"', page_source_xml or "")
        descs = re.findall(r'content-desc="([^"]+)"', page_source_xml or "")

        merged = [t.strip() for t in (texts + descs) if t and t.strip()]

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