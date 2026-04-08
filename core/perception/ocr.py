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


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _norm_pair(x: Any, y: Any) -> tuple[float, float]:
    return _clamp01(float(x)), _clamp01(float(y))


def _bbox_from_points(points: List[tuple[float, float]]) -> Optional[Dict[str, float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 < x1 or y2 < y1:
        return None
    return {"x1": _clamp01(x1), "y1": _clamp01(y1), "x2": _clamp01(x2), "y2": _clamp01(y2)}


def _bbox_from_rotate_rect(rr: Any) -> Optional[Dict[str, float]]:
    if not isinstance(rr, list) or len(rr) < 4:
        return None
    try:
        cx = float(rr[0])
        cy = float(rr[1])
        w = abs(float(rr[2]))
        h = abs(float(rr[3]))
    except Exception:
        return None

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return _bbox_from_points([(_clamp01(x1), _clamp01(y1)), (_clamp01(x2), _clamp01(y2))])


def _bbox_from_poly(poly: Any) -> Optional[Dict[str, float]]:
    if isinstance(poly, list) and len(poly) >= 8 and all(isinstance(v, (int, float, str)) for v in poly[:8]):
        try:
            nums = [float(v) for v in poly]
        except Exception:
            return None
        pts = [_norm_pair(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        return _bbox_from_points(pts)

    if isinstance(poly, list) and poly and all(isinstance(v, dict) for v in poly):
        pts: List[tuple[float, float]] = []
        for p in poly:
            if "x" not in p or "y" not in p:
                continue
            try:
                pts.append(_norm_pair(p["x"], p["y"]))
            except Exception:
                continue
        return _bbox_from_points(pts)
    return None


def _coerce_box_dict(b: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        if all(k in b for k in ("x1", "y1", "x2", "y2")):
            x1 = _clamp01(float(b["x1"]))
            y1 = _clamp01(float(b["y1"]))
            x2 = _clamp01(float(b["x2"]))
            y2 = _clamp01(float(b["y2"]))
            if x2 >= x1 and y2 >= y1:
                return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    except Exception:
        pass

    if "bbox" in b and isinstance(b["bbox"], list) and len(b["bbox"]) == 4:
        try:
            vals = [float(v) for v in b["bbox"]]
        except Exception:
            vals = []
        if vals:
            return _bbox_from_points([_norm_pair(vals[0], vals[1]), _norm_pair(vals[2], vals[3])])

    for key in ("rotate_rect", "rotated_rect", "rbox"):
        if key in b:
            out = _bbox_from_rotate_rect(b[key])
            if out:
                return out

    for key in ("polygon", "poly", "quad", "points"):
        if key in b:
            out = _bbox_from_poly(b[key])
            if out:
                return out

    return None


class QwenVisionOCRProvider:
    """
    通用 Qwen 视觉 OCR Provider：
    - recognize(): 纯文本 OCR
    - recognize_with_prompt(): 用自定义 prompt 让模型输出指定格式
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
        self.model = model or os.getenv("DASHSCOPE_MODEL", "qwen-vl-max")
        self.timeout_s = timeout_s

    def _post_chat(self, image_path: str, prompt: str) -> OCRResult:
        t0 = time.time()

        if not self.api_key:
            return OCRResult(text="", provider="qwen", model=self.model, elapsed_ms=0, error="DASHSCOPE_API_KEY not set")

        data_url = _encode_image_to_data_url(image_path)
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload, ensure_ascii=False), timeout=self.timeout_s)
            resp.raise_for_status()
            raw = resp.json()
            text = raw.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            elapsed_ms = int((time.time() - t0) * 1000)
            return OCRResult(text=text, provider="qwen", model=self.model, elapsed_ms=elapsed_ms, raw=raw)
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            return OCRResult(text="", provider="qwen", model=self.model, elapsed_ms=elapsed_ms, error=f"{type(e).__name__}: {e}")

    def recognize_with_prompt(self, image_path: str, prompt: str) -> OCRResult:
        return self._post_chat(image_path=image_path, prompt=prompt)

    def recognize(self, image_path: str) -> OCRResult:
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
        try:
            prompt = Path(self.prompt_path).read_text(encoding="utf-8")
        except Exception as e:
            return OCRBoxesResult(text="", boxes=[], provider="qwen", model=self.client.model, elapsed_ms=0, error=f"prompt load failed: {type(e).__name__}: {e}")

        r = self.client.recognize_with_prompt(image_path=image_path, prompt=prompt)
        if r.error:
            return OCRBoxesResult(text="", boxes=[], provider=r.provider, model=r.model, elapsed_ms=r.elapsed_ms, raw=r.raw, error=r.error)

        try:
            obj = json.loads(_extract_json_text(r.text))
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
                    norm = _coerce_box_dict(b)
                    if not norm:
                        continue
                    boxes.append({"text": txt, **norm})
            return OCRBoxesResult(text=text, boxes=boxes, provider=r.provider, model=r.model, elapsed_ms=r.elapsed_ms, raw=r.raw)
        except Exception as e:
            return OCRBoxesResult(text="", boxes=[], provider=r.provider, model=r.model, elapsed_ms=r.elapsed_ms, raw=r.raw, error=f"bbox json parse failed: {type(e).__name__}: {e}")


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
        return OCRResult(text="\n".join(out), provider="pagesource", model="uiautomator2", elapsed_ms=elapsed_ms)
