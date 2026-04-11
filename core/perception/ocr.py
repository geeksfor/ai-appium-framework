from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import requests
from PIL import Image


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
    boxes: List[Dict[str, Any]]
    provider: str
    model: str
    elapsed_ms: int
    raw: Optional[dict] = None
    error: Optional[str] = None


class OCRProvider(Protocol):
    def recognize(self, image_path: str) -> OCRResult: ...


class OCRBoxesProvider(Protocol):
    def recognize_with_boxes(self, image_path: str) -> OCRBoxesResult: ...


def get_image_size(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as im:
        return int(im.size[0]), int(im.size[1])


def _encode_image_to_data_url(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_json_text(s: str) -> str:
    s = (s or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", s, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    obj = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if obj:
        return obj.group(1).strip()
    return s


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _to_ratio(v: Any, denom: float) -> float:
    fv = float(v)
    if abs(fv) <= 1.0:
        return _clamp01(fv)
    if denom <= 0:
        return _clamp01(fv)
    return _clamp01(fv / denom)


def _bbox_from_points(points: List[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 < x1 or y2 < y1:
        return None
    return {"x1": _clamp01(x1), "y1": _clamp01(y1), "x2": _clamp01(x2), "y2": _clamp01(y2)}


def _bbox_from_rotate_rect(rr: Any, *, image_width: Optional[int] = None, image_height: Optional[int] = None) -> Optional[Dict[str, float]]:
    if not isinstance(rr, list) or len(rr) < 4:
        return None
    try:
        cx = _to_ratio(rr[0], float(image_width or 0))
        cy = _to_ratio(rr[1], float(image_height or 0))
        w = abs(_to_ratio(rr[2], float(image_width or 0)))
        h = abs(_to_ratio(rr[3], float(image_height or 0)))
    except Exception:
        return None
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return _bbox_from_points([(x1, y1), (x2, y2)])


def _poly_to_norm_box(
    poly: Any,
    *,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    points: List[Tuple[float, float]] = []

    if isinstance(poly, list) and len(poly) >= 8 and all(isinstance(v, (int, float, str)) for v in poly[:8]):
        try:
            nums = [float(v) for v in poly]
        except Exception:
            nums = []
        for i in range(0, len(nums) - 1, 2):
            points.append(
                (
                    _to_ratio(nums[i], float(image_width or 0)),
                    _to_ratio(nums[i + 1], float(image_height or 0)),
                )
            )
        return _bbox_from_points(points)

    if isinstance(poly, list) and poly and all(isinstance(v, dict) for v in poly):
        for p in poly:
            if "x" not in p or "y" not in p:
                continue
            try:
                points.append(
                    (
                        _to_ratio(p["x"], float(image_width or 0)),
                        _to_ratio(p["y"], float(image_height or 0)),
                    )
                )
            except Exception:
                continue
        return _bbox_from_points(points)

    return None


def _coerce_box_dict(
    b: Dict[str, Any],
    *,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    try:
        if all(k in b for k in ("x1", "y1", "x2", "y2")):
            x1 = _to_ratio(b["x1"], float(image_width or 0))
            y1 = _to_ratio(b["y1"], float(image_height or 0))
            x2 = _to_ratio(b["x2"], float(image_width or 0))
            y2 = _to_ratio(b["y2"], float(image_height or 0))
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
            pts = [
                (_to_ratio(vals[0], float(image_width or 0)), _to_ratio(vals[1], float(image_height or 0))),
                (_to_ratio(vals[2], float(image_width or 0)), _to_ratio(vals[3], float(image_height or 0))),
            ]
            return _bbox_from_points(pts)

    for key in ("rotate_rect", "rotated_rect", "rbox"):
        if key in b:
            out = _bbox_from_rotate_rect(b[key], image_width=image_width, image_height=image_height)
            if out:
                return out

    for key in ("polygon", "poly", "quad", "points"):
        if key in b:
            out = _poly_to_norm_box(b[key], image_width=image_width, image_height=image_height)
            if out:
                return out

    return None


def _as_seq(obj: Any):
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return obj
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    try:
        return list(obj)
    except Exception:
        return [obj]


class QwenVisionOCRProvider:
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
            model=model or os.getenv("DASHSCOPE_BBOX_MODEL", "qwen-vl-ocr-latest"),
            timeout_s=timeout_s,
        )
        self.prompt_path = prompt_path

    def recognize_with_boxes(self, image_path: str) -> OCRBoxesResult:
        try:
            prompt = Path(self.prompt_path).read_text(encoding="utf-8")
        except Exception as e:
            return OCRBoxesResult(text="", boxes=[], provider="qwen", model=self.client.model, elapsed_ms=0, error=f"prompt load failed: {type(e).__name__}: {e}")

        img_w, img_h = get_image_size(image_path)
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
                    norm = _coerce_box_dict(b, image_width=img_w, image_height=img_h)
                    if not norm:
                        continue
                    boxes.append({"text": txt, **norm})
            return OCRBoxesResult(text=text, boxes=boxes, provider=r.provider, model=r.model, elapsed_ms=r.elapsed_ms, raw=r.raw)
        except Exception as e:
            return OCRBoxesResult(text="", boxes=[], provider=r.provider, model=r.model, elapsed_ms=r.elapsed_ms, raw=r.raw, error=f"bbox json parse failed: {type(e).__name__}: {e}")


class RapidOCRBoxesProvider:
    def __init__(self, det: bool = True, rec: bool = True):
        self.provider = "rapidocr"
        self.model = "rapidocr"
        self.det = det
        self.rec = rec
        self._engine = None
        self._import_error: Optional[str] = None

        try:
            from rapidocr import RapidOCR  # type: ignore
            self._engine = RapidOCR()
            self.model = "rapidocr"
            return
        except Exception as e:
            self._import_error = f"rapidocr init failed: {type(e).__name__}: {e}"

        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            self._engine = RapidOCR()
            self.model = "rapidocr-onnxruntime"
            return
        except Exception as e:
            self._import_error = ((self._import_error + " | ") if self._import_error else "") + f"rapidocr_onnxruntime init failed: {type(e).__name__}: {e}"

    def recognize(self, image_path: str) -> OCRResult:
        r = self.recognize_with_boxes(image_path)
        return OCRResult(text=r.text, provider=r.provider, model=r.model, elapsed_ms=r.elapsed_ms, raw=r.raw, error=r.error)

    def recognize_with_boxes(self, image_path: str) -> OCRBoxesResult:
        t0 = time.time()
        if self._engine is None:
            return OCRBoxesResult(text="", boxes=[], provider=self.provider, model=self.model, elapsed_ms=0, error=f"rapidocr not available: {self._import_error or 'unknown'}")

        img_w, img_h = get_image_size(image_path)
        try:
            result = self._engine(image_path)
            elapsed_ms = int((time.time() - t0) * 1000)

            boxes: List[Dict[str, Any]] = []
            lines: List[str] = []

            if hasattr(result, "boxes") and hasattr(result, "txts"):
                polys = _as_seq(getattr(result, "boxes", []))
                txts = _as_seq(getattr(result, "txts", []))
                scores = _as_seq(getattr(result, "scores", []))

                n = min(len(polys), len(txts))
                for i in range(n):
                    poly = _as_seq(polys[i])
                    txt = str(txts[i] or "").strip()
                    if not txt:
                        continue
                    score = None
                    if i < len(scores) and scores[i] is not None:
                        try:
                            score = float(scores[i])
                        except Exception:
                            score = None

                    pts = []
                    for p in poly:
                        p2 = _as_seq(p)
                        if len(p2) >= 2:
                            pts.append({"x": float(p2[0]), "y": float(p2[1])})

                    norm = _coerce_box_dict({"text": txt, "points": pts}, image_width=img_w, image_height=img_h)
                    if not norm:
                        continue

                    row = {"text": txt, **norm}
                    if score is not None:
                        row["score"] = round(score, 4)
                    boxes.append(row)
                    lines.append(txt)

                return OCRBoxesResult(text="\n".join(lines), boxes=boxes, provider=self.provider, model=self.model, elapsed_ms=elapsed_ms, raw={"items": len(boxes)})

            raw_res = result[0] if isinstance(result, tuple) else result
            if raw_res is not None and not isinstance(raw_res, list):
                for attr in ("ocr_res", "result", "results"):
                    if hasattr(raw_res, attr):
                        raw_res = getattr(raw_res, attr)
                        break

            seq = _as_seq(raw_res)
            if len(seq) == 0:
                return OCRBoxesResult(text="", boxes=[], provider=self.provider, model=self.model, elapsed_ms=elapsed_ms)

            for item in seq:
                row_item = _as_seq(item)
                if len(row_item) < 2:
                    continue
                poly = _as_seq(row_item[0])
                txt = str(row_item[1] or "").strip()
                score = None
                if len(row_item) > 2 and row_item[2] is not None:
                    try:
                        score = float(row_item[2])
                    except Exception:
                        score = None
                if not txt:
                    continue

                pts = []
                for p in _as_seq(poly):
                    p2 = _as_seq(p)
                    if len(p2) >= 2:
                        pts.append({"x": float(p2[0]), "y": float(p2[1])})

                norm = _coerce_box_dict({"text": txt, "points": pts}, image_width=img_w, image_height=img_h)
                if not norm:
                    continue

                row = {"text": txt, **norm}
                if score is not None:
                    row["score"] = round(score, 4)
                boxes.append(row)
                lines.append(txt)

            return OCRBoxesResult(text="\n".join(lines), boxes=boxes, provider=self.provider, model=self.model, elapsed_ms=elapsed_ms, raw={"items": len(boxes)})
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            return OCRBoxesResult(text="", boxes=[], provider=self.provider, model=self.model, elapsed_ms=elapsed_ms, error=f"{type(e).__name__}: {e}")


class EasyOCRBoxesProvider:
    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        self.provider = "easyocr"
        self.model = "easyocr"
        self.languages = languages or ["ch_sim", "en"]
        self.gpu = gpu
        self._reader = None
        self._import_error: Optional[str] = None
        try:
            import easyocr  # type: ignore
            self._reader = easyocr.Reader(self.languages, gpu=gpu)
        except Exception as e:
            self._import_error = f"{type(e).__name__}: {e}"

    def recognize(self, image_path: str) -> OCRResult:
        r = self.recognize_with_boxes(image_path)
        return OCRResult(text=r.text, provider=r.provider, model=r.model, elapsed_ms=r.elapsed_ms, raw=r.raw, error=r.error)

    def recognize_with_boxes(self, image_path: str) -> OCRBoxesResult:
        t0 = time.time()
        if self._reader is None:
            return OCRBoxesResult(text="", boxes=[], provider=self.provider, model=self.model, elapsed_ms=0, error=f"easyocr not available: {self._import_error or 'unknown'}")
        img_w, img_h = get_image_size(image_path)
        try:
            results = self._reader.readtext(image_path, detail=1)
            elapsed_ms = int((time.time() - t0) * 1000)
            boxes: List[Dict[str, Any]] = []
            lines: List[str] = []
            for item in results:
                if not item or len(item) < 2:
                    continue
                poly, txt = item[0], str(item[1] or "").strip()
                score = float(item[2]) if len(item) > 2 and item[2] is not None else None
                if not txt:
                    continue
                pts = []
                for p in _as_seq(poly):
                    p2 = _as_seq(p)
                    if len(p2) >= 2:
                        pts.append({"x": float(p2[0]), "y": float(p2[1])})
                norm = _coerce_box_dict({"text": txt, "points": pts}, image_width=img_w, image_height=img_h)
                if not norm:
                    continue
                row = {"text": txt, **norm}
                if score is not None:
                    row["score"] = round(score, 4)
                boxes.append(row)
                lines.append(txt)
            return OCRBoxesResult(text="\n".join(lines), boxes=boxes, provider=self.provider, model=self.model, elapsed_ms=elapsed_ms, raw={"items": len(boxes)})
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            return OCRBoxesResult(text="", boxes=[], provider=self.provider, model=self.model, elapsed_ms=elapsed_ms, error=f"{type(e).__name__}: {e}")


class PageSourceTextProvider:
    def recognize_from_page_source(self, page_source_xml: str) -> OCRResult:
        t0 = time.time()
        texts = re.findall(r'text="([^"]+)"', page_source_xml or "")
        descs = re.findall(r'content-desc="([^"]+)"', page_source_xml or "")
        resource_ids = re.findall(r'resource-id="([^"]+)"', page_source_xml or "")
        merged = [t.strip() for t in (texts + descs + resource_ids) if t and t.strip()]
        seen = set()
        out = []
        for t in merged:
            if t not in seen:
                seen.add(t)
                out.append(t)
        elapsed_ms = int((time.time() - t0) * 1000)
        return OCRResult(text="\n".join(out), provider="pagesource", model="uiautomator2", elapsed_ms=elapsed_ms)


def build_ocr_provider_from_env() -> Optional[OCRProvider]:
    wanted = str(os.getenv("OCR_PROVIDER", "")).strip().lower()

    def _try_local() -> Optional[OCRProvider]:
        rapid = RapidOCRBoxesProvider()
        if rapid._engine is not None:
            return rapid
        easy = EasyOCRBoxesProvider()
        if easy._reader is not None:
            return easy
        return None

    if wanted == "qwen":
        if os.getenv("DASHSCOPE_API_KEY"):
            return QwenOCRBoxesProvider(
                base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                model=os.getenv("DASHSCOPE_BBOX_MODEL", "qwen-vl-ocr-latest"),
            )
        return None

    if wanted == "rapidocr":
        return RapidOCRBoxesProvider()

    if wanted == "easyocr":
        return EasyOCRBoxesProvider()

    local = _try_local()
    if local is not None:
        return local

    if os.getenv("DASHSCOPE_API_KEY"):
        return QwenOCRBoxesProvider(
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.getenv("DASHSCOPE_BBOX_MODEL", "qwen-vl-ocr-latest"),
        )
    return None
