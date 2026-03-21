# core/perception/perception.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from core.perception.ocr import OCRProvider, OCRResult, PageSourceTextProvider


@dataclass
class PerceptionPack:
    image_path: str
    ocr_text: str
    meta: Dict[str, Any]


class Perception:
    def __init__(self, ocr_provider: Optional[OCRProvider] = None):
        self.ocr_provider = ocr_provider

    def perceive_image(self, image_path: str) -> PerceptionPack:
        if not self.ocr_provider:
            return PerceptionPack(
                image_path=image_path,
                ocr_text="",
                meta={"available": False, "reason": "ocr_provider not configured"},
            )

        r: OCRResult = self.ocr_provider.recognize(image_path)
        meta = {
            "available": (r.error is None and bool(r.text.strip())),
            "provider": r.provider,
            "model": r.model,
            "elapsed_ms": r.elapsed_ms,
        }
        if r.error:
            meta["error"] = r.error
        return PerceptionPack(image_path=image_path, ocr_text=r.text, meta=meta)

    @staticmethod
    def perceive_from_page_source(page_source_xml: str) -> PerceptionPack:
        # 兜底：没有 OCR 时，把 pageSource 当成“可见文本来源”
        prov = PageSourceTextProvider()
        r = prov.recognize_from_page_source(page_source_xml)
        meta = {
            "available": bool(r.text.strip()),
            "provider": r.provider,
            "model": r.model,
            "elapsed_ms": r.elapsed_ms,
        }
        return PerceptionPack(image_path="", ocr_text=r.text, meta=meta)