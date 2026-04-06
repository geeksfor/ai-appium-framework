# core/perception/perception.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from core.perception.ocr import OCRProvider, OCRResult, OCRBoxesResult, PageSourceTextProvider


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

        # 1) 优先尝试带 bbox 的 OCR
        if hasattr(self.ocr_provider, "recognize_with_boxes"):
            try:
                r: OCRBoxesResult = self.ocr_provider.recognize_with_boxes(image_path)  # type: ignore[attr-defined]
                meta = {
                    "available": (r.error is None and bool((r.text or "").strip())),
                    "provider": r.provider,
                    "model": r.model,
                    "elapsed_ms": r.elapsed_ms,
                    "ocr_boxes": r.boxes or [],
                }
                if r.error:
                    meta["error"] = r.error
                return PerceptionPack(
                    image_path=image_path,
                    ocr_text=r.text or "",
                    meta=meta,
                )
            except Exception as e:
                # bbox OCR 失败时，自动降级为纯文本 OCR
                pass

        # 2) 纯文本 OCR
        r: OCRResult = self.ocr_provider.recognize(image_path)
        meta = {
            "available": (r.error is None and bool((r.text or "").strip())),
            "provider": r.provider,
            "model": r.model,
            "elapsed_ms": r.elapsed_ms,
        }
        if r.error:
            meta["error"] = r.error

        return PerceptionPack(image_path=image_path, ocr_text=r.text, meta=meta)

    @staticmethod
    def perceive_from_page_source(page_source_xml: str) -> PerceptionPack:
        prov = PageSourceTextProvider()
        r = prov.recognize_from_page_source(page_source_xml)
        meta = {
            "available": bool((r.text or "").strip()),
            "provider": r.provider,
            "model": r.model,
            "elapsed_ms": r.elapsed_ms,
        }
        return PerceptionPack(image_path="", ocr_text=r.text, meta=meta)