from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class TemplateMatch:
    x_pct: float
    y_pct: float
    score: float
    template_path: str
    bbox: dict


class TemplateMatcher:
    @staticmethod
    def available() -> bool:
        return cv2 is not None

    def __init__(self, threshold: float = 0.76, scales: Optional[Sequence[float]] = None):
        self.threshold = threshold
        self.scales = list(scales or [0.9, 1.0, 1.1])

    def match_first(
        self,
        image_path: str,
        template_paths: Iterable[str],
        preferred_regions: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> Optional[TemplateMatch]:
        if cv2 is None:
            return None

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        img_h, img_w = img.shape[:2]
        region = self._resolve_region(img_w, img_h, preferred_regions or [])
        x0, y0, x1, y1 = region
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        best: Optional[TemplateMatch] = None
        score_threshold = threshold if threshold is not None else self.threshold

        for template_path in template_paths:
            tp = Path(template_path)
            if not tp.exists():
                continue
            templ = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
            if templ is None:
                continue
            for scale in self.scales:
                if abs(scale - 1.0) < 1e-6:
                    scaled = templ
                else:
                    new_w = max(3, int(round(templ.shape[1] * scale)))
                    new_h = max(3, int(round(templ.shape[0] * scale)))
                    scaled = cv2.resize(templ, (new_w, new_h), interpolation=cv2.INTER_AREA)
                th, tw = scaled.shape[:2]
                if th >= roi.shape[0] or tw >= roi.shape[1]:
                    continue
                res = cv2.matchTemplate(roi, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val < score_threshold:
                    continue
                left = x0 + int(max_loc[0])
                top = y0 + int(max_loc[1])
                right = left + tw
                bottom = top + th
                cp = TemplateMatch(
                    x_pct=round((left + right) / 2.0 / img_w, 4),
                    y_pct=round((top + bottom) / 2.0 / img_h, 4),
                    score=round(float(max_val), 4),
                    template_path=str(tp),
                    bbox={
                        "x1": round(left / img_w, 4),
                        "y1": round(top / img_h, 4),
                        "x2": round(right / img_w, 4),
                        "y2": round(bottom / img_h, 4),
                    },
                )
                if best is None or cp.score > best.score:
                    best = cp
        return best

    def _resolve_region(self, w: int, h: int, preferred_regions: List[str]) -> Tuple[int, int, int, int]:
        if not preferred_regions:
            return (0, 0, w, h)
        region = preferred_regions[0]
        mapping = {
            "top_left": (0.0, 0.0, 0.4, 0.35),
            "top_center": (0.25, 0.0, 0.75, 0.32),
            "top_right": (0.6, 0.0, 1.0, 0.35),
            "center_left": (0.0, 0.25, 0.45, 0.75),
            "center": (0.15, 0.15, 0.85, 0.85),
            "center_right": (0.55, 0.25, 1.0, 0.75),
            "bottom_left": (0.0, 0.6, 0.45, 1.0),
            "bottom_center": (0.2, 0.58, 0.8, 1.0),
            "bottom_right": (0.55, 0.6, 1.0, 1.0),
            "bottom_primary_area": (0.15, 0.55, 0.85, 0.98),
        }
        box = mapping.get(region, (0.0, 0.0, 1.0, 1.0))
        return (int(w * box[0]), int(h * box[1]), int(w * box[2]), int(h * box[3]))
