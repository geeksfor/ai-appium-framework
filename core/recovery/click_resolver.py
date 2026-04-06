from __future__ import annotations

import math
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from core.perception.perception import PerceptionPack
from core.policy.policy_runner import PolicyRunner


@dataclass
class ClickPoint:
    x_pct: float
    y_pct: float
    source: str  # "bbox" | "ai"
    text: str = ""
    score: float = 0.0
    reason: str = ""
    bbox: Optional[Dict[str, float]] = None


class ClickResolver:
    """
    黑盒点击点解析器（项目收口版）：
    1) 优先从 OCR bbox 中找到最像的文本候选并计算中心点
    2) 找不到时，再让 PolicyRunner 生成一个 CLICK 动作兜底

    设计目标：
    - 保持现有接口：resolve_by_bbox / resolve_by_ai / resolve
    - 兼容 OCR box 的两种格式：x1/y1/x2/y2 或 bbox=[x1,y1,x2,y2]
    - 兼容百分比坐标与绝对像素坐标
    - 支持 locator_store + 内建别名 + UI hint 的区域偏好
    """

    def __init__(
        self,
        policy_runner: Optional[PolicyRunner] = None,
        locator_store_path: str = "core/heal/locator_store.yaml",
        candidate_threshold: float = 0.45,
    ):
        self.policy_runner = policy_runner
        self.locator_store_path = locator_store_path
        self.candidate_threshold = candidate_threshold
        self._locator_entries = self._load_locator_entries(locator_store_path)

        self._builtin_aliases: Dict[str, Dict[str, Any]] = {
            "close": {
                "aliases": ["关闭", "跳过", "点击跳过", "我知道了", "知道了", "稍后", "取消", "以后再说"],
                "preferred_regions": ["top_right", "center", "bottom_left"],
            },
            "agree": {
                "aliases": ["同意", "允许", "仅在使用期间", "始终允许", "确定", "继续", "开始使用"],
                "preferred_regions": ["bottom_center", "bottom_right"],
            },
            "next": {
                "aliases": ["下一步", "完成", "继续", "去看看", "开始"],
                "preferred_regions": ["bottom_center", "bottom_right"],
            },
            "back": {
                "aliases": ["返回", "上一步", "返回首页"],
                "preferred_regions": ["top_left", "center_left"],
            },
        }

    def resolve(
        self,
        pack: PerceptionPack,
        targets: List[str],
        ui_hints: Optional[List[str]] = None,
    ) -> Optional[ClickPoint]:
        cp = self.resolve_by_bbox(pack, targets, ui_hints=ui_hints)
        if cp is not None:
            return cp
        if not targets:
            return None
        return self.resolve_by_ai(pack, targets[0], ui_hints=ui_hints, target_candidates=targets)

    def resolve_by_bbox(
        self,
        pack: PerceptionPack,
        targets: List[str],
        ui_hints: Optional[List[str]] = None,
    ) -> Optional[ClickPoint]:
        boxes = self._read_boxes(pack)
        if not boxes:
            return None

        alias_pool, preferred_regions = self._build_alias_pool(targets, ui_hints=ui_hints)
        if not alias_pool:
            return None

        best: Optional[ClickPoint] = None

        for box in boxes:
            text = str(box.get("text", "") or "").strip()
            if not text:
                continue

            text_score = max((self._text_score(text, cand) for cand in alias_pool), default=0.0)
            if text_score < self.candidate_threshold:
                continue

            x1, y1, x2, y2 = self._normalize_box(box, pack.meta or {})
            cx = round((x1 + x2) / 2.0, 4)
            cy = round((y1 + y2) / 2.0, 4)

            pos_score = self._position_score(cx, cy, preferred_regions)
            score = round(min(1.0, text_score * 0.85 + pos_score * 0.15), 4)

            matched_alias = max(alias_pool, key=lambda cand: self._text_score(text, cand))
            reason = f'ocr命中“{text}”，最接近“{matched_alias}”，位置分={pos_score:.2f}'
            cp = ClickPoint(
                x_pct=cx,
                y_pct=cy,
                source="bbox",
                text=text,
                score=score,
                reason=reason,
                bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            )
            if best is None or cp.score > best.score:
                best = cp

        return best

    def resolve_by_ai(
        self,
        pack: PerceptionPack,
        target_text: str,
        ui_hints: Optional[List[str]] = None,
        target_candidates: Optional[List[str]] = None,
    ) -> Optional[ClickPoint]:
        if not self.policy_runner:
            return None

        alias_pool, _ = self._build_alias_pool(target_candidates or [target_text], ui_hints=ui_hints)
        synonyms = [x for x in alias_pool if x and x != target_text][:12]

        goal = {
            "intent": "CLICK_TEXT",
            "target_text": target_text,
            "synonyms": synonyms,
            "ui_hints": ui_hints or [],
            "constraints": {
                "max_steps": 1,
                "avoid_login": True,
                "prefer_close_over_enter": True,
            },
        }

        try:
            parsed = self.policy_runner.decide_next(
                pack=pack,
                goal=goal,
                recent={},
                hints={
                    "overlay_suspected": True,
                    "overlay_hints": ui_hints or [],
                    "no_progress": True,
                },
            )
        except Exception:
            return None

        if parsed.kind != "plan" or not parsed.plan or not parsed.plan.actions:
            return None

        a0 = parsed.plan.actions[0]
        a0_type = getattr(getattr(a0, "type", None), "value", getattr(a0, "type", None))
        if a0_type != "CLICK":
            return None

        x_pct = getattr(a0, "x_pct", None)
        y_pct = getattr(a0, "y_pct", None)

        if x_pct is None or y_pct is None:
            x = getattr(a0, "x", None)
            y = getattr(a0, "y", None)
            sw = self._meta_float(pack.meta or {}, ["screen_width", "width", "image_width"])
            sh = self._meta_float(pack.meta or {}, ["screen_height", "height", "image_height"])
            if x is not None and y is not None and sw and sh:
                x_pct = float(x) / sw
                y_pct = float(y) / sh
            else:
                return None

        text = str(
            getattr(a0, "target", None)
            or getattr(a0, "target_text", None)
            or getattr(a0, "selector", None)
            or target_text
        ).strip()

        return ClickPoint(
            x_pct=max(0.0, min(1.0, float(x_pct))),
            y_pct=max(0.0, min(1.0, float(y_pct))),
            source="ai",
            text=text,
            score=0.78,
            reason="PolicyRunner 返回 CLICK 坐标",
            bbox=None,
        )

    def _read_boxes(self, pack: PerceptionPack) -> List[Dict[str, Any]]:
        meta = pack.meta or {}
        boxes = meta.get("ocr_boxes", [])
        if not isinstance(boxes, list):
            return []
        return [b for b in boxes if isinstance(b, dict)]

    def _build_alias_pool(
        self,
        targets: List[str],
        ui_hints: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        alias_pool: List[str] = []
        preferred_regions: List[str] = []

        def add_text(v: str) -> None:
            s = str(v or "").strip()
            if s and s not in alias_pool:
                alias_pool.append(s)

        def add_region(v: str) -> None:
            s = str(v or "").strip()
            if s and s not in preferred_regions:
                preferred_regions.append(s)

        for t in targets:
            add_text(t)

        for entry in self._locator_entries:
            key = str(entry.get("key", "") or "").strip()
            aliases = [str(x).strip() for x in entry.get("aliases", []) if str(x).strip()]
            regions = [str(x).strip() for x in entry.get("preferred_regions", []) if str(x).strip()]

            for t in targets:
                if not t:
                    continue
                key_score = self._text_score(t, key) if key else 0.0
                alias_score = max((self._text_score(t, a) for a in aliases), default=0.0)
                if key_score >= 0.72 or alias_score >= 0.72:
                    if key:
                        add_text(key)
                    for a in aliases:
                        add_text(a)
                    for r in regions:
                        add_region(r)
                    break

        for group in self._builtin_aliases.values():
            aliases = group.get("aliases", [])
            hit = False
            for t in targets:
                if max((self._text_score(t, a) for a in aliases), default=0.0) >= 0.72:
                    hit = True
                    break
            if hit:
                for a in aliases:
                    add_text(a)
                for r in group.get("preferred_regions", []):
                    add_region(r)

        for r in self._hints_to_regions(ui_hints or []):
            add_region(r)

        return alias_pool, preferred_regions

    def _normalize_box(self, box: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[float, float, float, float]:
        if "bbox" in box and isinstance(box["bbox"], list) and len(box["bbox"]) == 4:
            x1, y1, x2, y2 = [float(v) for v in box["bbox"]]
        else:
            x1 = float(box.get("x1", 0.0))
            y1 = float(box.get("y1", 0.0))
            x2 = float(box.get("x2", 0.0))
            y2 = float(box.get("y2", 0.0))

        maxv = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if maxv > 1.5:
            sw = self._meta_float(meta, ["screen_width", "width", "image_width"]) or 1.0
            sh = self._meta_float(meta, ["screen_height", "height", "image_height"]) or 1.0
            x1, x2 = x1 / sw, x2 / sw
            y1, y2 = y1 / sh, y2 / sh

        return (
            round(self._clamp01(x1), 4),
            round(self._clamp01(y1), 4),
            round(self._clamp01(x2), 4),
            round(self._clamp01(y2), 4),
        )

    def _position_score(self, x_pct: float, y_pct: float, preferred_regions: List[str]) -> float:
        if not preferred_regions:
            return 0.50

        anchors = {
            "top_left": (0.15, 0.15),
            "top_center": (0.50, 0.15),
            "top_right": (0.85, 0.15),
            "center_left": (0.15, 0.50),
            "center": (0.50, 0.50),
            "center_right": (0.85, 0.50),
            "bottom_left": (0.15, 0.85),
            "bottom_center": (0.50, 0.85),
            "bottom_right": (0.85, 0.85),
        }

        best = 0.0
        for region in preferred_regions:
            anchor = anchors.get(region)
            if anchor is None:
                continue
            dist = math.sqrt((x_pct - anchor[0]) ** 2 + (y_pct - anchor[1]) ** 2)
            score = max(0.0, 1.0 - dist / 1.2)
            best = max(best, score)
        return round(best if best > 0 else 0.50, 4)

    def _hints_to_regions(self, ui_hints: List[str]) -> List[str]:
        joined = " ".join(str(x) for x in ui_hints if str(x).strip())
        regions: List[str] = []

        mapping = [
            (["右上", "右上角", "顶部右侧"], "top_right"),
            (["左上", "左上角", "顶部左侧"], "top_left"),
            (["顶部中间", "顶部居中"], "top_center"),
            (["底部右侧", "底部右边", "右下", "主按钮"], "bottom_right"),
            (["底部中间", "底部居中"], "bottom_center"),
            (["底部左侧", "左下"], "bottom_left"),
            (["中间", "遮罩", "蒙层"], "center"),
        ]
        for words, region in mapping:
            if any(w in joined for w in words) and region not in regions:
                regions.append(region)
        return regions

    def _load_locator_entries(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            raw_entries = data.get("entries", {})
            out: List[Dict[str, Any]] = []
            if isinstance(raw_entries, dict):
                for key, value in raw_entries.items():
                    if not isinstance(value, dict):
                        continue
                    out.append(
                        {
                            "key": str(key).strip(),
                            "aliases": [str(x).strip() for x in value.get("aliases", []) if str(x).strip()],
                            "preferred_regions": [str(x).strip() for x in value.get("preferred_regions", []) if str(x).strip()],
                        }
                    )
            return out
        except Exception:
            return []

    def _text_score(self, a: str, b: str) -> float:
        na = self._norm_text(a)
        nb = self._norm_text(b)
        if not na or not nb:
            return 0.0
        if na == nb:
            return 1.0
        if na in nb or nb in na:
            return 0.92
        return round(SequenceMatcher(None, na, nb).ratio(), 4)

    def _norm_text(self, s: str) -> str:
        return "".join(str(s or "").strip().lower().split())

    def _meta_float(self, meta: Dict[str, Any], keys: List[str]) -> Optional[float]:
        for k in keys:
            v = meta.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
                if fv > 0:
                    return fv
            except Exception:
                continue
        return None

    def _clamp01(self, v: float) -> float:
        return max(0.0, min(1.0, float(v)))
