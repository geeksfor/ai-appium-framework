from __future__ import annotations

import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image

from core.perception.ocr import build_ocr_provider_from_env
from core.perception.perception import PerceptionPack
from core.policy.policy_runner import PolicyRunner
from core.recovery.regions import REGIONS, get_region_box, merge_region_hints


@dataclass
class ClickPoint:
    x_pct: float
    y_pct: float
    source: str
    text: str = ""
    score: float = 0.0
    reason: str = ""
    bbox: Optional[Dict[str, float]] = None
    target_role: str = "button"


class ClickResolver:
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
        self.last_candidates: List[ClickPoint] = []
        self._region_defs: Dict[str, Tuple[float, float, float, float]] = {
            name: get_region_box(name) for name in REGIONS.keys()
        }
        # 为兼容旧逻辑保留 full。
        self._region_defs.setdefault("full", (0.0, 0.0, 1.0, 1.0))
        self._builtin_aliases: Dict[str, Dict[str, Any]] = {
            "close": {"aliases": ["关闭", "跳过", "点击跳过", "我知道了", "知道了", "稍后", "取消"], "preferred_regions": ["top_right", "center_dialog"]},
            "back": {"aliases": ["返回", "上一步", "返回首页"], "preferred_regions": ["top_left"]},
            "login_button": {"aliases": ["登录", "立即登录", "去登录"], "preferred_regions": ["bottom_primary_area"]},
            "confirm": {"aliases": ["确定", "确认", "完成", "继续", "下一步", "提交"], "preferred_regions": ["bottom_primary_area", "center_dialog"]},
            "username": {"aliases": ["用户名", "账号", "手机号"], "preferred_regions": ["form_input_area"]},
            "password": {"aliases": ["密码", "请输入6~20位密码", "请输入密码"], "preferred_regions": ["form_input_area"]},
        }

    def resolve_semantic(
        self,
        pack: PerceptionPack,
        primary_target: str,
        *,
        logical_name: str = "",
        target_type: str = "auto",
        text_candidates: Optional[List[str]] = None,
        region_hints: Optional[List[str]] = None,
        target_role: str = "button",
    ) -> Optional[ClickPoint]:
        alias_pool, preferred_regions = self._build_alias_pool(
            targets=[primary_target] + list(text_candidates or []),
            logical_name=logical_name,
            ui_hints=region_hints,
            target_role=target_role,
        )
        candidates: List[ClickPoint] = []
        disable_native = str(os.getenv("CLICK_DISABLE_NATIVE_XML", "")).strip().lower() in {"1", "true", "yes"}
        if not disable_native:
            native_candidates = self._resolve_by_page_source_candidates(
                pack,
                alias_pool,
                preferred_regions,
                logical_name=logical_name,
                target_role=target_role,
            )
            candidates.extend(native_candidates)
            if native_candidates:
                best_native = self._pick_best(native_candidates)
                self.last_candidates = sorted(native_candidates, key=lambda c: c.score, reverse=True)
                return best_native

        full_ocr_candidates = self._resolve_by_ocr_candidates(
            pack,
            alias_pool,
            preferred_regions,
            target_role=target_role,
            source_name="ocr",
            primary_target=primary_target,
            strict_semantic=True,
        )
        candidates.extend(full_ocr_candidates)

        # 对“登录/提交/确认/下一步”等主按钮，如果全屏 OCR 没有强命中，则继续做区域 OCR。
        if pack.image_path and self._should_retry_region_ocr(primary_target, alias_pool, target_role, full_ocr_candidates):
            region_candidates = self._resolve_by_region_ocr(
                pack,
                alias_pool,
                preferred_regions,
                target_role=target_role,
                primary_target=primary_target,
            )
            candidates.extend(region_candidates)

        best = self._pick_best(candidates)
        self.last_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        if best is not None:
            return best

        ai = self._resolve_by_ai_semantic(pack, primary_target, logical_name, alias_pool, preferred_regions, target_role=target_role)
        if ai and ai.get("x_pct") is not None and ai.get("y_pct") is not None:
            cp = ClickPoint(
                x_pct=self._clamp01(float(ai["x_pct"])),
                y_pct=self._clamp01(float(ai["y_pct"])),
                source="ai-point",
                text=str(ai.get("text") or primary_target or "").strip(),
                score=round(float(ai.get("score", 0.65)), 4),
                reason=str(ai.get("reason") or "AI 直接给出了坐标").strip(),
                bbox=None,
                target_role=target_role,
            )
            self.last_candidates = [cp]
            return cp

        if ai:
            rerun_targets = self._dedupe_texts(alias_pool + list(ai.get("text_candidates") or []))
            rerun_regions = self._dedupe_texts(preferred_regions + list(ai.get("region_hints") or []))
            rerun = self._resolve_by_region_ocr(pack, rerun_targets, rerun_regions, target_role=target_role)
            best = self._pick_best(rerun)
            self.last_candidates = sorted(rerun, key=lambda c: c.score, reverse=True)
            return best
        return None

    def _resolve_by_page_source_candidates(
        self,
        pack: PerceptionPack,
        alias_pool: List[str],
        preferred_regions: List[str],
        *,
        logical_name: str = "",
        target_role: str = "button",
    ) -> List[ClickPoint]:
        nodes = self._read_page_source_nodes(pack)
        out: List[ClickPoint] = []
        for node in nodes:
            text_value = str(node.get("text") or "").strip()
            desc_value = str(node.get("content_desc") or "").strip()
            rid_value = str(node.get("resource_id") or "").strip()
            class_name = str(node.get("class_name") or "").strip().lower()
            haystacks = [x for x in [text_value, desc_value, rid_value] if x]
            if not haystacks:
                continue
            match_score = max((self._text_score(h, cand) for h in haystacks for cand in alias_pool), default=0.0)
            if logical_name and rid_value:
                match_score = max(match_score, self._text_score(rid_value, logical_name))
            if match_score < self.candidate_threshold:
                continue
            x1, y1, x2, y2 = self._normalize_box(node, pack.meta or {})
            if x2 <= x1 or y2 <= y1:
                continue
            if not self._within_preferred_regions((x1 + x2) / 2.0, (y1 + y2) / 2.0, preferred_regions):
                region_penalty = 0.12
            else:
                region_penalty = 0.0
            is_input = "edittext" in class_name or node.get("focusable")
            is_clickable = bool(node.get("clickable")) or "button" in class_name or is_input
            if target_role == "input" and not is_input:
                continue
            if target_role != "input" and not is_clickable:
                continue
            cx, cy = self._pick_click_point_from_box(text_value or desc_value or rid_value, x1, y1, x2, y2, preferred_regions, target_role=target_role)
            pos_score = self._position_score(cx, cy, preferred_regions)
            score = round(min(1.0, match_score * 0.68 + pos_score * 0.18 + (0.12 if is_clickable else 0.0) - region_penalty), 4)
            label = text_value or desc_value or rid_value or logical_name
            out.append(ClickPoint(x_pct=cx, y_pct=cy, source="native-xml", text=label, score=score, reason=f"pageSource命中 {label}", bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2}, target_role=target_role))
        return out

    def _resolve_by_ocr_candidates(
        self,
        pack: PerceptionPack,
        alias_pool: List[str],
        preferred_regions: List[str],
        *,
        target_role: str,
        source_name: str,
        primary_target: str = "",
        strict_semantic: bool = False,
    ) -> List[ClickPoint]:
        out: List[ClickPoint] = []
        for box in self._read_boxes(pack):
            text = str(box.get("text", "") or "").strip()
            if not text:
                continue
            match_score = max((self._text_score(text, cand) for cand in alias_pool), default=0.0)
            if match_score < self.candidate_threshold:
                continue
            if not self._is_semantic_candidate_valid(
                text,
                primary_target=primary_target,
                alias_pool=alias_pool,
                target_role=target_role,
                strict=strict_semantic,
            ):
                continue
            x1, y1, x2, y2 = self._normalize_box(box, pack.meta or {})
            if x2 <= x1 or y2 <= y1:
                continue
            if not self._is_box_reasonable(text, x1, y1, x2, y2, preferred_regions, target_role):
                continue
            cx, cy = self._pick_click_point_from_box(text, x1, y1, x2, y2, preferred_regions, target_role=target_role)
            pos_score = self._position_score(cx, cy, preferred_regions)
            edge_penalty = self._edge_penalty(x1, y1, x2, y2)
            score = round(min(1.0, match_score * 0.68 + pos_score * 0.20 + 0.10 - edge_penalty), 4)
            out.append(ClickPoint(x_pct=cx, y_pct=cy, source=source_name, text=text, score=score, reason=f"{source_name}命中 {text}", bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2}, target_role=target_role))
        return out

    def _resolve_by_region_ocr(
        self,
        pack: PerceptionPack,
        alias_pool: List[str],
        preferred_regions: List[str],
        *,
        target_role: str,
        primary_target: str = "",
    ) -> List[ClickPoint]:
        provider = build_ocr_provider_from_env()
        if provider is None or not hasattr(provider, "recognize_with_boxes") or not pack.image_path:
            return []
        image_path = str(pack.image_path)
        regions = preferred_regions or self._default_regions_for_role(alias_pool, target_role)
        out: List[ClickPoint] = []
        for region_name in regions[:3]:
            bounds = self._region_defs.get(region_name)
            if not bounds:
                continue
            region_result = self._run_region_ocr(provider, image_path, bounds)
            if not region_result:
                continue
            full_boxes = region_result["full_boxes"]
            meta = dict(pack.meta or {})
            meta["screen_width"] = region_result["image_width"]
            meta["screen_height"] = region_result["image_height"]
            temp_pack = PerceptionPack(image_path=image_path, ocr_text=region_result.get("text", ""), meta={**meta, "ocr_boxes": full_boxes})
            region_candidates = self._resolve_by_ocr_candidates(
                temp_pack,
                alias_pool,
                [region_name],
                target_role=target_role,
                source_name="ocr-region",
                primary_target=primary_target,
                strict_semantic=False,
            )
            out.extend(region_candidates)
        return out

    def _should_retry_region_ocr(
        self,
        primary_target: str,
        alias_pool: List[str],
        target_role: str,
        full_ocr_candidates: List[ClickPoint],
    ) -> bool:
        if target_role == "input":
            return False
        if not self._is_primary_action_target(primary_target, alias_pool):
            return False
        if not full_ocr_candidates:
            return True
        best = self._pick_best(full_ocr_candidates)
        if best is None:
            return True
        # 只有强语义命中才允许直接收手；否则继续区域 OCR。
        return not self._is_semantic_candidate_valid(
            best.text,
            primary_target=primary_target,
            alias_pool=alias_pool,
            target_role=target_role,
            strict=True,
        )

    def _is_primary_action_target(self, primary_target: str, alias_pool: List[str]) -> bool:
        merged = " ".join([primary_target] + list(alias_pool or []))
        return any(x in merged for x in ["登录", "提交", "确认", "完成", "下一步", "继续", "保存"])

    def _contains_cjk(self, s: str) -> bool:
        return bool(re.search(r"[一-鿿]", str(s or "")))

    def _is_semantic_candidate_valid(
        self,
        text: str,
        *,
        primary_target: str,
        alias_pool: List[str],
        target_role: str,
        strict: bool,
    ) -> bool:
        cand = str(text or "").strip()
        if not cand:
            return False
        if target_role == "input":
            return True
        # 按钮类候选，先去掉明显噪声：单字符、纯字母角标、状态栏时间等。
        if len(cand) <= 1:
            return False
        if re.fullmatch(r"[A-Za-z]+", cand):
            return False
        if re.fullmatch(r"[\d:/.%-]+", cand):
            return False

        meaningful_aliases: List[str] = []
        for a in [primary_target] + list(alias_pool or []):
            s = str(a or "").strip()
            if not s:
                continue
            # 过滤像 T 这种太短、容易误命中的 alias
            if len(s) <= 1 and s != cand:
                continue
            meaningful_aliases.append(s)
        if not meaningful_aliases:
            meaningful_aliases = [primary_target] if primary_target else []

        best = max((self._text_score(cand, a) for a in meaningful_aliases), default=0.0)
        if strict:
            # 全屏 OCR 阶段更严格：要求强匹配，避免 T/用户名 之类误拦截掉区域 OCR。
            return best >= 0.88
        return best >= max(self.candidate_threshold, 0.55)

    def _resolve_by_ai_semantic(
        self,
        pack: PerceptionPack,
        primary_target: str,
        logical_name: str,
        text_candidates: List[str],
        region_hints: List[str],
        *,
        target_role: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.policy_runner:
            return None
        goal = {
            "intent": "CLICK_TEXT" if target_role != "input" else "INPUT_TEXT",
            "target_text": primary_target or logical_name or (text_candidates[0] if text_candidates else "目标控件"),
            "synonyms": text_candidates[:10],
            "ui_hints": region_hints[:6],
            "constraints": {"max_steps": 1, "prefer_semantic_click": True, "prefer_region": region_hints[0] if region_hints else None, "target_role": target_role},
        }
        try:
            parsed = self.policy_runner.decide_next(pack=pack, goal=goal, recent={}, hints={"prefer_semantic_click": True, "overlay_hints": region_hints, "target_role": target_role})
        except Exception:
            return None
        if parsed.kind != "plan" or not parsed.plan or not parsed.plan.actions:
            return None
        a0 = parsed.plan.actions[0]
        x_pct = getattr(a0, "x_pct", None)
        y_pct = getattr(a0, "y_pct", None)
        if x_pct is not None and y_pct is not None:
            return {"text": str(getattr(a0, "target", None) or getattr(a0, "target_text", None) or primary_target).strip(), "x_pct": x_pct, "y_pct": y_pct, "score": 0.65, "reason": "AI 兜底直接给坐标"}
        out = {
            "text": str(getattr(a0, "target", None) or getattr(a0, "target_text", None) or primary_target).strip(),
            "text_candidates": [str(x).strip() for x in getattr(a0, "text_candidates", []) if str(x).strip()],
            "region_hints": [str(x).strip() for x in getattr(a0, "region_hints", []) if str(x).strip()],
            "score": 0.58,
            "reason": "AI 给了语义目标和区域",
        }
        if not out["region_hints"] and region_hints:
            out["region_hints"] = region_hints
        return out

    def _run_region_ocr(self, provider: Any, image_path: str, bounds: Tuple[float, float, float, float]) -> Optional[Dict[str, Any]]:
        x1p, y1p, x2p, y2p = bounds
        with Image.open(image_path) as im:
            full_w, full_h = im.size
            left = int(round(full_w * x1p))
            top = int(round(full_h * y1p))
            right = int(round(full_w * x2p))
            bottom = int(round(full_h * y2p))
            crop = im.crop((left, top, right, bottom))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                crop_path = tf.name
            crop.save(crop_path)
        try:
            result = provider.recognize_with_boxes(crop_path)
            crop_boxes = result.boxes or []
            full_boxes: List[Dict[str, Any]] = []
            crop_w = max(1, right - left)
            crop_h = max(1, bottom - top)
            for b in crop_boxes:
                try:
                    bx1 = float(b["x1"])
                    by1 = float(b["y1"])
                    bx2 = float(b["x2"])
                    by2 = float(b["y2"])
                except Exception:
                    continue
                fx1 = (left + bx1 * crop_w) / full_w
                fy1 = (top + by1 * crop_h) / full_h
                fx2 = (left + bx2 * crop_w) / full_w
                fy2 = (top + by2 * crop_h) / full_h
                row = {"text": b.get("text", ""), "x1": self._clamp01(fx1), "y1": self._clamp01(fy1), "x2": self._clamp01(fx2), "y2": self._clamp01(fy2)}
                if "score" in b:
                    row["score"] = b["score"]
                full_boxes.append(row)
            return {"text": result.text or "", "full_boxes": full_boxes, "image_width": full_w, "image_height": full_h, "error": result.error}
        except Exception:
            return None
        finally:
            try:
                os.remove(crop_path)
            except Exception:
                pass

    def _read_boxes(self, pack: PerceptionPack) -> List[Dict[str, Any]]:
        meta = pack.meta or {}
        boxes = meta.get("ocr_boxes", [])
        if not isinstance(boxes, list):
            return []
        return [b for b in boxes if isinstance(b, dict)]

    def _read_page_source_nodes(self, pack: PerceptionPack) -> List[Dict[str, Any]]:
        path = self._find_page_source_path(pack)
        if not path or not path.exists():
            return []
        try:
            root = ET.fromstring(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        self._walk_nodes(root, out, pack.meta or {})
        return out

    def _walk_nodes(self, elem: Any, out: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        attrs = dict(elem.attrib or {})
        bounds = self._parse_bounds(attrs.get("bounds", ""), meta)
        out.append({
            "text": attrs.get("text", ""),
            "content_desc": attrs.get("content-desc", ""),
            "resource_id": attrs.get("resource-id", ""),
            "class_name": attrs.get("class", ""),
            "clickable": str(attrs.get("clickable", "")).lower() == "true",
            "focusable": str(attrs.get("focusable", "")).lower() == "true",
            "enabled": str(attrs.get("enabled", "")).lower() != "false",
            **bounds,
        })
        for child in list(elem):
            self._walk_nodes(child, out, meta)

    def _find_page_source_path(self, pack: PerceptionPack) -> Optional[Path]:
        meta = pack.meta or {}
        p = str(meta.get("page_source_path") or "").strip()
        if p:
            pp = Path(p)
            if pp.exists():
                return pp
        step_dir = str(meta.get("step_dir") or "").strip()
        if step_dir:
            base = Path(step_dir)
            for name in ["page_source_after.xml", "page_source.xml", "page_source_fail.xml"]:
                p2 = base / name
                if p2.exists():
                    return p2
        if pack.image_path:
            base = Path(pack.image_path).parent
            for name in ["page_source_after.xml", "page_source.xml", "page_source_fail.xml"]:
                p2 = base / name
                if p2.exists():
                    return p2
        return None

    def _parse_bounds(self, raw: str, meta: Dict[str, Any]) -> Dict[str, float]:
        m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", str(raw or ""))
        if not m:
            return {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}
        sw = self._meta_float(meta, ["screen_width", "width", "image_width"]) or 1.0
        sh = self._meta_float(meta, ["screen_height", "height", "image_height"]) or 1.0
        x1, y1, x2, y2 = [float(x) for x in m.groups()]
        return {"x1": self._clamp01(x1 / sw), "y1": self._clamp01(y1 / sh), "x2": self._clamp01(x2 / sw), "y2": self._clamp01(y2 / sh)}

    def _load_locator_entries(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw_entries = data.get("entries", {})
        out: List[Dict[str, Any]] = []
        if isinstance(raw_entries, dict):
            for key, value in raw_entries.items():
                if not isinstance(value, dict):
                    continue
                out.append({
                    "key": str(key).strip(),
                    "aliases": [str(x).strip() for x in value.get("aliases", []) if str(x).strip()],
                    "preferred_regions": [str(x).strip() for x in value.get("preferred_regions", []) if str(x).strip()],
                })
        return out

    def _build_alias_pool(
        self,
        targets: List[str],
        *,
        logical_name: str = "",
        ui_hints: Optional[List[str]] = None,
        target_role: str = "button",
    ) -> Tuple[List[str], List[str]]:
        alias_pool: List[str] = []
        explicit_regions: List[str] = []

        def add_text(v: str) -> None:
            s = str(v or "").strip()
            if s and s not in alias_pool:
                alias_pool.append(s)

        def add_region(v: str) -> None:
            s = str(v or "").strip()
            if s in self._region_defs and s not in explicit_regions:
                explicit_regions.append(s)

        for t in targets:
            add_text(t)
        if logical_name:
            add_text(logical_name)

        for entry in self._locator_entries:
            key = str(entry.get("key", "") or "").strip()
            aliases = [str(x).strip() for x in entry.get("aliases", []) if str(x).strip()]
            regions = [str(x).strip() for x in entry.get("preferred_regions", []) if str(x).strip()]
            hit = False
            if logical_name and key == logical_name:
                hit = True
            for t in targets:
                key_score = self._text_score(t, key) if key else 0.0
                alias_score = max((self._text_score(t, a) for a in aliases), default=0.0)
                if key_score >= 0.72 or alias_score >= 0.72:
                    hit = True
                    break
            if hit:
                if key:
                    add_text(key)
                for a in aliases:
                    add_text(a)
                for r in regions:
                    add_region(r)

        for name, group in self._builtin_aliases.items():
            aliases = group.get("aliases", [])
            if logical_name == name or max((self._text_score(t, a) for t in targets for a in aliases), default=0.0) >= 0.72:
                for a in aliases:
                    add_text(a)
                for r in group.get("preferred_regions", []):
                    add_region(r)

        for hint in ui_hints or []:
            add_region(hint)

        primary_target = str(targets[0]).strip() if targets else ""
        preferred_regions = merge_region_hints(
            raw_target=primary_target,
            target_role=target_role,
            explicit_hints=explicit_regions,
        )
        if not preferred_regions:
            preferred_regions = ["full"]
        return alias_pool, preferred_regions

    def _pick_click_point_from_box(
        self,
        text: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        preferred_regions: List[str],
        *,
        target_role: str,
    ) -> Tuple[float, float]:
        if target_role == "input":
            width = x2 - x1
            height = y2 - y1
            if width >= 0.25:
                cx = (x1 + x2) / 2.0
            else:
                cx = min(0.92, x2 + max(0.10, width * 1.8))
            cy = (y1 + y2) / 2.0
            return round(self._clamp01(cx), 4), round(self._clamp01(cy), 4)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        if width < 0.12:
            cy = min(1.0, y1 + height * 0.62)
        return round(self._clamp01(cx), 4), round(self._clamp01(cy), 4)

    def _is_box_reasonable(self, text: str, x1: float, y1: float, x2: float, y2: float, preferred_regions: List[str], target_role: str) -> bool:
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        if width <= 0 or height <= 0:
            return False
        if area < 0.00005:
            return False
        if target_role != "input" and len(text.strip()) <= 2 and width > 0.45:
            return False
        if y2 < 0.08 and re.fullmatch(r"[\d:%.A-Za-z]+", text.strip()):
            return False
        return True

    def _within_preferred_regions(self, x: float, y: float, preferred_regions: List[str]) -> bool:
        if not preferred_regions:
            return True
        for name in preferred_regions:
            r = self._region_defs.get(name)
            if not r:
                continue
            if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                return True
        return False

    def _position_score(self, x_pct: float, y_pct: float, preferred_regions: List[str]) -> float:
        if not preferred_regions:
            return 0.7
        best = 0.0
        for name in preferred_regions:
            region = self._region_defs.get(name)
            if not region:
                continue
            cx = (region[0] + region[2]) / 2.0
            cy = (region[1] + region[3]) / 2.0
            dx = x_pct - cx
            dy = y_pct - cy
            dist = (dx * dx + dy * dy) ** 0.5
            score = max(0.0, 1.0 - dist * 1.8)
            best = max(best, score)
        return round(best if best > 0 else 0.55, 4)

    def _edge_penalty(self, x1: float, y1: float, x2: float, y2: float) -> float:
        penalty = 0.0
        if x1 <= 0.001 or y1 <= 0.001 or x2 >= 0.999 or y2 >= 0.999:
            penalty += 0.12
        return penalty

    def _pick_best(self, candidates: List[ClickPoint]) -> Optional[ClickPoint]:
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[0]

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
        return (round(self._clamp01(x1), 4), round(self._clamp01(y1), 4), round(self._clamp01(x2), 4), round(self._clamp01(y2), 4))

    def _meta_float(self, meta: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
        for k in keys:
            if k in meta:
                try:
                    return float(meta[k])
                except Exception:
                    continue
        return None

    def _text_score(self, a: str, b: str) -> float:
        a = str(a or "").strip().lower()
        b = str(b or "").strip().lower()
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        if a in b or b in a:
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            return max(0.82, min(0.96, len(shorter) / max(1, len(longer))))
        return round(SequenceMatcher(None, a, b).ratio(), 4)

    def _dedupe_texts(self, values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for v in values:
            s = str(v or "").strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _clamp01(self, v: float) -> float:
        return max(0.0, min(1.0, float(v)))
