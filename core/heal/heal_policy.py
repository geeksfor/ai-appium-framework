from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import yaml

from core.perception.ocr import QwenVisionOCRProvider, _extract_json_text
from core.policy.goal_schema import GoalIntent


class HealAIProvider(Protocol):
    def suggest_click(
        self,
        pack: Any,
        raw_target: str,
        candidate_texts: List[str],
        locator_aliases: List[str],
        preferred_regions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]: ...


@dataclass
class LocatorEntry:
    key: str
    aliases: List[str]
    preferred_regions: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class HealCandidate:
    text: str
    x_pct: float
    y_pct: float
    score: float
    source: str
    reason: str
    raw_target: str
    bbox: Optional[Dict[str, float]] = None


@dataclass
class HealResult:
    raw_target: str
    healed: bool
    confidence: float
    chosen: Optional[HealCandidate]
    candidates: List[HealCandidate]


class LocatorStore:
    def __init__(self, entries: List[LocatorEntry]):
        self.entries = entries

    @classmethod
    def load(cls, path: str = "core/heal/locator_store.yaml") -> "LocatorStore":
        p = Path(path)
        if not p.exists():
            return cls(entries=[])
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw_entries = data.get("entries", {})
        entries: List[LocatorEntry] = []
        if isinstance(raw_entries, dict):
            for key, value in raw_entries.items():
                if not isinstance(value, dict):
                    continue
                aliases = [str(x).strip() for x in value.get("aliases", []) if str(x).strip()]
                preferred_regions = [str(x).strip() for x in value.get("preferred_regions", []) if str(x).strip()]
                notes = str(value.get("notes", "")).strip()
                entries.append(LocatorEntry(key=str(key).strip(), aliases=aliases, preferred_regions=preferred_regions, notes=notes))
        return cls(entries=entries)

    def find_related(self, raw_target: str, logical_name: Optional[str] = None) -> List[LocatorEntry]:
        target = raw_target.strip()
        results: List[LocatorEntry] = []
        if logical_name:
            for item in self.entries:
                if item.key == logical_name:
                    results.append(item)
        for item in self.entries:
            if item in results:
                continue
            key_score = _sim(target, item.key)
            alias_score = max((_sim(target, x) for x in item.aliases), default=0.0)
            if key_score >= 0.60 or alias_score >= 0.60:
                results.append(item)
        return results


class PolicyRunnerHealAIProvider:
    def __init__(self, policy_runner: Any):
        self.policy_runner = policy_runner
        self.vision = QwenVisionOCRProvider(
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.getenv("DASHSCOPE_CLICK_MODEL", os.getenv("DASHSCOPE_BBOX_MODEL", "qwen-vl-max")),
        )

    def suggest_click(
        self,
        pack: Any,
        raw_target: str,
        candidate_texts: List[str],
        locator_aliases: List[str],
        preferred_regions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        preferred_regions = preferred_regions or []
        visual = self._suggest_click_by_vision(pack, raw_target, candidate_texts, locator_aliases, preferred_regions)
        if visual is not None:
            return visual

        goal = self._build_goal(raw_target, candidate_texts, locator_aliases, preferred_regions)
        decide_next = getattr(self.policy_runner, "decide_next", None)
        if callable(decide_next):
            try:
                parsed = decide_next(pack=pack, goal=goal, recent={}, hints={})
                parsed_res = self._parse_result(parsed.plan if getattr(parsed, "plan", None) is not None else parsed, raw_target=raw_target, candidate_texts=candidate_texts)
                if parsed_res:
                    return parsed_res
            except Exception:
                pass

        for method_name, kwargs in [
            ("suggest_click", {"pack": pack, "raw_target": raw_target, "candidate_texts": candidate_texts, "locator_aliases": locator_aliases}),
            ("heal_click", {"pack": pack, "raw_target": raw_target, "candidate_texts": candidate_texts, "locator_aliases": locator_aliases}),
            ("plan_for_heal", {"pack": pack, "goal": goal}),
            ("run", {"pack": pack, "goal": goal}),
            ("plan", {"pack": pack, "goal": goal}),
        ]:
            method = getattr(self.policy_runner, method_name, None)
            if callable(method):
                try:
                    result = method(**kwargs)
                    if result is not None:
                        parsed = self._parse_result(result, raw_target=raw_target, candidate_texts=candidate_texts)
                        if parsed:
                            return parsed
                except TypeError:
                    continue
                except Exception:
                    return None
        return None

    def _build_goal(self, raw_target: str, candidate_texts: List[str], locator_aliases: List[str], preferred_regions: List[str]) -> Dict[str, Any]:
        hints = []
        if preferred_regions:
            hints.append(f"prefer_regions={','.join(preferred_regions)}")
        if candidate_texts:
            hints.append(f"ocr_candidates={','.join(candidate_texts[:8])}")
        return {
            "intent": GoalIntent.CLICK_TEXT.value,
            "target_text": raw_target,
            "synonyms": locator_aliases[:8],
            "ui_hints": hints,
            "constraints": {
                "avoid_login": False,
                "avoid_permission": False,
                "prefer_close_over_enter": False,
                "max_steps": 1,
                "prefer_region": preferred_regions[0] if preferred_regions else None,
            },
        }

    def _suggest_click_by_vision(self, pack: Any, raw_target: str, candidate_texts: List[str], locator_aliases: List[str], preferred_regions: List[str]) -> Optional[Dict[str, Any]]:
        image_path = str(getattr(pack, "image_path", "") or "").strip()
        if not image_path or not os.getenv("DASHSCOPE_API_KEY"):
            return None
        aliases = [x for x in locator_aliases if x]
        if raw_target and raw_target not in aliases:
            aliases.insert(0, raw_target)
        alias_str = "、".join(aliases[:10]) or raw_target
        ocr_str = "、".join(candidate_texts[:12]) if candidate_texts else "无"
        region_str = ", ".join(preferred_regions) if preferred_regions else "无"
        prompt = (
            "你是一个移动端 UI 点击助手。请基于截图，找到最应该点击的目标，并且只返回 JSON。\n"
            f"目标文案：{raw_target}\n"
            f"同义词/别名：{alias_str}\n"
            f"现有 OCR 候选：{ocr_str}\n"
            f"优先区域：{region_str}\n"
            "要求：\n"
            "1) 返回一个点击点，而不是 OCR 框；\n"
            "2) 如果画面中存在目标按钮，请返回其中心点的相对坐标 x_pct/y_pct（0~1）；\n"
            "3) 如果无法确定，confidence 降低；\n"
            "4) 只返回 JSON，不要解释。\n"
            'JSON 格式：{"text":"登录","x_pct":0.5,"y_pct":0.85,"confidence":0.88,"reason":"..."}'
        )
        try:
            r = self.vision.recognize_with_prompt(image_path=image_path, prompt=prompt)
            if r.error or not (r.text or "").strip():
                return None
            obj = json.loads(_extract_json_text(r.text))
            if not isinstance(obj, dict):
                return None
            parsed = self._parse_dict(obj)
            if parsed:
                parsed["reason"] = (parsed.get("reason") or "AI建议的替代点击") + " | 视觉AI"
            return parsed
        except Exception:
            return None

    def _parse_result(self, result: Any, raw_target: str, candidate_texts: List[str]) -> Optional[Dict[str, Any]]:
        if isinstance(result, dict):
            parsed = self._parse_dict(result)
            if parsed:
                return parsed
        actions = getattr(result, "actions", None)
        if isinstance(actions, list):
            for act in actions:
                parsed = self._parse_action_like(act)
                if parsed:
                    return parsed
        parsed = self._parse_action_like(result)
        if parsed:
            return parsed
        if isinstance(result, str) and result.strip():
            text = result.strip()
            confidence = 0.72 if text in candidate_texts else 0.66
            return {"text": text, "x_pct": 0.5, "y_pct": 0.5, "confidence": confidence, "reason": "PolicyRunner 返回文本建议，坐标未提供"}
        return None

    def _parse_dict(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = str(data.get("text") or data.get("target") or data.get("target_text") or "").strip()
        x_pct = data.get("x_pct")
        y_pct = data.get("y_pct")
        confidence = data.get("confidence", data.get("score", 0.0))
        reason = str(data.get("reason", "AI建议的替代点击")).strip()
        if text or (x_pct is not None and y_pct is not None):
            try:
                return {"text": text, "x_pct": _clamp01(float(x_pct) if x_pct is not None else 0.5), "y_pct": _clamp01(float(y_pct) if y_pct is not None else 0.5), "confidence": float(confidence), "reason": reason}
            except Exception:
                return None
        return None

    def _parse_action_like(self, obj: Any) -> Optional[Dict[str, Any]]:
        action_type = str(getattr(obj, "type", None) or getattr(obj, "action_type", None) or "").upper()
        if action_type and action_type != "CLICK":
            return None
        text = str(getattr(obj, "target", None) or getattr(obj, "text", None) or getattr(obj, "target_text", None) or "").strip()
        x_pct = getattr(obj, "x_pct", None)
        y_pct = getattr(obj, "y_pct", None)
        confidence = getattr(obj, "confidence", getattr(obj, "score", 0.0))
        reason = str(getattr(obj, "reason", "AI建议的替代点击")).strip()
        if text or (x_pct is not None and y_pct is not None):
            try:
                return {"text": text, "x_pct": _clamp01(float(x_pct) if x_pct is not None else 0.5), "y_pct": _clamp01(float(y_pct) if y_pct is not None else 0.5), "confidence": float(confidence), "reason": reason}
            except Exception:
                return None
        return None


class HealPolicy:
    def __init__(self, locator_store_path: str = "core/heal/locator_store.yaml", ai_provider: Optional[HealAIProvider] = None, accept_threshold: float = 0.72, candidate_threshold: float = 0.45):
        self.locator_store = LocatorStore.load(locator_store_path)
        self.ai_provider = ai_provider
        self.accept_threshold = accept_threshold
        self.candidate_threshold = candidate_threshold

    def heal_click(self, pack: Any, selector: str, logical_name: Optional[str] = None, save_path: Optional[str | Path] = None) -> HealResult:
        raw_target = self._extract_target_text(selector)
        related_entries = self.locator_store.find_related(raw_target, logical_name=logical_name)
        alias_pool = self._build_alias_pool(raw_target, related_entries)
        preferred_regions = self._collect_preferred_regions(related_entries)
        candidates = self._collect_ocr_candidates(pack=pack, raw_target=raw_target, alias_pool=alias_pool, preferred_regions=preferred_regions)
        ai_candidate = self._collect_ai_candidate(pack=pack, raw_target=raw_target, alias_pool=alias_pool, existing_texts=[c.text for c in candidates], preferred_regions=preferred_regions)
        if ai_candidate is not None:
            candidates.append(ai_candidate)
        candidates = self._dedupe_and_sort(candidates)
        chosen = candidates[0] if candidates and candidates[0].score >= self.accept_threshold else None
        result = HealResult(raw_target=raw_target, healed=chosen is not None, confidence=round(chosen.score, 4) if chosen else 0.0, chosen=chosen, candidates=candidates)
        if save_path:
            self.save_candidates(result, save_path)
        return result

    def save_candidates(self, result: HealResult, save_path: str | Path) -> None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"raw_target": result.raw_target, "healed": result.healed, "confidence": result.confidence, "chosen": asdict(result.chosen) if result.chosen else None, "candidates": [asdict(c) for c in result.candidates]}
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _collect_ocr_candidates(self, pack: Any, raw_target: str, alias_pool: List[str], preferred_regions: List[str]) -> List[HealCandidate]:
        out: List[HealCandidate] = []
        meta = getattr(pack, "meta", {}) or {}
        for box in self._read_ocr_boxes(pack):
            text = str(box.get("text", "")).strip()
            if not text:
                continue
            text_score = max((_sim(text, x) for x in alias_pool), default=0.0)
            if text_score < self.candidate_threshold:
                continue
            x1, y1, x2, y2 = self._normalize_box(box, meta)
            if not self._is_box_reasonable(text, x1, y1, x2, y2, preferred_regions):
                continue
            x_pct = round((x1 + x2) / 2.0, 4)
            y_pct = round((y1 + y2) / 2.0, 4)
            pos_score = self._position_score(x_pct, y_pct, preferred_regions)
            score = round(min(1.0, text_score * 0.68 + pos_score * 0.32), 4)
            matched_alias = max(alias_pool, key=lambda x: _sim(text, x))
            reason = f'ocr命中，文本最接近“{matched_alias}”，位置分={pos_score:.2f}'
            out.append(HealCandidate(text=text, x_pct=x_pct, y_pct=y_pct, score=score, source="ocr+store" if text in alias_pool or matched_alias != raw_target else "ocr", reason=reason, raw_target=raw_target, bbox={"x1": x1, "y1": y1, "x2": x2, "y2": y2}))
        return out

    def _collect_ai_candidate(self, pack: Any, raw_target: str, alias_pool: List[str], existing_texts: List[str], preferred_regions: List[str]) -> Optional[HealCandidate]:
        if self.ai_provider is None:
            return None
        res = self.ai_provider.suggest_click(pack=pack, raw_target=raw_target, candidate_texts=existing_texts, locator_aliases=alias_pool, preferred_regions=preferred_regions)
        if not res:
            return None
        text = str(res.get("text", "")).strip() or raw_target
        x_pct = float(res.get("x_pct", 0.5))
        y_pct = float(res.get("y_pct", 0.5))
        score = round(max(0.0, min(1.0, float(res.get("confidence", 0.0)))), 4)
        reason = str(res.get("reason", "")).strip() or "AI建议的替代点击"
        return HealCandidate(text=text, x_pct=round(_clamp01(x_pct), 4), y_pct=round(_clamp01(y_pct), 4), score=score, source="ai", reason=reason, raw_target=raw_target, bbox=None)

    def _build_alias_pool(self, raw_target: str, entries: List[LocatorEntry]) -> List[str]:
        pool = [raw_target] if raw_target else []
        for item in entries:
            if item.key and item.key not in pool:
                pool.append(item.key)
            for alias in item.aliases:
                if alias not in pool:
                    pool.append(alias)
        return pool

    def _collect_preferred_regions(self, entries: List[LocatorEntry]) -> List[str]:
        regions: List[str] = []
        for item in entries:
            for region in item.preferred_regions:
                if region not in regions:
                    regions.append(region)
        return regions

    def _read_ocr_boxes(self, pack: Any) -> List[Dict[str, Any]]:
        meta = getattr(pack, "meta", {}) or {}
        boxes = meta.get("ocr_boxes", [])
        if not isinstance(boxes, list):
            return []
        return [x for x in boxes if isinstance(x, dict)]

    def _normalize_box(self, box: Dict[str, Any], meta: Dict[str, Any]) -> tuple[float, float, float, float]:
        if "bbox" in box and isinstance(box["bbox"], list) and len(box["bbox"]) == 4:
            x1, y1, x2, y2 = [float(v) for v in box["bbox"]]
        else:
            x1 = float(box.get("x1", 0.0))
            y1 = float(box.get("y1", 0.0))
            x2 = float(box.get("x2", 0.0))
            y2 = float(box.get("y2", 0.0))
        maxv = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if maxv > 1.5:
            sw = float(meta.get("screen_width", 1.0)) or 1.0
            sh = float(meta.get("screen_height", 1.0)) or 1.0
            x1, x2 = x1 / sw, x2 / sw
            y1, y2 = y1 / sh, y2 / sh
        x1 = round(_clamp01(x1), 4)
        y1 = round(_clamp01(y1), 4)
        x2 = round(_clamp01(x2), 4)
        y2 = round(_clamp01(y2), 4)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    def _is_box_reasonable(self, text: str, x1: float, y1: float, x2: float, y2: float, preferred_regions: List[str]) -> bool:
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        if w <= 0.0 or h <= 0.0:
            return False
        if w >= 0.96 or h >= 0.96:
            return False
        if area >= 0.40 and len(text) <= 8:
            return False
        if h >= 0.42 and len(text) <= 8:
            return False
        if y1 <= 0.02 and y2 >= 0.98 and len(text) <= 8:
            return False
        x_pct = (x1 + x2) / 2.0
        y_pct = (y1 + y2) / 2.0
        pos_score = self._position_score(x_pct, y_pct, preferred_regions)
        if preferred_regions and pos_score < 0.56 and area > 0.12:
            return False
        return True

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

    def _dedupe_and_sort(self, candidates: List[HealCandidate]) -> List[HealCandidate]:
        best_map: Dict[str, HealCandidate] = {}
        for item in candidates:
            key = f"{item.text}|{item.x_pct:.4f}|{item.y_pct:.4f}"
            old = best_map.get(key)
            if old is None or item.score > old.score:
                best_map[key] = item
        return sorted(best_map.values(), key=lambda x: x.score, reverse=True)

    def _extract_target_text(self, selector: str) -> str:
        s = str(selector or "").strip()
        if not s:
            return ""
        for prefix in ["text=", "id=", "accessibility_id=", "name="]:
            if s.startswith(prefix):
                return s[len(prefix):].strip().strip('"').strip("'")
        if "@text=" in s:
            part = s.split("@text=", 1)[1].strip()
            if part.startswith(("'", '"')):
                quote = part[0]
                end = part.find(quote, 1)
                if end > 0:
                    return part[1:end].strip()
        if "contains(@text," in s:
            part = s.split("contains(@text,", 1)[1].strip()
            if part.startswith(("'", '"')):
                quote = part[0]
                end = part.find(quote, 1)
                if end > 0:
                    return part[1:end].strip()
        return s


def _sim(a: str, b: str) -> float:
    a2 = str(a).strip().lower()
    b2 = str(b).strip().lower()
    if not a2 or not b2:
        return 0.0
    if a2 == b2:
        return 1.0
    if a2 in b2 or b2 in a2:
        return 0.92
    return round(SequenceMatcher(None, a2, b2).ratio(), 4)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))
