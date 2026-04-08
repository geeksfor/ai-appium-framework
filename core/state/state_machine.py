from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from core.perception.perception import PerceptionPack
from core.state.registry import build_rules_for_project
from core.state.rules.base import Rule, MatchResult


@dataclass
class StateDetectResult:
    state: str
    score: float
    matches: List[MatchResult]
    best: Optional[MatchResult]
    meta: Dict[str, Any]


class StateMachine:
    """
    通用状态机。
    具体规则由项目适配层提供，而不是写死在 core/ 里。
    """

    def __init__(self, rules: Optional[List[Rule]] = None, project_id: Optional[str] = None):
        self.project_id = project_id
        self.rules: List[Rule] = rules or build_rules_for_project(project_id)

    def detect_state(self, pack: PerceptionPack) -> StateDetectResult:
        raw_text = (pack.ocr_text or "")
        text = " ".join(raw_text.split())
        t_low = text.lower()

        matches: List[MatchResult] = []
        candidates: List[MatchResult] = []

        best: Optional[MatchResult] = None
        best_rule: Optional[Rule] = None
        evaluated_states: List[str] = []

        def _is_popup_state(state: str) -> bool:
            return state.startswith("Popup.")

        def _brief(m: MatchResult) -> Dict[str, Any]:
            meta = m.meta or {}
            return {
                "state": m.state,
                "score": m.score,
                "hits": (m.hits or [])[:8],
                "misses": (m.misses or [])[:8],
                "reason": m.reason,
                "neg_hits": meta.get("neg_hits", []),
                "all_hits": meta.get("all_hits", []),
                "any_hits": meta.get("any_hits", []),
                "ok_candidate": meta.get("ok_candidate", False),
                "confirm": meta.get("confirm", None),
            }

        def _second_confirm(rule: Rule, mr: MatchResult) -> tuple[bool, str]:
            for s in getattr(rule, "confirm_not_contains", []) or []:
                if s and s.lower() in t_low:
                    return False, f"confirm_not_contains hit: {s}"

            confirm_any = getattr(rule, "confirm_any", []) or []
            confirm_min_any_hits = int(getattr(rule, "confirm_min_any_hits", 0) or 0)
            any_hits = (mr.meta or {}).get("any_hits", []) or []

            if not confirm_any and confirm_min_any_hits <= 0:
                return True, "no confirm constraints"

            if confirm_any:
                for s in confirm_any:
                    if s and s.lower() in t_low:
                        return True, f"confirm_any hit: {s}"

            if confirm_min_any_hits > 0:
                if len(any_hits) >= confirm_min_any_hits:
                    return True, f"confirm_min_any_hits ok: {len(any_hits)} >= {confirm_min_any_hits}"

            return False, "confirm conditions not met"

        overlay_hint_words = [
            "请登录", "登录", "去登录",
            "关闭", "跳过", "点击跳过", "我知道了", "知道了",
            "下一步", "上一步", "完成",
            "确定", "取消", "稍后", "以后再说",
            "同意", "拒绝", "隐私", "服务协议",
            "允许", "仅在使用期间", "始终允许", "权限",
        ]

        def _overlay_hints() -> List[str]:
            hits = []
            for w in overlay_hint_words:
                if w and w.lower() in t_low:
                    hits.append(w)
            seen = set()
            out = []
            for h in hits:
                if h not in seen:
                    seen.add(h)
                    out.append(h)
            return out

        for r in self.rules:
            mr = r.score(text)
            matches.append(mr)
            evaluated_states.append(mr.state)

            if best is None or mr.score > best.score:
                best = mr
                best_rule = r

            ok_candidate = bool((mr.meta or {}).get("ok_candidate", False))
            if not ok_candidate:
                continue

            candidates.append(mr)

            if _is_popup_state(mr.state):
                topk = sorted(matches, key=lambda m: m.score, reverse=True)[:5]
                meta = {
                    "perception_meta": pack.meta,
                    "normalized": True,
                    "text_len": len(text),
                    "short_circuit": True,
                    "reason": "popup short-circuit",
                    "evaluated_rules": evaluated_states,
                    "candidates": [_brief(c) for c in sorted(candidates, key=lambda m: m.score, reverse=True)],
                    "topk": [_brief(t) for t in topk],
                    "best_rule": r.state_name,
                    "overlay_suspected": False,
                    "overlay_hints": [],
                    "project_id": self.project_id,
                }
                return StateDetectResult(mr.state, mr.score, matches, mr, meta)

            ok2, why2 = _second_confirm(r, mr)
            if ok2:
                topk = sorted(matches, key=lambda m: m.score, reverse=True)[:5]
                meta = {
                    "perception_meta": pack.meta,
                    "normalized": True,
                    "text_len": len(text),
                    "short_circuit": True,
                    "reason": f"page confirmed: {why2}",
                    "evaluated_rules": evaluated_states,
                    "candidates": [_brief(c) for c in sorted(candidates, key=lambda m: m.score, reverse=True)],
                    "topk": [_brief(t) for t in topk],
                    "best_rule": r.state_name,
                    "confirm": {"ok": True, "detail": why2},
                    "overlay_suspected": False,
                    "overlay_hints": [],
                    "project_id": self.project_id,
                }
                return StateDetectResult(mr.state, mr.score, matches, mr, meta)

            mr.meta = mr.meta or {}
            mr.meta["confirm"] = {"ok": False, "detail": why2}

        topk = sorted(matches, key=lambda m: m.score, reverse=True)[:5]
        hints = _overlay_hints()
        meta = {
            "perception_meta": pack.meta,
            "normalized": True,
            "text_len": len(text),
            "short_circuit": False,
            "evaluated_rules": evaluated_states,
            "candidates": [_brief(c) for c in sorted(candidates, key=lambda m: m.score, reverse=True)],
            "topk": [_brief(t) for t in topk],
            "best_rule": best_rule.state_name if best_rule else None,
            "overlay_suspected": bool(hints),
            "overlay_hints": hints,
            "project_id": self.project_id,
        }

        return StateDetectResult("Unknown", 0.0, matches, best, meta)
