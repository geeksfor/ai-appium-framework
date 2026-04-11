from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from core.recovery.click_resolver import ClickPoint, ClickResolver


class HealAIProvider(Protocol):
    policy_runner: Any


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
    target_role: str = "button"


@dataclass
class HealResult:
    raw_target: str
    healed: bool
    confidence: float
    chosen: Optional[HealCandidate]
    candidates: List[HealCandidate]


class PolicyRunnerHealAIProvider:
    def __init__(self, policy_runner: Any):
        self.policy_runner = policy_runner


class HealPolicy:
    def __init__(
        self,
        locator_store_path: str = "core/heal/locator_store.yaml",
        ai_provider: Optional[HealAIProvider] = None,
        accept_threshold: float = 0.72,
        candidate_threshold: float = 0.45,
    ):
        self.accept_threshold = accept_threshold
        self.resolver = ClickResolver(
            policy_runner=getattr(ai_provider, "policy_runner", None),
            locator_store_path=locator_store_path,
            candidate_threshold=candidate_threshold,
        )

    def heal_click(
        self,
        pack: Any,
        selector: str,
        logical_name: Optional[str] = None,
        save_path: Optional[str | Path] = None,
        target_type: str = "auto",
        text_candidates: Optional[List[str]] = None,
        region_hints: Optional[List[str]] = None,
    ) -> HealResult:
        raw_target = self._extract_target_text(selector)
        cp = self.resolver.resolve_semantic(
            pack=pack,
            primary_target=raw_target,
            logical_name=str(logical_name or "").strip(),
            target_type=target_type,
            text_candidates=text_candidates or [],
            region_hints=region_hints or [],
            target_role="button",
        )
        result = self._build_result(raw_target, cp, self.resolver.last_candidates, target_role="button")
        if save_path:
            self.save_candidates(result, save_path)
        return result

    def heal_input(
        self,
        pack: Any,
        selector: str,
        logical_name: Optional[str] = None,
        save_path: Optional[str | Path] = None,
        text_candidates: Optional[List[str]] = None,
        region_hints: Optional[List[str]] = None,
    ) -> HealResult:
        raw_target = self._extract_target_text(selector)
        cp = self.resolver.resolve_semantic(
            pack=pack,
            primary_target=raw_target,
            logical_name=str(logical_name or "").strip(),
            target_type="input",
            text_candidates=text_candidates or [],
            region_hints=region_hints or [],
            target_role="input",
        )
        result = self._build_result(raw_target, cp, self.resolver.last_candidates, target_role="input")
        if save_path:
            self.save_candidates(result, save_path)
        return result

    def save_candidates(self, result: HealResult, save_path: str | Path) -> None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "raw_target": result.raw_target,
            "healed": result.healed,
            "confidence": result.confidence,
            "chosen": asdict(result.chosen) if result.chosen else None,
            "candidates": [asdict(c) for c in result.candidates],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_result(self, raw_target: str, chosen_point: Optional[ClickPoint], all_points: List[ClickPoint], *, target_role: str) -> HealResult:
        candidates = [self._to_candidate(raw_target, p, target_role=target_role) for p in all_points]
        chosen = self._to_candidate(raw_target, chosen_point, target_role=target_role) if chosen_point and chosen_point.score >= self.accept_threshold else None
        return HealResult(raw_target=raw_target, healed=chosen is not None, confidence=round(chosen.score, 4) if chosen else 0.0, chosen=chosen, candidates=candidates)

    def _to_candidate(self, raw_target: str, point: Optional[ClickPoint], *, target_role: str) -> Optional[HealCandidate]:
        if point is None:
            return None
        return HealCandidate(
            text=point.text,
            x_pct=point.x_pct,
            y_pct=point.y_pct,
            score=point.score,
            source=point.source,
            reason=point.reason,
            raw_target=raw_target,
            bbox=point.bbox,
            target_role=target_role,
        )

    def _extract_target_text(self, selector: str) -> str:
        raw = str(selector or "").strip()
        if not raw:
            return ""
        for prefix in ["text=", "id=", "xpath=", "accessibility_id="]:
            if raw.startswith(prefix):
                return raw[len(prefix):].strip()
        return raw
