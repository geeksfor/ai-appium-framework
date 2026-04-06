# core/executor/runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from core.executor.action_schema import ActionPlan
from core.executor.executor import Executor
from core.perception.perception import PerceptionPack
from core.policy.parser import ParsedPolicy
from core.policy.policy_runner import PolicyRunner
from core.state.state_machine import StateDetectResult, StateMachine


@dataclass
class RunnerConfig:
    max_rounds: int = 10
    ai_max_calls: int = 3
    ai_only_on_trigger: bool = True
    ai_on_unknown_without_overlay: bool = True
    ai_unknown_round_limit: int = 2
    stop_on_ask_human: bool = True
    stop_on_ai_stop: bool = True
    verbose: bool = True


@dataclass
class RunnerResult:
    ok: bool
    rounds: int
    ai_calls: int
    last_state: str
    last_reason: str
    run_id: str


class Runner:
    """
    收敛版闭环：observe -> detect -> rule/ai -> execute -> observe...
    今天这版额外修了两件事：
    1) 当前回合的 no_progress 会在 step() 内即时刷新
    2) 真实 App 场景下，Unknown 页面可在前几轮直接触发一次 AI，不再强依赖 overlay
    """

    def __init__(
        self,
        evidence_manager,
        step_runner,
        state_machine: StateMachine,
        policy_runner: PolicyRunner,
        executor: Executor,
        cfg: Optional[RunnerConfig] = None,
        rule_handler: Optional[Callable[[StateDetectResult, PerceptionPack], Optional[ActionPlan]]] = None,
        success_check: Optional[Callable[[StateDetectResult, PerceptionPack], Tuple[bool, str]]] = None,
    ):
        self.evidence = evidence_manager
        self.step_runner = step_runner
        self.sm = state_machine
        self.pr = policy_runner
        self.executor = executor
        self.cfg = cfg or RunnerConfig()
        self.rule_handler = rule_handler
        self.success_check = success_check

        self._ai_calls = 0
        self._rounds = 0
        self._recent: Dict[str, Any] = {}
        self._hints: Dict[str, Any] = {}

    def _load_latest_perception(self) -> PerceptionPack:
        run_dir = Path(self.evidence.run.run_dir)
        step_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        if not step_dirs:
            return PerceptionPack(image_path="", ocr_text="", meta={"available": False, "reason": "no steps yet"})

        last = step_dirs[-1]
        ocr_path = last / "ocr.txt"
        meta_path = last / "perception.json"

        ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"available": False}

        img = ""
        for cand in ["screenshot_after.png", "screenshot.png", "screenshot_fail.png"]:
            p = last / cand
            if p.exists():
                img = str(p)
                break

        return PerceptionPack(image_path=img, ocr_text=ocr_text, meta=meta)

    def _should_call_ai(self, state_res: StateDetectResult) -> bool:
        if self._ai_calls >= self.cfg.ai_max_calls:
            return False

        overlay_suspected = bool((state_res.meta or {}).get("overlay_suspected", False))
        no_progress = bool((self._hints or {}).get("no_progress", False))
        unknown_state = state_res.state == "Unknown"

        if not self.cfg.ai_only_on_trigger:
            return unknown_state or overlay_suspected or no_progress

        if no_progress:
            return True

        if unknown_state and overlay_suspected:
            return True

        if unknown_state and self.cfg.ai_on_unknown_without_overlay:
            return self._rounds <= self.cfg.ai_unknown_round_limit

        return False

    def step(self, goal: Dict[str, Any]) -> Tuple[bool, str, StateDetectResult]:
        pack = self._load_latest_perception()
        self._refresh_hints_from_latest_step()
        state_res = self.sm.detect_state(pack)

        if self.cfg.verbose:
            print(
                f"[runner] state={state_res.state} score={state_res.score} "
                f"overlay={state_res.meta.get('overlay_suspected')} no_progress={self._hints.get('no_progress')}"
            )

        if self.success_check:
            ok, reason = self.success_check(state_res, pack)
            if ok:
                return True, f"success: {reason}", state_res

        if self.rule_handler:
            plan = self.rule_handler(state_res, pack)
            if plan is not None:
                self.executor.run_plan(plan)
                self._recent = {"source": "rule", "state": state_res.state}
                return False, "executed rule plan", state_res

        if not self._should_call_ai(state_res):
            return False, "AI not triggered or budget exceeded", state_res

        self._ai_calls += 1
        parsed: ParsedPolicy = self.pr.decide_next(
            pack=pack,
            goal=goal,
            recent=self._recent,
            hints={
                **(self._hints or {}),
                "state": state_res.state,
                "overlay_suspected": bool((state_res.meta or {}).get("overlay_suspected", False)),
                "overlay_hints": (state_res.meta or {}).get("overlay_hints", []),
            },
        )

        if parsed.kind == "plan" and parsed.plan is not None:
            self.executor.run_plan(parsed.plan)
            self._recent = {"source": "ai", "kind": "plan", "ai_calls": self._ai_calls}
            return False, "executed ai plan", state_res

        if parsed.kind == "ask_human":
            if self.cfg.stop_on_ask_human:
                return False, f"stop: ask_human: {parsed.reason}", state_res
            return False, f"ask_human: {parsed.reason}", state_res

        if self.cfg.stop_on_ai_stop:
            return False, f"stop: {parsed.reason}", state_res
        return False, f"ai stop: {parsed.reason}", state_res

    def run(self, goal: Dict[str, Any]) -> RunnerResult:
        run_id = self.evidence.run.run_id
        last_state = "Unknown"
        last_reason = "not started"

        for i in range(1, self.cfg.max_rounds + 1):
            self._rounds = i

            def _observe():
                return None

            self.step_runner.run(name=f"observe_{i}", action="OBSERVE", fn=_observe)

            ok, reason, state_res = self.step(goal)
            last_state = state_res.state
            last_reason = reason

            if ok:
                return RunnerResult(
                    ok=True,
                    rounds=i,
                    ai_calls=self._ai_calls,
                    last_state=last_state,
                    last_reason=last_reason,
                    run_id=run_id,
                )

            if reason.startswith("stop:"):
                break

        return RunnerResult(
            ok=False,
            rounds=self._rounds,
            ai_calls=self._ai_calls,
            last_state=last_state,
            last_reason=last_reason,
            run_id=run_id,
        )

    def _refresh_hints_from_latest_step(self) -> None:
        run_dir = Path(self.evidence.run.run_dir)
        step_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        if not step_dirs:
            return

        last = step_dirs[-1]
        meta_path = last / "meta.json"
        if not meta_path.exists():
            return

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            extra = meta.get("extra") or {}
            if "no_progress" in extra:
                self._hints["no_progress"] = bool(extra.get("no_progress"))
        except Exception:
            pass
