from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Iterable

from core.executor.action_schema import ActionPlan, ClickAction, InputAction, WaitAction
from core.runtime.observer import Observer
from core.runtime.project_loader import ProjectProfile, load_project_profile
from core.state.state_machine import StateMachine, StateDetectResult
from core.recovery.regions import merge_region_hints


@dataclass
class ObserveResult:
    state: StateDetectResult
    pack: object
    step_id: str


class AppSession:
    def __init__(
        self,
        adapter,
        step_runner,
        executor,
        perception=None,
        policy_runner=None,
        project: Optional[str] = None,
        project_profile: Optional[ProjectProfile] = None,
    ):
        self.adapter = adapter
        self.step_runner = step_runner
        self.executor = executor
        self.perception = perception
        self.policy_runner = policy_runner
        self.project_profile = project_profile or load_project_profile(project)
        self.observer = Observer(step_runner)
        self.state_machine = StateMachine(project_id=self.project_profile.project_id)

    @property
    def project_id(self) -> str:
        return self.project_profile.project_id

    def launch(self, app_package: Optional[str] = None, wait_seconds: float = 1.0) -> None:
        pkg = (app_package or self.project_profile.app_package or "").strip()
        if not pkg:
            raise RuntimeError(f"No app package configured for project={self.project_id}")
        self.step_runner.run(name=f"launch_{self.project_id}", action="activate_app", fn=lambda: self.adapter.activate_app(pkg))
        if wait_seconds > 0:
            self.wait(wait_seconds, name="wait_after_launch")

    def wait(self, seconds: float, name: str = "wait"):
        plan = ActionPlan.model_validate({"actions": [{"type": "WAIT", "name": name, "seconds": float(seconds)}]})
        return self.executor.run_plan(plan)

    def observe(self, name: str = "observe") -> ObserveResult:
        obs = self.observer.observe(name=name)
        state = self.state_machine.detect_state(obs.pack)
        return ObserveResult(state=state, pack=obs.pack, step_id=obs.step_id)

    def detect_state(self, name: str = "observe") -> StateDetectResult:
        return self.observe(name=name).state

    def resolve_state_name(self, state_name: str) -> str:
        key = str(state_name or "").strip()
        if not key:
            return key
        return self.project_profile.resolve_state(key)

    def _is_loading_state(self, state_res: StateDetectResult) -> bool:
        state_name = str(state_res.state or "")
        if state_name == "Busy.Loading":
            return True
        hints = list((state_res.meta or {}).get("overlay_hints", []) or [])
        loading_words = ("正在加载", "请稍后", "加载中", "处理中", "提交中")
        return any(any(w in str(h) for w in loading_words) for h in hints)

    def _should_treat_as_ready(self, state_res: StateDetectResult, expected: str) -> bool:
        if self._is_loading_state(state_res):
            return False
        return state_res.state == expected

    def expect_state(
        self,
        state_name: str,
        name: str = "observe_expect",
        *,
        stable_rounds: int = 2,
        max_rounds: int = 8,
        interval_s: float = 0.8,
    ) -> ObserveResult:
        expected = self.resolve_state_name(state_name)
        last_ok = 0
        last_res: Optional[ObserveResult] = None

        for idx in range(max_rounds):
            res = self.observe(name=name if idx == 0 else f"{name}_retry_{idx}")
            last_res = res

            if self._is_loading_state(res.state):
                last_ok = 0
                if idx < max_rounds - 1 and interval_s > 0:
                    self.wait(interval_s, name=f"wait_loading_{idx}")
                continue

            if self._should_treat_as_ready(res.state, expected):
                last_ok += 1
                if last_ok >= max(1, stable_rounds):
                    return res
            else:
                last_ok = 0

            if idx < max_rounds - 1 and interval_s > 0:
                self.wait(interval_s, name=f"wait_expect_retry_{idx}")

        if last_res is None:
            raise AssertionError(f"Expected state={expected}, but no observation was produced")
        raise AssertionError(
            f"Expected state={expected}, got={last_res.state.state}, score={last_res.state.score}, "
            f"topk={last_res.state.meta.get('topk')}"
        )

    def tap_text(self, text: str, logical_name: str = "", wait_after: float = 0.8, step_name: str = "tap_text"):
        text = str(text or "").strip()
        if not text:
            raise ValueError("tap_text requires non-empty text")
        actions = [ClickAction(name=step_name, selector=f"text={text}", target=text, logical_name=logical_name, allow_heal=True)]
        if wait_after > 0:
            actions.append(WaitAction(name="wait_after_tap", seconds=float(wait_after)))
        return self.executor.run_plan(ActionPlan(actions=actions))

    def tap_semantic(self, logical_name: str, fallback_text: str = "", wait_after: float = 0.8, step_name: str = "tap_semantic"):
        spec = dict(self.project_profile.semantic_targets.get(logical_name, {}) or {})
        aliases = self.project_profile.semantic_target_aliases(logical_name)
        primary = fallback_text or (aliases[0] if aliases else logical_name)
        region_hints = [str(x).strip() for x in (spec.get("region_hints") or []) if str(x).strip()]
        region_hint = str(spec.get("region_hint") or "").strip()
        if region_hint and region_hint not in region_hints:
            region_hints.insert(0, region_hint)
        region_hints = merge_region_hints(raw_target=primary, target_role="button", explicit_hints=region_hints)
        target_type = str(spec.get("target_type") or "auto").strip() or "auto"
        actions = [
            ClickAction(
                name=step_name,
                selector=f"text={primary}" if primary else logical_name,
                target=primary,
                logical_name=logical_name,
                allow_heal=True,
                target_type=target_type,
                text_candidates=aliases,
                region_hints=region_hints,
            )
        ]
        if wait_after > 0:
            actions.append(WaitAction(name="wait_after_tap", seconds=float(wait_after)))
        return self.executor.run_plan(ActionPlan(actions=actions))

    def input_semantic(self, logical_name: str, text: str, fallback_text: str = "", wait_after: float = 0.2, step_name: str = "input_semantic"):
        spec = dict(self.project_profile.semantic_targets.get(logical_name, {}) or {})
        aliases = self.project_profile.semantic_target_aliases(logical_name)
        primary = fallback_text or (aliases[0] if aliases else logical_name)
        region_hints = [str(x).strip() for x in (spec.get("region_hints") or []) if str(x).strip()]
        region_hint = str(spec.get("region_hint") or "").strip()
        if region_hint and region_hint not in region_hints:
            region_hints.insert(0, region_hint)
        region_hints = merge_region_hints(raw_target=primary, target_role="input", explicit_hints=region_hints)
        actions = [
            InputAction(
                name=step_name,
                selector=f"text={primary}" if primary else logical_name,
                target=primary,
                logical_name=logical_name,
                text=text,
                allow_heal=True,
                text_candidates=aliases,
                region_hints=region_hints,
            )
        ]
        if wait_after > 0:
            actions.append(WaitAction(name="wait_after_input", seconds=float(wait_after)))
        return self.executor.run_plan(ActionPlan(actions=actions))

    def run_plan(self, plan: ActionPlan):
        return self.executor.run_plan(plan)
