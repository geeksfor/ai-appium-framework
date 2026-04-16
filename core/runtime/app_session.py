from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.executor.action_schema import ActionPlan, ClickAction, InputAction, WaitAction
from core.recovery.regions import merge_region_hints
from core.runtime.observer import Observer
from core.runtime.project_loader import ProjectProfile, load_project_profile
from core.state.state_machine import StateMachine, StateDetectResult


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
        self.step_runner.run(
            name=f"launch_{self.project_id}",
            action="activate_app",
            fn=lambda: self.adapter.activate_app(pkg),
        )
        if wait_seconds > 0:
            self.wait(wait_seconds, name="wait_after_launch")

    def wait(self, seconds: float, name: str = "wait"):
        plan = ActionPlan.model_validate(
            {"actions": [{"type": "WAIT", "name": name, "seconds": float(seconds)}]}
        )
        return self.executor.run_plan(plan)

    def observe(self, name: str = "observe") -> ObserveResult:
        obs = self.observer.observe(name=name)
        state = self.state_machine.detect_state(obs.pack)
        return ObserveResult(state=state, pack=obs.pack, step_id=obs.step_id)

    def detect_state(self, name: str = "observe") -> StateDetectResult:
        return self.observe(name=name).state

    def resolve_state_name(self, state_name: str) -> str:
        return self.project_profile.resolve_state(state_name)

    def expect_state(
        self,
        state_name: str,
        name: str = "observe_expect",
        *,
        max_rounds: int = 6,
        stable_rounds: int = 2,
        wait_after_loading_s: float = 1.0,
        wait_between_rounds_s: float = 0.8,
        loading_state_name: str = "Busy.Loading",
    ) -> ObserveResult:
        expected = self.resolve_state_name(state_name)
        loading_state = self.resolve_state_name(loading_state_name)

        consecutive = 0
        last_state: Optional[str] = None
        last_res: Optional[ObserveResult] = None

        for i in range(max_rounds):
            obs_name = name if i == 0 else f"{name}_retry_{i}"
            res = self.observe(name=obs_name)
            last_res = res
            current = res.state.state

            if current == loading_state:
                consecutive = 0
                last_state = current
                if i < max_rounds - 1 and wait_after_loading_s > 0:
                    self.wait(wait_after_loading_s, name="wait_loading_overlay")
                continue

            if current == expected:
                if last_state == expected:
                    consecutive += 1
                else:
                    consecutive = 1

                if consecutive >= stable_rounds:
                    return res
            else:
                consecutive = 0

            last_state = current

            if i < max_rounds - 1 and wait_between_rounds_s > 0:
                self.wait(wait_between_rounds_s, name="wait_state_stable")

        if last_res is None:
            raise AssertionError(f"Expected state={expected}, but no observation result was produced")

        raise AssertionError(
            f"Expected state={expected}, got={last_res.state.state}, "
            f"score={last_res.state.score}, topk={last_res.state.meta.get('topk')}"
        )

    def tap_text(
        self,
        text: str,
        logical_name: str = "",
        wait_after: float = 0.8,
        step_name: str = "tap_text",
    ):
        text = str(text or "").strip()
        if not text:
            raise ValueError("tap_text requires non-empty text")
        actions = [
            ClickAction(
                name=step_name,
                selector=f"text={text}",
                target=text,
                logical_name=logical_name,
                allow_heal=True,
            )
        ]
        if wait_after > 0:
            actions.append(WaitAction(name="wait_after_tap", seconds=float(wait_after)))
        return self.executor.run_plan(ActionPlan(actions=actions))

    def tap_semantic(
        self,
        logical_name: str,
        fallback_text: str = "",
        wait_after: float = 0.8,
        step_name: str = "tap_semantic",
    ):
        spec = dict(self.project_profile.semantic_targets.get(logical_name, {}) or {})
        aliases = self.project_profile.semantic_target_aliases(logical_name)
        primary = fallback_text or (aliases[0] if aliases else logical_name)
        explicit_hints = [str(x).strip() for x in (spec.get("region_hints") or []) if str(x).strip()]
        region_hint = str(spec.get("region_hint") or "").strip()
        if region_hint:
            explicit_hints.insert(0, region_hint)
        target_type = str(spec.get("target_type") or "auto").strip() or "auto"
        target_role = str(
            spec.get("target_role") or ("input" if target_type == "input" else "button")
        ).strip() or "button"
        region_hints = merge_region_hints(
            raw_target=primary,
            target_role=target_role,
            explicit_hints=explicit_hints,
        )
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

    def input_semantic(
        self,
        logical_name: str,
        text: str,
        fallback_text: str = "",
        wait_after: float = 0.2,
        step_name: str = "input_semantic",
    ):
        spec = dict(self.project_profile.semantic_targets.get(logical_name, {}) or {})
        aliases = self.project_profile.semantic_target_aliases(logical_name)
        primary = fallback_text or (aliases[0] if aliases else logical_name)
        explicit_hints = [str(x).strip() for x in (spec.get("region_hints") or []) if str(x).strip()]
        region_hint = str(spec.get("region_hint") or "").strip()
        if region_hint:
            explicit_hints.insert(0, region_hint)
        region_hints = merge_region_hints(
            raw_target=primary,
            target_role="input",
            explicit_hints=explicit_hints,
        )
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
