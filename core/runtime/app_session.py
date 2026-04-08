from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.executor.action_schema import ActionPlan, ClickAction, WaitAction
from core.runtime.observer import Observer
from core.runtime.project_loader import ProjectProfile, load_project_profile
from core.state.state_machine import StateMachine, StateDetectResult


@dataclass
class ObserveResult:
    state: StateDetectResult
    pack: object
    step_id: str


class AppSession:
    """
    面向测试用例的统一门面。
    测试代码不再直接拼 StepRunner / Executor / Perception / StateMachine。
    """

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

    def launch(self, app_package: Optional[str] = None, wait_seconds: float = 2.0) -> None:
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
        plan = ActionPlan.model_validate({
            "actions": [{"type": "WAIT", "name": name, "seconds": float(seconds)}]
        })
        return self.executor.run_plan(plan)

    def observe(self, name: str = "observe") -> ObserveResult:
        obs = self.observer.observe(name=name)
        state = self.state_machine.detect_state(obs.pack)
        return ObserveResult(state=state, pack=obs.pack, step_id=obs.step_id)

    def detect_state(self, name: str = "observe") -> StateDetectResult:
        return self.observe(name=name).state

    def resolve_state_name(self, state_name: str) -> str:
        return self.project_profile.resolve_state(state_name)

    def expect_state(self, state_name: str, name: str = "observe_expect") -> ObserveResult:
        res = self.observe(name=name)
        expected = self.resolve_state_name(state_name)
        if res.state.state != expected:
            raise AssertionError(
                f"Expected state={expected}, got={res.state.state}, score={res.state.score}, topk={res.state.meta.get('topk')}"
            )
        return res

    def tap_text(self, text: str, logical_name: str = "", wait_after: float = 0.8, step_name: str = "tap_text"):
        text = str(text or "").strip()
        if not text:
            raise ValueError("tap_text requires non-empty text")
        actions = [
            ClickAction(name=step_name, selector=f"text={text}", target=text, logical_name=logical_name, allow_heal=True),
        ]
        if wait_after > 0:
            actions.append(WaitAction(name="wait_after_tap", seconds=float(wait_after)))
        plan = ActionPlan(actions=actions)
        return self.executor.run_plan(plan)

    def tap_semantic(self, logical_name: str, fallback_text: str = "", wait_after: float = 0.8, step_name: str = "tap_semantic"):
        aliases = self.project_profile.semantic_target_aliases(logical_name)
        primary = fallback_text or (aliases[0] if aliases else logical_name)
        actions = [
            ClickAction(
                name=step_name,
                selector=f"text={primary}" if primary else logical_name,
                target=primary,
                logical_name=logical_name,
                allow_heal=True,
            )
        ]
        if wait_after > 0:
            actions.append(WaitAction(name="wait_after_tap", seconds=float(wait_after)))
        plan = ActionPlan(actions=actions)
        return self.executor.run_plan(plan)

    def run_plan(self, plan: ActionPlan):
        return self.executor.run_plan(plan)
