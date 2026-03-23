# core/policy/decider.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.perception.perception import PerceptionPack
from core.state.state_machine import StateMachine, StateDetectResult
from core.policy.policy_runner import PolicyRunner
from core.policy.parser import ParsedPolicy
from core.executor.action_schema import ActionPlan


@dataclass
class DecideContext:
    goal: str
    recent: Optional[Dict[str, Any]] = None
    hints: Optional[Dict[str, Any]] = None


class Decider:
    """
    Day6 最小闭环 Decider：
    - 先用 StateMachine 做便宜识别
    - 满足触发条件才调用 Qwen（PolicyRunner）
    - 返回 ParsedPolicy（plan/ask/stop）
    """

    def __init__(self, state_machine: StateMachine, policy_runner: PolicyRunner):
        self.sm = state_machine
        self.pr = policy_runner

    def should_call_ai(self, state_res: StateDetectResult, hints: Dict[str, Any]) -> bool:
        # 触发条件：Unknown 且 overlay_suspected / 或 no_progress
        overlay_suspected = bool((state_res.meta or {}).get("overlay_suspected", False))
        no_progress = bool((hints or {}).get("no_progress", False))
        return (state_res.state == "Unknown" and overlay_suspected) or no_progress

    def decide_next(self, pack: PerceptionPack, ctx: DecideContext) -> tuple[StateDetectResult, ParsedPolicy]:
        hints = ctx.hints or {}
        recent = ctx.recent or {}

        state_res = self.sm.detect_state(pack)

        # ✅ 今天不做“泛化弹窗规则处理”，但保留已识别弹窗的规则动作（可选）
        # 如果你想完全只走 AI，也可以把这段删除，直接走 should_call_ai
        if state_res.state.startswith("Popup."):
            # 这里先不给固定动作，避免发散；直接让上层处理或 Day6 走 AI
            pass

        if self.should_call_ai(state_res, hints):
            # 调用 Qwen，返回结构化动作 JSON（plan 或 ask/stop）
            parsed = self.pr.decide_next(
                pack=pack,
                goal=ctx.goal,
                recent=recent,
                hints={
                    **hints,
                    "state": state_res.state,
                    "overlay_suspected": bool((state_res.meta or {}).get("overlay_suspected", False)),
                    "overlay_hints": (state_res.meta or {}).get("overlay_hints", []),
                },
            )
            return state_res, parsed

        # 不触发 AI：安全停止（今天收官不要求走到执行）
        return state_res, ParsedPolicy(kind="stop", reason="AI not triggered (no overlay_suspected/no_progress)")