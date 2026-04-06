# core/flows/navigate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from core.executor.action_schema import ActionPlan
from core.executor.executor import Executor
from core.executor.runner import Runner
from core.policy.policy_runner import PolicyRunner
from core.perception.perception import PerceptionPack


@dataclass
class NavigateResult:
    ok: bool
    reason: str
    run_id: str


class Navigator:
    """
    收官版导航：只实现 goto("BloodSugarReport")，策略最稳，不做复杂分支。
    - 优先用固定动作序列（点击搜索框/输入/回车/点结果）
    - 定位不准时交给 Runner/PolicyRunner（CLICK_TEXT）兜底
    """

    def __init__(self, runner: Runner, policy_runner: PolicyRunner, executor: Executor):
        self.runner = runner
        self.pr = policy_runner
        self.exe = executor

    def goto(self, target: str) -> NavigateResult:
        if target != "BloodSugarReport":
            return NavigateResult(False, f"unsupported target: {target}", self.runner.evidence.run.run_id)

        # 目标：进入“我的血糖报告”页（后续你可以用 state+ocr 来做 success_check）
        goal = {"intent": "REACH_STATE", "target_state": "BloodSugarReport", "must_contain_text": ["我的血糖报告"]}

        # Step1: 尝试点微信首页顶部搜索（模板坐标：靠上居中偏左）
        # 注意：不同机型会变，这里只是“先试”，失败由 runner 的 no_progress/AI 兜底处理
        plan1 = ActionPlan.model_validate({
            "actions": [
                {"type": "CLICK", "name": "tap_wechat_search", "x_pct": 0.50, "y_pct": 0.12},
                {"type": "WAIT", "name": "wait_search", "seconds": 0.6},
                {"type": "INPUT", "name": "input_miniprogram", "x_pct": 0.50, "y_pct": 0.12, "text": "轻糖乐活", "clear_first": True, "press_enter": True},
                {"type": "WAIT", "name": "wait_results", "seconds": 1.0},
            ]
        })
        self.exe.run_plan(plan1)

        # Step2: 用 AI click_text 点搜索结果/入口（更稳），再点“我的血糖报告”
        # 这里直接调用 policy_runner 两次（收敛，不写复杂循环），失败交给 runner 兜底。
        pack = self._latest_pack()
        p1 = self.pr.decide_next(pack, goal={"intent": "CLICK_TEXT", "target_text": "轻糖乐活", "constraints": {"max_steps": 1}}, hints={"no_progress": True})
        if p1.kind == "plan" and p1.plan:
            self.exe.run_plan(p1.plan)

        pack = self._latest_pack()
        p2 = self.pr.decide_next(pack, goal={"intent": "CLICK_TEXT", "target_text": "我的血糖报告", "constraints": {"max_steps": 1}}, hints={"no_progress": True})
        if p2.kind == "plan" and p2.plan:
            self.exe.run_plan(p2.plan)

        # Step3: 交给 runner 跑一个短循环，处理弹窗 + 最终到达检查（你可以在 runner.success_check 做更准）
        result = self.runner.run(goal)
        return NavigateResult(result.ok, result.last_reason, result.run_id)

    def _latest_pack(self) -> PerceptionPack:
        # 复用 Runner 的能力（不重复造轮子）
        return self.runner._load_latest_perception()