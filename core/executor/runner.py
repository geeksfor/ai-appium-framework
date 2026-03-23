# run(goal) 启动
# 第 1 轮先做 observe_1
# StepRunner 产出截图、OCR、meta
# step(goal) 读取最新 perception
# 状态机判断当前页面状态
# 如果已经成功，结束
# 否则先看规则能不能处理
# 规则不行，再判断当前是否值得调用 AI
# 如果满足触发条件，AI 生成下一步 plan
# Executor 执行 plan
# 回到下一轮，再 observe
# 一直到成功、停止或达到最大轮数

# 这就是一个典型的“感知—决策—执行—再感知”的闭环。
# core/executor/runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List, Tuple

from core.perception.perception import PerceptionPack
from core.state.state_machine import StateMachine, StateDetectResult
from core.policy.policy_runner import PolicyRunner
from core.policy.parser import ParsedPolicy
from core.executor.executor import Executor
from core.executor.action_schema import ActionPlan


@dataclass
class RunnerConfig:
    max_rounds: int = 10              # 最大回合数（防止死循环）
    ai_max_calls: int = 3             # 最多调用 AI 次数（预算/安全）
    ai_only_on_trigger: bool = True   # 只在 overlay_suspected/no_progress 时调用 AI
    stop_on_ask_human: bool = True    # ASK_HUMAN 是否直接停止
    stop_on_ai_stop: bool = True      # SCREENSHOT_AND_STOP 是否停止
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
    Day7 终极闭环（收敛版）：
    - 每回合：detect_state -> rule_handle? -> else ai_policy -> executor
    - AI 只有在触发条件成立时调用（overlay_suspected/no_progress/Unknown）
    - 所有 AI 输入/输出归档到 evidence/<run_id>/policy/
    - plan 必须通过 ActionPlan 校验（PolicyRunner 已做 parse+validate）
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

    # ---------- PerceptionPack 从 evidence step 目录读取 ----------
    def _load_latest_perception(self) -> PerceptionPack:
        """
        约定：StepRunner 每步都会在 step 目录写：
        - ocr.txt（可选）
        - perception.json（可选）
        Runner 在每回合开始时，读取“最后一个 step 目录”的 perception。
        """
        run_dir = Path(self.evidence.run.run_dir)
        step_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        if not step_dirs:
            # 没有 step：返回空 pack
            return PerceptionPack(image_path="", ocr_text="", meta={"available": False, "reason": "no steps yet"})

        last = step_dirs[-1]
        ocr_path = last / "ocr.txt"
        meta_path = last / "perception.json"

        ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"available": False}

        # screenshot 通常叫 screenshot.png / screenshot_after.png，优先取 after
        img = ""
        for cand in ["screenshot_after.png", "screenshot.png", "screenshot_fail.png"]:
            p = last / cand
            if p.exists():
                img = str(p)
                break

        return PerceptionPack(image_path=img, ocr_text=ocr_text, meta=meta)

    # ---------- 触发条件 ----------
    def _should_call_ai(self, state_res: StateDetectResult) -> bool:
        if self._ai_calls >= self.cfg.ai_max_calls:
            return False
        overlay_suspected = bool((state_res.meta or {}).get("overlay_suspected", False))
        no_progress = bool((self._hints or {}).get("no_progress", False))

        if not self.cfg.ai_only_on_trigger:
            # 不建议，但保留开关
            return state_res.state == "Unknown" or overlay_suspected or no_progress

        # 收敛策略：Unknown+overlay_suspected 或 no_progress 才调用
        return (state_res.state == "Unknown" and overlay_suspected) or no_progress

    # ---------- 一回合 ----------
    def step(self, goal: Dict[str, Any]) -> Tuple[bool, str, StateDetectResult]:
        """
        执行一个回合：
        - 读 perception
        - state detect
        - success_check?
        - rule_handler?（可选）
        - AI fallback（PolicyRunner）-> plan -> executor
        """
        pack = self._load_latest_perception()
        state_res = self.sm.detect_state(pack)

        if self.cfg.verbose:
            print(f"[runner] state={state_res.state} score={state_res.score} overlay={state_res.meta.get('overlay_suspected')}")

        # 1) 成功判定（可选）
        if self.success_check:
            ok, reason = self.success_check(state_res, pack)
            if ok:
                return True, f"success: {reason}", state_res

        # 2) 规则处理（可选，不发散：默认不处理）
        if self.rule_handler:
            plan = self.rule_handler(state_res, pack)
            if plan is not None:
                self.executor.run_plan(plan)
                self._recent = {"source": "rule", "state": state_res.state}
                return False, "executed rule plan", state_res

        # 3) AI 兜底
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

        # stop
        if self.cfg.stop_on_ai_stop:
            return False, f"stop: {parsed.reason}", state_res
        return False, f"ai stop: {parsed.reason}", state_res

    # ---------- 主循环 ----------
    def run(self, goal: Dict[str, Any]) -> RunnerResult:
        """
        主循环：每轮都用 StepRunner 跑一个“观测 step”，确保有新证据目录。
        为了不发散，我们把观测做成一个 WAIT step（强制截图/ocr/no_progress 产出）。
        """
        run_id = self.evidence.run.run_id
        last_state = "Unknown"
        last_reason = "not started"

        for i in range(1, self.cfg.max_rounds + 1):
            self._rounds = i

            # 每回合先跑一个观测 step，产生新的 evidence step 目录（截图/ocr/no_progress）
            def _observe():
                # 这里不做任何操作，只是给 StepRunner 一个“采集证据”的机会
                return None

            self.step_runner.run(name=f"observe_{i}", action="OBSERVE", fn=_observe)

            ok, reason, state_res = self.step(goal)
            last_state = state_res.state
            last_reason = reason

            if ok:
                return RunnerResult(
                    ok=True, rounds=i, ai_calls=self._ai_calls,
                    last_state=last_state, last_reason=last_reason, run_id=run_id
                )

            # 如果 stop，就结束（收敛策略：不无限迭代）
            if reason.startswith("stop:"):
                break

            # 更新 hints：从最新 step 的 meta extra 读取 no_progress（如果你 StepRunner 已写入）
            # 简化：我们从最后一个 step 的 meta.json 里读 extra.no_progress
            self._refresh_hints_from_latest_step()

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
            # no_progress 是关键触发信号之一
            if "no_progress" in extra:
                self._hints["no_progress"] = bool(extra.get("no_progress"))
        except Exception:
            pass