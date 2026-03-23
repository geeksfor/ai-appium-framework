# tests/smoke/test_ai_fallback.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.executor.runner import Runner, RunnerConfig
from core.executor.action_schema import ActionPlan
from core.policy.parser import ParsedPolicy
from core.perception.perception import PerceptionPack


# --- Stub StateMachine：强制触发 AI ---
@dataclass
class _StubStateRes:
    state: str
    score: float
    matches: list
    best: object | None
    meta: dict


class StubStateMachine:
    def detect_state(self, pack: PerceptionPack):
        # 强制 Unknown + overlay_suspected -> 触发 AI
        return _StubStateRes(
            state="Unknown",
            score=0.0,
            matches=[],
            best=None,
            meta={"overlay_suspected": True, "overlay_hints": ["关闭", "跳过"]},
        )


class FakePolicyRunner:
    """模拟 Qwen：永远返回 plan，并写 evidence/policy/ 归档"""
    def __init__(self, evidence_manager):
        self.evidence = evidence_manager

    def decide_next(self, pack: PerceptionPack, goal, recent=None, hints=None) -> ParsedPolicy:
        policy_dir = Path(self.evidence.run.run_dir) / "policy"
        policy_dir.mkdir(parents=True, exist_ok=True)

        (policy_dir / "policy_input.json").write_text(
            json.dumps(
                {
                    "goal": goal,
                    "perception": {"ocr_text": pack.ocr_text, "meta": pack.meta, "image_path": pack.image_path},
                    "recent": recent or {},
                    "hints": hints or {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        raw = {"actions": [{"type": "WAIT", "name": "wait", "seconds": 1}]}
        (policy_dir / "policy_raw.txt").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

        plan = ActionPlan.model_validate(raw)
        (policy_dir / "policy_parsed.json").write_text(
            json.dumps({"kind": "plan", "plan": plan.model_dump()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ParsedPolicy(kind="plan", plan=plan, raw_json=raw)


class FakeExecutor:
    """不连真机：只记录是否执行过 plan"""
    def __init__(self):
        self.called = 0

    def run_plan(self, plan: ActionPlan):
        self.called += 1
        return None


def test_ai_fallback_closed_loop(evidence_manager, step_runner):
    sm = StubStateMachine()
    pr = FakePolicyRunner(evidence_manager)
    exe = FakeExecutor()

    runner = Runner(
        evidence_manager=evidence_manager,
        step_runner=step_runner,
        state_machine=sm,      # ✅ stub，保证触发 AI
        policy_runner=pr,      # ✅ fake，写 policy 归档
        executor=exe,          # ✅ fake，统计执行次数
        cfg=RunnerConfig(max_rounds=1, ai_max_calls=1, verbose=False),
        success_check=lambda state_res, pack: (False, "not used"),
    )

    goal = {"intent": "CLOSE_OVERLAY", "strategy": "SAFE"}
    result = runner.run(goal)

    assert exe.called >= 1
    assert result.ai_calls == 1

    policy_dir = Path(evidence_manager.run.run_dir) / "policy"
    assert (policy_dir / "policy_input.json").exists()
    assert (policy_dir / "policy_raw.txt").exists()
    assert (policy_dir / "policy_parsed.json").exists()