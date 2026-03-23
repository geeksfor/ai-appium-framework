# tests/smoke/test_day6_final_closeout.py
import pytest
from pathlib import Path

from core.perception.perception import PerceptionPack
from core.state.state_machine import StateMachine
from core.policy.policy_runner import PolicyRunner, PolicyRunnerConfig
from core.policy.decider import Decider, DecideContext
from core.policy.qwen_client import QwenResponse


class FakeQwenClient:
    """
    稳定返回一个合法的 ActionPlan（不打外网、不计费）。
    用于验证“能拿回结构化动作 JSON + parser 校验 + evidence 归档”。
    """
    def chat(self, system_prompt: str, user_payload: str, temperature: float = 0.0, max_tokens: int = 512, extra=None):
        # 返回 JSON（字符串）
        content = """
        {
          "actions": [
            { "type": "WAIT", "name": "wait_a_bit", "seconds": 1 },
            { "type": "BACK", "name": "back_close_overlay" }
          ]
        }
        """.strip()
        return QwenResponse(content=content, elapsed_ms=5, raw={"model": "fake-qwen", "id": "fake"})


@pytest.fixture
def policy_prompt_file(tmp_path: Path):
    # 给 PolicyRunner 一个最小 prompt 文件（避免依赖 repo 文件路径/内容）
    p = tmp_path / "policy_v1.txt"
    p.write_text("只输出JSON。", encoding="utf-8")
    return str(p)


def test_day6_final_closeout(evidence_manager, policy_prompt_file):
    # 1) 准备 perception pack（模拟一个可疑 overlay/卡住场景）
    pack = PerceptionPack(
        image_path="",
        ocr_text="嗨，请登录 登录之后有新天地 关闭 跳过",  # 让 overlay_suspected 有机会为 True
        meta={"available": True},
    )

    # 2) 组装最小链路
    sm = StateMachine()
    runner = PolicyRunner(
        qwen=FakeQwenClient(),
        evidence_manager=evidence_manager,
        cfg=PolicyRunnerConfig(prompt_path=policy_prompt_file),
    )
    decider = Decider(state_machine=sm, policy_runner=runner)

    # 3) goal 用 schema（保持你 Day6 的收敛设计）
    goal = {"intent": "CLOSE_OVERLAY", "strategy": "SAFE"}

    # 4) 触发 AI（用 no_progress 强制触发，避免依赖规则输出）
    state_res, parsed = decider.decide_next(
        pack=pack,
        ctx=DecideContext(
            goal=goal,
            recent={"type": "CLICK", "x_pct": 0.5, "y_pct": 0.5},
            hints={"no_progress": True},
        ),
    )

    # 5) 验收：能拿到结构化 plan，并通过 ActionPlan 校验
    assert parsed.kind == "plan"
    assert parsed.plan is not None
    assert len(parsed.plan.actions) == 2

    # 6) 验收：evidence 归档完整（policy 输入/输出都能回放）
    policy_dir = Path(evidence_manager.run.run_dir) / "policy"
    assert (policy_dir / "policy_input.json").exists()
    assert (policy_dir / "policy_prompt.txt").exists()
    assert (policy_dir / "policy_raw.txt").exists()
    assert (policy_dir / "policy_parsed.json").exists()