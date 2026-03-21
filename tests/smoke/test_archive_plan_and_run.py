from core.executor.plan_archive import archive_plan_to_evidence, archive_raw_text_to_evidence
from core.executor.plan_loader import load_plan


def test_archive_plan_and_run(executor, evidence_manager):
    # 假设这是 AI 输出的 JSON 字符串（先用静态代替）
    ai_output = """
    {
      "actions": [
        { "type": "WAIT", "name": "wait_ready", "seconds": 1 },
        { "type": "CLICK", "name": "tap_center", "x_pct": 0.5, "y_pct": 0.35 },
        { "type": "BACK", "name": "go_back" }
      ]
    }
    """.strip()

    # 1) 原始输出归档（就算不是合法 JSON 也能留档）
    archive_raw_text_to_evidence(evidence_manager, ai_output, filename="ai_raw_output.txt")

    # 2) 解析并归档成标准 plan（ai_plan.json）
    plan_path = archive_plan_to_evidence(evidence_manager, ai_output, filename="ai_plan.json")

    # 3) 从文件 load + schema 校验，再执行（可回放）
    plan = load_plan(plan_path)
    result = executor.run_plan(plan)

    assert result.ok is True
    assert result.executed == 3