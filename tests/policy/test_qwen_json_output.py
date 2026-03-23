# tests/policy/test_qwen_json_output.py
import pytest

from core.policy.parser import parse_policy_output


def test_parser_accepts_action_plan_json():
    model_out = """
    ```json
    {
      "actions": [
        { "type": "WAIT", "name": "wait_ready", "seconds": 1 },
        { "type": "BACK", "name": "go_back" }
      ]
    }
    ```
    """.strip()

    parsed = parse_policy_output(model_out)
    assert parsed.kind == "plan"
    assert parsed.plan is not None
    assert len(parsed.plan.actions) == 2


def test_parser_accepts_ask_human():
    model_out = '{ "type": "ASK_HUMAN", "reason": "OCR不足以定位关闭按钮" }'
    parsed = parse_policy_output(model_out)
    assert parsed.kind == "ask_human"
    assert "OCR不足" in (parsed.reason or "")


def test_parser_rejects_non_json():
    with pytest.raises(Exception):
        parse_policy_output("我觉得你应该点右上角的X")