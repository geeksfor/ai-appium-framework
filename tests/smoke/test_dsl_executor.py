from core.executor.executor import Executor
from core.executor.plan_loader import load_plan


def test_dsl_executor_from_file(appium_adapter, step_runner):
    plan = load_plan("tests/plans/wechat_smoke.json")

    exe = Executor(adapter=appium_adapter, step_runner=step_runner)
    result = exe.run_plan(plan)

    assert result.ok is True
    assert result.executed == 3