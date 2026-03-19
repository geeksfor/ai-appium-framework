# tests/smoke/test_launch_wechat.py
import time


def test_launch_wechat(appium_adapter, step_runner):
    step_runner.run(
        name="launch_wechat",
        action="activate_app",
        fn=lambda: appium_adapter.driver.activate_app("com.tencent.mm")
    )

    step_runner.run(
        name="wait_home",
        action="sleep",
        fn=lambda: time.sleep(5)
    )

    # 你也可以显式再存一次“最终态截图”
    step_runner.run(
        name="final_screenshot",
        action="screenshot",
        fn=lambda: appium_adapter.screenshot("launch_wechat_final")
    )