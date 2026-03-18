import os
import time


def test_launch_wechat(appium_adapter, device_config):
    """
    Day 1 Smoke:
    1. 启动微信
    2. 等待首页稳定
    3. 截图保存到 evidence
    """
    app_package = device_config["app"]["appPackage"]
    
    try:
        appium_adapter.terminate_app(app_package)
        time.sleep(2)
    except Exception:
        pass
    
    appium_adapter.activate_app(app_package)
    time.sleep(6)
    size = appium_adapter.get_window_size()
    print(f"screen size: {size}")
    screenshot_path = appium_adapter.screenshot("launch_wechat")
    size = appium_adapter.get_window_size()
    print(f"screen size: {size}")

    assert os.path.exists(screenshot_path), f"Screenshot not found: {screenshot_path}"