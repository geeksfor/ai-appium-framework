from __future__ import annotations

from typing import Optional

from appium.webdriver.webdriver import WebDriver
from selenium.common.exceptions import WebDriverException

from core.driver.driver_factory import create_android_driver
from core.utils.file_utils import build_screenshot_path
from core.utils.logger import get_logger


class AppiumAdapter:
    """
    对 Appium Driver 做一层轻封装，屏蔽测试层的底层细节。

    Day 1 先提供：
    - 启动 driver
    - 关闭 driver
    - 截图
    - 返回
    - 点击坐标
    - 获取屏幕尺寸

    后续可继续加：
    - 页面源码获取
    - 元素查找
    - AI动作执行
    - 等待策略
    """

    def __init__(
        self,
        server_url: str,
        capabilities: dict,
        implicit_wait: int = 10,
        evidence_dir: str = "evidence",
    ):
        self.server_url = server_url
        self.capabilities = capabilities
        self.implicit_wait = implicit_wait
        self.evidence_dir = evidence_dir
        self.driver: Optional[WebDriver] = None
        self.logger = get_logger(self.__class__.__name__)

    def start(self) -> WebDriver:
        """启动 Appium 会话"""
        if self.driver is not None:
            self.logger.info("Driver already started.")
            return self.driver

        self.logger.info("Starting Appium driver...")
        self.driver = create_android_driver(
            server_url=self.server_url,
            capabilities=self.capabilities,
        )
        self.driver.implicitly_wait(self.implicit_wait)
        self.logger.info("Appium driver started successfully.")
        return self.driver

    def quit(self) -> None:
        """关闭 Appium 会话"""
        if self.driver is not None:
            self.logger.info("Quitting Appium driver...")
            self.driver.quit()
            self.driver = None
            self.logger.info("Appium driver quit successfully.")

    def screenshot(self, name_prefix: str = "screenshot") -> str:
        """截图并保存到 evidence 目录"""
        self._ensure_driver()
        file_path = build_screenshot_path(self.evidence_dir, name_prefix)
        ok = self.driver.save_screenshot(file_path)
        if not ok:
            raise RuntimeError(f"Failed to save screenshot to {file_path}")
        self.logger.info(f"Screenshot saved: {file_path}")
        return file_path

    def back(self) -> None:
        """返回上一层"""
        self._ensure_driver()
        self.logger.info("Navigate back.")
        self.driver.back()

    def tap(self, x: int, y: int) -> None:
        """
        点击坐标。
        优先使用 mobile: clickGesture，兼容新版本 Appium / UiAutomator2
        """
        self._ensure_driver()
        self.logger.info(f"Tap at coordinates: ({x}, {y})")
        try:
            self.driver.execute_script("mobile: clickGesture", {"x": x, "y": y})
        except WebDriverException:
            # 部分环境不支持 clickGesture 时，可退回旧方式
            # 不过新版更推荐 mobile gesture 系列
            raise

    def get_window_size(self) -> dict:
        """获取屏幕尺寸"""
        self._ensure_driver()
        size = self.driver.get_window_size()
        self.logger.info(f"Window size: {size}")
        return size

    def _ensure_driver(self) -> None:
        if self.driver is None:
            raise RuntimeError("Driver is not started. Please call start() first.")

    def activate_app(self, app_package: str) -> None:
      self._ensure_driver()
      self.logger.info(f"Activate app: {app_package}")
      self.driver.activate_app(app_package)

    def terminate_app(self, app_package: str) -> None:
        self._ensure_driver()
        self.logger.info(f"Terminate app: {app_package}")
        self.driver.terminate_app(app_package)