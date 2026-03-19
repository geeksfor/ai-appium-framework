# core/report/step_runner.py
from __future__ import annotations

import traceback
from typing import Callable, TypeVar

from core.report.evidence import EvidenceManager
from core.report.step_logger import StepLogger

T = TypeVar("T")


class StepRunner:
    def __init__(self, evidence: EvidenceManager, step_logger: StepLogger, driver_getter):
        """
        driver_getter: callable -> driver（通常传 lambda: adapter.driver）
        """
        self.evidence = evidence
        self.log = step_logger
        self.driver_getter = driver_getter
    def _collect_light_context(self, step, driver):
        # 这些“轻量信息”很少失败，且对排障/AI有用
        try:
            step.add_extra("current_package", getattr(driver, "current_package", None))
        except Exception as e:
            step.add_extra("current_package_error", f"{type(e).__name__}: {e}")

        try:
            step.add_extra("current_activity", getattr(driver, "current_activity", None))
        except Exception as e:
            step.add_extra("current_activity_error", f"{type(e).__name__}: {e}")

        try:
            step.add_extra("window_size", driver.get_window_size())
        except Exception as e:
            step.add_extra("window_size_error", f"{type(e).__name__}: {e}")

    def run(self, name: str, action: str, fn: Callable[[], T]) -> T:
        step = self.evidence.new_step(name=name, action=action)

        # 开始日志
        self.log.start(self.evidence.run.run_id, step.step_id, action, name)

        try:
            driver = self.driver_getter()
            # 每步开始先保存一份 screenshot + pageSource（你也可以改成结束再存）
            _ = step.attach_screenshot(driver)
            _ = step.attach_page_source(driver)
            self._collect_light_context(step, driver)

            ret = fn()

            meta = step.finalize()
            self.log.end(meta.run_id, meta.step_id, meta.action, meta.name, meta.result, meta.duration_ms)
            return ret

        except Exception as e:
            stack = traceback.format_exc()
            step.mark_fail(e, stack=stack)

            # 尽量在失败时再补一张现场图
            stack = traceback.format_exc()
            step.mark_fail(e, stack=stack)

            # ✅ 失败时再补一次证据：仍然 best-effort
            try:
                driver = self.driver_getter()
                _ = step.attach_screenshot(driver, filename="screenshot_fail.png")
                _ = step.attach_page_source(driver, filename="page_source_fail.xml")
                self._collect_light_context(step, driver)
            except Exception:
                pass

            meta = step.finalize()
            self.log.end(
                meta.run_id, meta.step_id, meta.action, meta.name,
                meta.result, meta.duration_ms,
                error={"type": meta.error_type, "message": meta.error_message}
            )
            raise