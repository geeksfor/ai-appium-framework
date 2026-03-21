# core/report/step_runner.py
from __future__ import annotations

import traceback
from typing import Callable, TypeVar

from core.report.evidence import EvidenceManager
from core.report.step_logger import StepLogger
from pathlib import Path
from core.perception.perception import Perception as PerceptionFacade

T = TypeVar("T")


class StepRunner:
    def __init__(self, evidence: EvidenceManager, step_logger: StepLogger, driver_getter, perception=None):
        """
        driver_getter: callable -> driver（通常传 lambda: adapter.driver）
        """
        self.evidence = evidence
        self.log = step_logger
        self.driver_getter = driver_getter
        self.perception = perception

    def _attach_perception(self, step, screenshot_path: str | None, page_source_path: str | None):
        """
        Perception v1:
        1) 优先：截图 + OCR（Qwen OCR / Qwen VL）
        2) 兜底：page_source 提取可见 text
        都是 best-effort：失败不影响 step 成败
        """
        try:
            # 1) 有 OCR provider 且有截图 -> OCR
            if self.perception and screenshot_path:
                pack = self.perception.perceive_image(screenshot_path)
                step.attach_text("ocr.txt", pack.ocr_text or "")
                step.attach_json("perception.json", pack.meta)

                step.add_extra("perception_available", bool(pack.meta.get("available")))
                step.add_extra("perception_provider", pack.meta.get("provider"))
                step.add_extra("perception_model", pack.meta.get("model"))
                return

            # 2) 没 OCR 或截图失败 -> 用 page_source 兜底
            if page_source_path:
                xml = Path(page_source_path).read_text(encoding="utf-8", errors="ignore")
                pack = PerceptionFacade.perceive_from_page_source(xml)

                step.attach_text("ocr.txt", pack.ocr_text or "")
                step.attach_json("perception.json", pack.meta)

                step.add_extra("perception_available", bool(pack.meta.get("available")))
                step.add_extra("perception_provider", pack.meta.get("provider"))
                step.add_extra("perception_model", pack.meta.get("model"))
                return

            # 3) 两者都没有
            step.add_extra("perception_available", False)
            step.add_extra("perception_reason", "no screenshot and no page_source")

        except Exception as e:
            step.add_extra("perception_available", False)
            step.add_extra("perception_error", f"{type(e).__name__}: {e}")

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
            screenshot_path = step.attach_screenshot(driver)
            page_source_path = step.attach_page_source(driver)
            self._collect_light_context(step, driver)
            
            # ✅ Day4：每步自动做 Perception（best-effort）
            self._attach_perception(step, screenshot_path, page_source_path)

            ret = fn()

            meta = step.finalize()
            self.log.end(meta.run_id, meta.step_id, meta.action, meta.name, meta.result, meta.duration_ms)
            return ret

        except Exception as e:
            stack = traceback.format_exc()
            step.mark_fail(e, stack=stack)

            # ✅ 失败时再补一次证据：仍然 best-effort
            try:
                driver = self.driver_getter()
                s2 = step.attach_screenshot(driver, filename="screenshot_fail.png")
                p2 = step.attach_page_source(driver, filename="page_source_fail.xml")
                self._collect_light_context(step, driver)

                # ✅ 失败时也做一次 Perception（优先用 fail 的截图/DOM）
                self._attach_perception(step, s2 or screenshot_path, p2 or page_source_path)
            except Exception:
                pass

            meta = step.finalize()
            self.log.end(
                meta.run_id, meta.step_id, meta.action, meta.name,
                meta.result, meta.duration_ms,
                error={"type": meta.error_type, "message": meta.error_message}
            )
            raise