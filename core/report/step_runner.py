# core/report/step_runner.py
from __future__ import annotations

import hashlib
import traceback
import time
from pathlib import Path
from typing import Callable, TypeVar, Optional

from core.report.evidence import EvidenceManager
from core.report.step_logger import StepLogger

T = TypeVar("T")


class StepRunner:
    def __init__(self, evidence: EvidenceManager, step_logger: StepLogger, driver_getter, perception=None):
        """
        driver_getter: callable -> driver（通常传 lambda: adapter.driver）
        perception: 可选（Day4），不传也能跑
        """
        self.evidence = evidence
        self.log = step_logger
        self.driver_getter = driver_getter
        self.perception = perception

    def _hash_file(self, path: str) -> Optional[str]:
        try:
            p = Path(path)
            if not p.exists():
                return None
            h = hashlib.md5()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _hash_text(self, text: str) -> Optional[str]:
        try:
            return hashlib.md5((text or "").encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _collect_light_context(self, step, driver):
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

    def _attach_perception(self, step, screenshot_path: Optional[str], page_source_path: Optional[str]):
        """
        Day4（可选）：有 OCR provider + screenshot -> OCR；否则用 page_source 兜底提取可见 text
        不影响 step 成败，best-effort
        """
        try:
            if self.perception and screenshot_path:
                pack = self.perception.perceive_image(screenshot_path)
                step.attach_text("ocr.txt", pack.ocr_text or "")
                step.attach_json("perception.json", pack.meta)
                step.add_extra("perception_available", bool(pack.meta.get("available")))
                step.add_extra("perception_provider", pack.meta.get("provider"))
                step.add_extra("perception_model", pack.meta.get("model"))
                return

            if page_source_path:
                xml = Path(page_source_path).read_text(encoding="utf-8", errors="ignore")
                from core.perception.perception import Perception as P
                pack = P.perceive_from_page_source(xml)
                step.attach_text("ocr.txt", pack.ocr_text or "")
                step.attach_json("perception.json", pack.meta)
                step.add_extra("perception_available", bool(pack.meta.get("available")))
                step.add_extra("perception_provider", pack.meta.get("provider"))
                step.add_extra("perception_model", pack.meta.get("model"))
                return

            step.add_extra("perception_available", False)
            step.add_extra("perception_reason", "no screenshot and no page_source")
        except Exception as e:
            step.add_extra("perception_available", False)
            step.add_extra("perception_error", f"{type(e).__name__}: {e}")

    def _detect_no_progress(
        self,
        step,
        screenshot_before: Optional[str],
        screenshot_after: Optional[str],
        page_source_before: Optional[str],
        page_source_after: Optional[str],
    ):
        """
        no_progress 判定（best-effort）：
        - 优先用 screenshot md5 比对
        - 截图不可用则用 page_source 内容 md5 比对
        """
        try:
            # 1) screenshot hash
            hb = self._hash_file(screenshot_before) if screenshot_before else None
            ha = self._hash_file(screenshot_after) if screenshot_after else None
            if hb and ha:
                same = (hb == ha)
                step.add_extra("no_progress", bool(same))
                step.add_extra("no_progress_basis", "screenshot_md5")
                step.add_extra("before_hash", hb)
                step.add_extra("after_hash", ha)
                return

            # 2) page_source hash（用文件内容）
            tb = None
            ta = None
            if page_source_before:
                tb = self._hash_text(Path(page_source_before).read_text(encoding="utf-8", errors="ignore"))
            if page_source_after:
                ta = self._hash_text(Path(page_source_after).read_text(encoding="utf-8", errors="ignore"))

            if tb and ta:
                same = (tb == ta)
                step.add_extra("no_progress", bool(same))
                step.add_extra("no_progress_basis", "page_source_md5")
                step.add_extra("before_hash", tb)
                step.add_extra("after_hash", ta)
                return

            # 3) 无法判定
            step.add_extra("no_progress", False)
            step.add_extra("no_progress_basis", "unavailable")
            step.add_extra("no_progress_reason", "no comparable screenshot/page_source")
        except Exception as e:
            step.add_extra("no_progress", False)
            step.add_extra("no_progress_basis", "error")
            step.add_extra("no_progress_error", f"{type(e).__name__}: {e}")

    def run(self, name: str, action: str, fn: Callable[[], T]) -> T:
        step = self.evidence.new_step(name=name, action=action)
        self.log.start(self.evidence.run.run_id, step.step_id, action, name)

        screenshot_path = None
        page_source_path = None

        try:
            driver = self.driver_getter()

            # ===== BEFORE evidence =====
            screenshot_path = step.attach_screenshot(driver)      # best-effort
            page_source_path = step.attach_page_source(driver)    # best-effort
            self._collect_light_context(step, driver)
            self._attach_perception(step, screenshot_path, page_source_path)

            # ===== execute action =====
            ret = fn()

            # ===== AFTER evidence (for no_progress) =====
            # 不想每步都多存一份也行，但 Day5 你要 no_progress 触发就需要 after
            screenshot_after = step.attach_screenshot(driver, filename="screenshot_after.png")
            page_source_after = step.attach_page_source(driver, filename="page_source_after.xml")

            self._detect_no_progress(
                step,
                screenshot_before=screenshot_path,
                screenshot_after=screenshot_after,
                page_source_before=page_source_path,
                page_source_after=page_source_after,
            )

            meta = step.finalize()
            self.log.end(meta.run_id, meta.step_id, meta.action, meta.name, meta.result, meta.duration_ms)
            return ret

        except Exception as e:
            stack = traceback.format_exc()
            step.mark_fail(e, stack=stack)

            # 失败时再补一份证据（best-effort）
            try:
                driver = self.driver_getter()
                s2 = step.attach_screenshot(driver, filename="screenshot_fail.png")
                p2 = step.attach_page_source(driver, filename="page_source_fail.xml")
                self._collect_light_context(step, driver)
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