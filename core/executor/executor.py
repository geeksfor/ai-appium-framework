# core/executor/executor.py
from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from selenium.webdriver.common.keys import Keys

from core.driver.appium_adapter import AppiumAdapter
from core.executor.action_schema import (
    Action,
    ActionPlan,
    ActionType,
    AssertAction,
    BackAction,
    ClickAction,
    InputAction,
    SelectAction,
    SwipeAction,
    WaitAction,
)
from core.perception.perception import Perception, PerceptionPack
from core.report.evidence import EvidenceStep
from core.report.step_runner import StepRunner


@dataclass
class SimplePack:
    image_path: str
    ocr_text: str
    meta: dict


@dataclass
class ExecResult:
    ok: bool
    executed: int
    last_action: dict | None = None
    actions_path: str | None = None


class Executor:
    def __init__(
        self,
        adapter: AppiumAdapter,
        step_runner: StepRunner,
        perception: Any | None = None,
        click_healer: Any | None = None,
    ):
        self.adapter = adapter
        self.step_runner = step_runner
        self.perception = perception
        self.click_healer = click_healer
        self._heal_seq = 0

    def run_plan(self, plan: ActionPlan) -> ExecResult:
        executed = 0
        last_action: dict | None = None
        actions_path = self._archive_actions(plan)

        for idx, act in enumerate(plan.actions, start=1):
            step_name = act.name or f"{idx:02d}_{act.type}"
            last_action = act.model_dump()

            self.step_runner.run(
                name=step_name,
                action=str(act.type),
                fn=lambda a=act: self._execute_one(a),
            )
            executed += 1

        return ExecResult(ok=True, executed=executed, last_action=last_action, actions_path=actions_path)

    def _execute_one(self, act: Action) -> Any:
        if act.type == ActionType.CLICK:
            return self._do_click(act)  # type: ignore[arg-type]
        if act.type == ActionType.WAIT:
            return self._do_wait(act)  # type: ignore[arg-type]
        if act.type == ActionType.BACK:
            return self._do_back(act)  # type: ignore[arg-type]
        if act.type == ActionType.ASSERT:
            return self._do_assert(act)  # type: ignore[arg-type]
        if act.type == ActionType.INPUT:
            return self._do_input(act)  # type: ignore[arg-type]
        if act.type == ActionType.SELECT:
            return self._do_select(act)
        if act.type == ActionType.SWIPE:
            return self._do_swipe(act)
        raise ValueError(f"Unsupported action type: {act.type}")

    def _do_click(self, act: ClickAction) -> None:
        if act.x is not None and act.y is not None:
            x, y = int(act.x), int(act.y)
        else:
            size = self.adapter.get_window_size()
            x = int(size["width"] * float(act.x_pct))
            y = int(size["height"] * float(act.y_pct))

        if hasattr(act, "allow_heal") and not act.allow_heal:
            self.adapter.tap(x, y)
            return

        try:
            self.adapter.tap(x, y)
            return
        except Exception as tap_err:
            if self.click_healer is None:
                raise tap_err

            heal_selector = self._pick_heal_selector(act)
            if not heal_selector:
                raise tap_err

            pack = self._build_pack()
            save_path = self._next_heal_save_path(heal_selector)

            try:
                result = self.click_healer.heal_click(
                    pack=pack,
                    selector=heal_selector,
                    logical_name=self._str_or_none(getattr(act, "logical_name", "")),
                    save_path=save_path,
                )
            except Exception:
                raise tap_err

            if not result or not result.healed or result.chosen is None:
                raise tap_err

            size = self.adapter.get_window_size()
            heal_x = int(size["width"] * float(result.chosen.x_pct))
            heal_y = int(size["height"] * float(result.chosen.y_pct))
            self.adapter.tap(heal_x, heal_y)

    def _do_wait(self, act: WaitAction) -> None:
        time.sleep(float(act.seconds))

    def _do_back(self, act: BackAction) -> None:
        self.adapter.back()

    def _do_assert(self, act: AssertAction) -> None:
        src = self.adapter.driver.page_source if self.adapter.driver else ""
        if not src:
            raise AssertionError("ASSERT failed: page_source is empty/unavailable.")

        hay = src.lower() if act.ignore_case else src
        needle = act.contains_text.lower() if act.ignore_case else act.contains_text
        if needle not in hay:
            raise AssertionError(f"ASSERT failed: text not found: {act.contains_text}")

    def _xy_from_abs_or_pct(self, x, y, x_pct, y_pct) -> tuple[int, int]:
        if x is not None and y is not None:
            return int(x), int(y)
        size = self.adapter.get_window_size()
        return int(size["width"] * float(x_pct)), int(size["height"] * float(y_pct))

    def _do_input(self, act: InputAction) -> None:
        driver = self.adapter.driver
        if not driver:
            raise RuntimeError("Driver not started")

        x, y = self._xy_from_abs_or_pct(act.x, act.y, act.x_pct, act.y_pct)
        self.adapter.tap(x, y)
        time.sleep(0.2)

        if act.clear_first:
            try:
                driver.switch_to.active_element.clear()
            except Exception:
                pass

        try:
            driver.switch_to.active_element.send_keys(act.text)
        except Exception:
            driver.execute_script("mobile: type", {"text": act.text})

        if act.press_enter:
            try:
                driver.switch_to.active_element.send_keys(Keys.ENTER)
            except Exception:
                pass

    def _do_select(self, act: SelectAction) -> None:
        ox, oy = self._xy_from_abs_or_pct(
            act.open_x,
            act.open_y,
            act.open_x_pct,
            act.open_y_pct,
        )
        self.adapter.tap(ox, oy)
        time.sleep(0.3)

        px, py = self._xy_from_abs_or_pct(
            act.option_x,
            act.option_y,
            act.option_x_pct,
            act.option_y_pct,
        )
        self.adapter.tap(px, py)

    def _do_swipe(self, act: SwipeAction) -> None:
        driver = self.adapter.driver
        if not driver:
            raise RuntimeError("Driver not started")

        size = self.adapter.get_window_size()
        w, h = size["width"], size["height"]

        left = int(w * act.left_pct)
        top = int(h * act.top_pct)
        width = int(w * act.width_pct)
        height = int(h * act.height_pct)

        driver.execute_script(
            "mobile: swipeGesture",
            {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "direction": act.direction.value.lower(),
                "percent": float(act.percent),
                "speed": int(act.speed),
            },
        )

    def _pick_heal_selector(self, act: ClickAction) -> str | None:
        selector = self._str_or_none(getattr(act, "selector", ""))
        if selector:
            return selector

        target = self._str_or_none(getattr(act, "target", ""))
        if target:
            return f"text={target}"

        target_text = self._str_or_none(getattr(act, "target_text", ""))
        if target_text:
            return f"text={target_text}"

        logical_name = self._str_or_none(getattr(act, "logical_name", ""))
        if logical_name:
            return logical_name

        if act.name:
            return str(act.name)

        return None

    def _str_or_none(self, value: Any) -> str | None:
        s = str(value or "").strip()
        return s if s else None

    def _build_pack(self) -> Any:
        """
        给点击自愈构造一个真正可用的当前页面 pack：
        1) 优先现场截图 + OCR
        2) 再退化到 page_source 文本感知
        3) 最后再退回最小空 pack
        """
        driver = getattr(self.adapter, "driver", None)

        screenshot_path = self._capture_tmp_screenshot(driver)
        page_source = self._safe_page_source(driver)

        pack = self._build_pack_from_live_inputs(screenshot_path=screenshot_path, page_source=page_source)
        if pack is not None:
            return pack

        if self.perception is not None:
            for method_name in ["build_pack", "build", "collect", "perceive", "run"]:
                method = getattr(self.perception, method_name, None)
                if callable(method):
                    try:
                        pack = method()
                        if pack is not None:
                            return pack
                    except TypeError:
                        continue
                    except Exception:
                        break

        return SimplePack(
            image_path=screenshot_path or "",
            ocr_text="",
            meta={
                "ocr_boxes": [],
                "available": False,
                "reason": "live_perception_unavailable",
                "page_source_available": bool(page_source),
            },
        )

    def _build_pack_from_live_inputs(
        self,
        screenshot_path: Optional[str],
        page_source: Optional[str],
    ) -> Optional[PerceptionPack | SimplePack]:
        if self.perception is not None and screenshot_path:
            try:
                if hasattr(self.perception, "perceive_image"):
                    pack = self.perception.perceive_image(screenshot_path)
                    pack.meta.setdefault("page_source_available", bool(page_source))
                    if page_source:
                        pack.meta.setdefault("page_source_length", len(page_source))
                    return pack
            except Exception:
                pass

        if page_source:
            try:
                pack = Perception.perceive_from_page_source(page_source)
                pack.image_path = screenshot_path or ""
                pack.meta.setdefault("ocr_boxes", [])
                pack.meta["page_source_available"] = True
                return pack
            except Exception:
                pass

        return None

    def _capture_tmp_screenshot(self, driver: Any) -> Optional[str]:
        if driver is None:
            return None

        root = Path("artifacts") / "heal" / "_tmp"
        root.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(prefix="live_", suffix=".png", dir=str(root), delete=False) as f:
                path = f.name
        except Exception:
            path = str(root / f"live_{int(time.time() * 1000)}.png")

        try:
            ok = driver.save_screenshot(path)
            if ok:
                return path
        except Exception:
            return None
        return None

    def _safe_page_source(self, driver: Any, timeout_sec: int = 3) -> Optional[str]:
        if driver is None:
            return None

        try:
            return EvidenceStep.get_page_source_with_timeout(driver, timeout_sec=timeout_sec)
        except Exception:
            pass

        try:
            return driver.page_source
        except Exception:
            return None

    def _next_heal_save_path(self, target_name: str) -> Path:
        self._heal_seq += 1
        root = Path("artifacts") / "heal"
        root.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", str(target_name or "unknown"))
        safe_name = safe_name.strip("_") or "unknown"

        d = root / f"{self._heal_seq:03d}_{safe_name[:40]}"
        d.mkdir(parents=True, exist_ok=True)
        return d / "heal_candidates.json"

    def _archive_actions(self, plan: ActionPlan) -> Optional[str]:
        try:
            root = Path(self.step_runner.evidence.run.run_dir)
            p = root / "actions.json"
            payload = plan.model_dump(mode="json")
            p.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(p)
        except Exception:
            return None
