from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from core.perception.perception import PerceptionPack
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
    def __init__(self, adapter: AppiumAdapter, step_runner: StepRunner, perception: Any | None = None, click_healer: Any | None = None):
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
            self.step_runner.run(name=step_name, action=str(act.type), fn=lambda a=act: self._execute_one(a))
            executed += 1
        return ExecResult(ok=True, executed=executed, last_action=last_action, actions_path=actions_path)

    def _execute_one(self, act: Action) -> Any:
        if act.type == ActionType.CLICK:
            return self._do_click(act)
        if act.type == ActionType.WAIT:
            return self._do_wait(act)
        if act.type == ActionType.BACK:
            return self._do_back(act)
        if act.type == ActionType.ASSERT:
            return self._do_assert(act)
        if act.type == ActionType.INPUT:
            return self._do_input(act)
        if act.type == ActionType.SELECT:
            return self._do_select(act)
        if act.type == ActionType.SWIPE:
            return self._do_swipe(act)
        raise ValueError(f"Unsupported action type: {act.type}")

    def _do_click(self, act: ClickAction) -> None:
        has_abs = act.x is not None and act.y is not None
        has_pct = act.x_pct is not None and act.y_pct is not None
        if has_abs or has_pct:
            if has_abs:
                x, y = int(act.x), int(act.y)
            else:
                size = self.adapter.get_window_size()
                x = int(size["width"] * float(act.x_pct))
                y = int(size["height"] * float(act.y_pct))
            self.adapter.tap(x, y)
            return

        if self.click_healer is None:
            raise RuntimeError("Semantic CLICK requires click_healer to be configured.")
        self._heal_and_tap(act, None)

    def _heal_and_tap(self, act: ClickAction, original_error: Exception | None) -> None:
        heal_selector = self._pick_heal_selector(act)
        if not heal_selector:
            if original_error is not None:
                raise original_error
            raise RuntimeError("Semantic CLICK missing selector/target/logical_name.")
        pack = self._build_pack()
        save_path = self._next_heal_save_path(heal_selector)
        try:
            result = self.click_healer.heal_click(
                pack=pack,
                selector=heal_selector,
                logical_name=self._str_or_none(getattr(act, "logical_name", "")),
                save_path=save_path,
                target_type=self._str_or_none(getattr(act, "target_type", "")) or "auto",
                text_candidates=list(getattr(act, "text_candidates", []) or []),
                region_hints=self._merge_region_hints(getattr(act, "region_hint", ""), getattr(act, "region_hints", [])),
            )
        except Exception:
            if original_error is not None:
                raise original_error
            raise
        if not result or not result.healed or result.chosen is None:
            if original_error is not None:
                raise original_error
            raise RuntimeError(f"Semantic CLICK unresolved: selector={heal_selector}")
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

        has_abs = act.x is not None and act.y is not None
        has_pct = act.x_pct is not None and act.y_pct is not None
        if has_abs or has_pct:
            x, y = self._xy_from_abs_or_pct(act.x, act.y, act.x_pct, act.y_pct)
        else:
            if self.click_healer is None:
                raise RuntimeError("Semantic INPUT requires click_healer to be configured.")
            selector = self._pick_input_selector(act)
            pack = self._build_pack()
            save_path = self._next_heal_save_path(selector or "input")
            result = self.click_healer.heal_input(
                pack=pack,
                selector=selector or act.logical_name or act.name or "input",
                logical_name=self._str_or_none(getattr(act, "logical_name", "")),
                save_path=save_path,
                text_candidates=list(getattr(act, "text_candidates", []) or []),
                region_hints=self._merge_region_hints(getattr(act, "region_hint", ""), getattr(act, "region_hints", [])),
            )
            if not result or not result.healed or result.chosen is None:
                raise RuntimeError(f"Semantic INPUT unresolved: selector={selector}")
            size = self.adapter.get_window_size()
            x = int(size["width"] * float(result.chosen.x_pct))
            y = int(size["height"] * float(result.chosen.y_pct))

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
        ox, oy = self._xy_from_abs_or_pct(act.open_x, act.open_y, act.open_x_pct, act.open_y_pct)
        self.adapter.tap(ox, oy)
        time.sleep(0.3)
        px, py = self._xy_from_abs_or_pct(act.option_x, act.option_y, act.option_x_pct, act.option_y_pct)
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

    def _pick_input_selector(self, act: InputAction) -> str | None:
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

    def _merge_region_hints(self, region_hint: str, region_hints: list[str]) -> list[str]:
        out: list[str] = []
        if str(region_hint or "").strip():
            out.append(str(region_hint).strip())
        for x in region_hints or []:
            s = str(x or "").strip()
            if s and s not in out:
                out.append(s)
        return out

    def _str_or_none(self, value: Any) -> str | None:
        s = str(value or "").strip()
        return s if s else None

    def _build_pack(self) -> Any:
        pack = self._pack_from_latest_step()
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
        return SimplePack(image_path="", ocr_text="", meta={"ocr_boxes": []})

    def _pack_from_latest_step(self) -> PerceptionPack | None:
        try:
            run_dir = Path(self.step_runner.evidence.run.run_dir)
            step_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
            if not step_dirs:
                return None
            step_dir = step_dirs[-1]
            ocr_path = step_dir / "ocr.txt"
            meta_path = step_dir / "perception.json"
            image_path = step_dir / "screenshot.png"
            if not image_path.exists():
                image_path = step_dir / "screenshot_after.png"
            text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"ocr_boxes": []}
            for name in ["page_source_after.xml", "page_source.xml", "page_source_fail.xml"]:
                p = step_dir / name
                if p.exists():
                    meta["page_source_path"] = str(p)
                    break
            meta["step_dir"] = str(step_dir)
            return PerceptionPack(image_path=str(image_path) if image_path.exists() else "", ocr_text=text, meta=meta)
        except Exception:
            return None

    def _next_heal_save_path(self, target_name: str) -> Path:
        self._heal_seq += 1
        root = Path("artifacts") / "heal"
        root.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w一-鿿\-]+", "_", str(target_name or "unknown"))
        safe_name = safe_name.strip("_") or "unknown"
        d = root / f"{self._heal_seq:03d}_{safe_name[:40]}"
        d.mkdir(parents=True, exist_ok=True)
        return d / "heal_candidates.json"

    def _archive_actions(self, plan: ActionPlan) -> str | None:
        try:
            run_dir = Path(self.step_runner.evidence.run.run_dir)
            p = run_dir / "actions.json"
            p.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            return str(p)
        except Exception:
            return None
