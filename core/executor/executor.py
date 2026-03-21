# core/executor/executor.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from core.driver.appium_adapter import AppiumAdapter
from selenium.webdriver.common.keys import Keys
from core.executor.action_schema import (
    Action,
    ActionPlan,
    ActionType,
    ClickAction,
    WaitAction,
    BackAction,
    AssertAction,
    InputAction,
    SelectAction,
    SwipeAction,
    SwipeDirection,
)
from core.report.step_runner import StepRunner


@dataclass
class ExecResult:
    ok: bool
    executed: int
    last_action: dict | None = None


class Executor:
    def __init__(self, adapter: AppiumAdapter, step_runner: StepRunner):
        self.adapter = adapter
        self.step_runner = step_runner

    def run_plan(self, plan: ActionPlan) -> ExecResult:
        executed = 0
        last_action: dict | None = None

        for idx, act in enumerate(plan.actions, start=1):
            step_name = act.name or f"{idx:02d}_{act.type}"
            last_action = act.model_dump()

            # 每个动作都当成一个 step（证据链关键）
            self.step_runner.run(
                name=step_name,
                action=str(act.type),
                fn=lambda a=act: self._execute_one(a),
            )

            executed += 1

        return ExecResult(ok=True, executed=executed, last_action=last_action)

    def _execute_one(self, act: Action) -> Any:
        if act.type == ActionType.CLICK:
            return self._do_click(act)  # type: ignore[arg-type]
        if act.type == ActionType.WAIT:
            return self._do_wait(act)   # type: ignore[arg-type]
        if act.type == ActionType.BACK:
            return self._do_back(act)   # type: ignore[arg-type]
        if act.type == ActionType.ASSERT:
            return self._do_assert(act) # type: ignore[arg-type]
        if act.type == ActionType.INPUT:
          return self._do_input(act)   # type: ignore[arg-type]
        if act.type == ActionType.SELECT:
          return self._do_select(act)
        if act.type == ActionType.SWIPE:
          return self._do_swipe(act)
        raise ValueError(f"Unsupported action type: {act.type}")

    def _do_click(self, act: ClickAction) -> None:
        if act.x is not None and act.y is not None:
            x, y = act.x, act.y
        else:
            size = self.adapter.get_window_size()
            x = int(size["width"] * float(act.x_pct))
            y = int(size["height"] * float(act.y_pct))

        self.adapter.tap(x, y)

    def _do_wait(self, act: WaitAction) -> None:
        time.sleep(float(act.seconds))

    def _do_back(self, act: BackAction) -> None:
        self.adapter.back()

    def _do_assert(self, act: AssertAction) -> None:
        # Day3：最简单的断言：page_source contains
        # 注意：page_source 可能拿不到 -> 直接 fail，因为断言无法判断
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

      # 1) 点一下聚焦
      self.adapter.tap(x, y)
      time.sleep(0.2)

      # 2) 清空（best-effort）
      if act.clear_first:
        try:
          driver.switch_to.active_element.clear()
        except Exception:
          pass
      # 3) 输入（优先 active_element）
      try:
          driver.switch_to.active_element.send_keys(act.text)
      except Exception:
          # fallback：Appium mobile: type
          driver.execute_script("mobile: type", {"text": act.text})
      
      # 4) 回车（可选）
      if act.press_enter:
          try:
              driver.switch_to.active_element.send_keys(Keys.ENTER)
          except Exception:
              pass

    def _do_select(self, act: SelectAction) -> None:
        # open
        ox, oy = self._xy_from_abs_or_pct(act.open_x, act.open_y, act.open_x_pct, act.open_y_pct)
        self.adapter.tap(ox, oy)
        time.sleep(0.3)

        # option
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

        driver.execute_script("mobile: swipeGesture", {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "direction": act.direction.value.lower(),  # up/down/left/right
            "percent": float(act.percent),
            "speed": int(act.speed),
        })