# core/executor/action_schema.py
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    CLICK = "CLICK"
    WAIT = "WAIT"
    BACK = "BACK"
    ASSERT = "ASSERT"
    INPUT = "INPUT"
    SELECT = "SELECT"
    SWIPE = "SWIPE"

class BaseAction(BaseModel):
    type: ActionType
    name: str = Field(default="", description="可读的动作名/步骤名，用于 evidence 目录命名")
    timeout_s: float = Field(default=10.0, ge=0, description="动作级超时(预留)")

    model_config = {
        "extra": "forbid",  # JSON 里多余字段直接报错，避免 silent bug
    }


class ClickAction(BaseAction):
    type: Literal[ActionType.CLICK] = ActionType.CLICK

    # 两种点击方式：
    # 1) 绝对坐标：x/y
    x: Optional[int] = Field(default=None, ge=0)
    y: Optional[int] = Field(default=None, ge=0)

    # 2) 百分比：x_pct/y_pct（0~1）
    x_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_xy(self):
        abs_ok = self.x is not None and self.y is not None
        pct_ok = self.x_pct is not None and self.y_pct is not None
        if abs_ok == pct_ok:
            # 要么都用绝对坐标，要么都用百分比，不能混用/都不填
            raise ValueError("CLICK requires either (x,y) OR (x_pct,y_pct).")
        return self


class WaitAction(BaseAction):
    type: Literal[ActionType.WAIT] = ActionType.WAIT
    seconds: float = Field(ge=0.0, le=120.0, description="等待秒数，上限先设 120 避免误填")


class BackAction(BaseAction):
    type: Literal[ActionType.BACK] = ActionType.BACK


class AssertAction(BaseAction):
    type: Literal[ActionType.ASSERT] = ActionType.ASSERT

    # Day3 先做最简单的断言：文本存在于 page_source
    contains_text: str = Field(min_length=1, description="断言 page_source 包含该文本")
    # 可选：是否忽略大小写
    ignore_case: bool = Field(default=True)

class InputAction(BaseAction):
    type: Literal[ActionType.INPUT] = ActionType.INPUT

    # 输入目标位置（和 CLICK 一样：abs 或 pct 二选一）
    x: Optional[int] = Field(default=None, ge=0)
    y: Optional[int] = Field(default=None, ge=0)
    x_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    text: str = Field(min_length=1)
    clear_first: bool = Field(default=True)
    press_enter: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_xy(self):
        abs_ok = self.x is not None and self.y is not None
        pct_ok = self.x_pct is not None and self.y_pct is not None
        if abs_ok == pct_ok:
            raise ValueError("INPUT requires either (x,y) OR (x_pct,y_pct).")
        return self


class SelectAction(BaseAction):
    """
    黑盒 SELECT：点开(open) -> 点选(option)
    """
    type: Literal[ActionType.SELECT] = ActionType.SELECT

    # open 坐标
    open_x: Optional[int] = Field(default=None, ge=0)
    open_y: Optional[int] = Field(default=None, ge=0)
    open_x_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    open_y_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # option 坐标
    option_x: Optional[int] = Field(default=None, ge=0)
    option_y: Optional[int] = Field(default=None, ge=0)
    option_x_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    option_y_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    option_label: str = Field(default="")

    @model_validator(mode="after")
    def validate_open_option(self):
        open_abs_ok = self.open_x is not None and self.open_y is not None
        open_pct_ok = self.open_x_pct is not None and self.open_y_pct is not None
        if open_abs_ok == open_pct_ok:
            raise ValueError("SELECT requires open_(x,y) OR open_(x_pct,y_pct).")

        opt_abs_ok = self.option_x is not None and self.option_y is not None
        opt_pct_ok = self.option_x_pct is not None and self.option_y_pct is not None
        if opt_abs_ok == opt_pct_ok:
            raise ValueError("SELECT requires option_(x,y) OR option_(x_pct,y_pct).")
        return self


class SwipeDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class SwipeAction(BaseAction):
    type: Literal[ActionType.SWIPE] = ActionType.SWIPE
    direction: SwipeDirection

    # swipe 区域（用 pct 更稳）
    left_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    top_pct: float = Field(default=0.2, ge=0.0, le=1.0)
    width_pct: float = Field(default=0.8, ge=0.0, le=1.0)
    height_pct: float = Field(default=0.6, ge=0.0, le=1.0)

    percent: float = Field(default=0.7, ge=0.0, le=1.0)
    speed: int = Field(default=900, ge=1, le=5000)

Action = Annotated[
    Union[ClickAction, WaitAction, BackAction, AssertAction, InputAction, SelectAction, SwipeAction],
    Field(discriminator="type")
]


class ActionPlan(BaseModel):
    actions: list[Action]

    model_config = {"extra": "forbid"}