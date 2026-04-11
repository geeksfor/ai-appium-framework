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
    name: str = Field(default="", description="可读步骤名")
    timeout_s: float = Field(default=10.0, ge=0)

    model_config = {"extra": "forbid"}


class ClickAction(BaseAction):
    type: Literal[ActionType.CLICK] = ActionType.CLICK

    x: Optional[int] = Field(default=None, ge=0)
    y: Optional[int] = Field(default=None, ge=0)
    x_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    selector: str = Field(default="")
    target: str = Field(default="")
    target_text: str = Field(default="")
    logical_name: str = Field(default="")
    target_type: str = Field(default="auto", description="auto/text/button/icon/input")
    text_candidates: list[str] = Field(default_factory=list)
    region_hint: str = Field(default="")
    region_hints: list[str] = Field(default_factory=list)
    allow_heal: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_xy(self):
        abs_ok = self.x is not None and self.y is not None
        pct_ok = self.x_pct is not None and self.y_pct is not None
        semantic_ok = any([
            self.selector.strip(),
            self.target.strip(),
            self.target_text.strip(),
            self.logical_name.strip(),
            any(str(x).strip() for x in self.text_candidates),
        ])
        if abs_ok and pct_ok:
            raise ValueError("CLICK cannot provide both absolute and percentage coordinates.")
        if not abs_ok and not pct_ok and not semantic_ok:
            raise ValueError("CLICK requires either coords or semantic fields.")
        return self


class WaitAction(BaseAction):
    type: Literal[ActionType.WAIT] = ActionType.WAIT
    seconds: float = Field(ge=0.0, le=120.0)


class BackAction(BaseAction):
    type: Literal[ActionType.BACK] = ActionType.BACK


class AssertAction(BaseAction):
    type: Literal[ActionType.ASSERT] = ActionType.ASSERT
    contains_text: str = Field(min_length=1)
    ignore_case: bool = Field(default=True)


class InputAction(BaseAction):
    type: Literal[ActionType.INPUT] = ActionType.INPUT

    x: Optional[int] = Field(default=None, ge=0)
    y: Optional[int] = Field(default=None, ge=0)
    x_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    y_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    text: str = Field(min_length=1)
    clear_first: bool = Field(default=True)
    press_enter: bool = Field(default=False)

    selector: str = Field(default="")
    target: str = Field(default="")
    target_text: str = Field(default="")
    logical_name: str = Field(default="")
    target_type: str = Field(default="input", description="input")
    text_candidates: list[str] = Field(default_factory=list)
    region_hint: str = Field(default="")
    region_hints: list[str] = Field(default_factory=list)
    allow_heal: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_xy(self):
        abs_ok = self.x is not None and self.y is not None
        pct_ok = self.x_pct is not None and self.y_pct is not None
        semantic_ok = any([
            self.selector.strip(),
            self.target.strip(),
            self.target_text.strip(),
            self.logical_name.strip(),
            any(str(x).strip() for x in self.text_candidates),
        ])
        if abs_ok and pct_ok:
            raise ValueError("INPUT cannot provide both absolute and percentage coordinates.")
        if not abs_ok and not pct_ok and not semantic_ok:
            raise ValueError("INPUT requires either coords or semantic fields.")
        return self


class SelectAction(BaseAction):
    type: Literal[ActionType.SELECT] = ActionType.SELECT
    open_x: Optional[int] = Field(default=None, ge=0)
    open_y: Optional[int] = Field(default=None, ge=0)
    open_x_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    open_y_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
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
    left_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    top_pct: float = Field(default=0.2, ge=0.0, le=1.0)
    width_pct: float = Field(default=0.8, ge=0.0, le=1.0)
    height_pct: float = Field(default=0.6, ge=0.0, le=1.0)
    percent: float = Field(default=0.7, ge=0.0, le=1.0)
    speed: int = Field(default=900, ge=1, le=5000)


Action = Annotated[
    Union[
        ClickAction,
        WaitAction,
        BackAction,
        AssertAction,
        InputAction,
        SelectAction,
        SwipeAction,
    ],
    Field(discriminator="type"),
]


class ActionPlan(BaseModel):
    actions: list[Action]
    model_config = {"extra": "forbid"}
