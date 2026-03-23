# core/policy/goal_schema.py
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union, List, Dict, Any

from pydantic import BaseModel, Field


class GoalIntent(str, Enum):
    CLOSE_OVERLAY = "CLOSE_OVERLAY"      # 关闭遮罩/弹窗，尽量不进入登录/授权
    CLICK_TEXT = "CLICK_TEXT"            # 点击某个可见文案（没有 bbox 时只能结合区域提示）
    REACH_STATE = "REACH_STATE"          # 到达某个 state（配 must_contain_text 更稳）
    WAIT_SAFE = "WAIT_SAFE"              # 不确定时安全等待/停止（用于节流或让页面稳定）


class GoalConstraints(BaseModel):
    """
    全局约束：减少 AI 乱点，降低维护成本（可长期稳定）
    """
    avoid_login: bool = True
    avoid_permission: bool = False
    prefer_close_over_enter: bool = True
    max_steps: int = Field(default=3, ge=1, le=10)

    # 如果你想强约束点击区域，可加：
    prefer_region: Optional[str] = Field(
        default=None,
        description="Optional: bottom|center|top|full etc."
    )

    model_config = {"extra": "forbid"}


class BaseGoal(BaseModel):
    intent: GoalIntent
    constraints: GoalConstraints = Field(default_factory=GoalConstraints)

    model_config = {"extra": "forbid"}


class CloseOverlayGoal(BaseGoal):
    intent: Literal[GoalIntent.CLOSE_OVERLAY] = GoalIntent.CLOSE_OVERLAY
    strategy: Literal["SAFE", "FAST"] = "SAFE"
    # SAFE: 优先关闭/X/遮罩/Back；FAST: 更激进（你后面可用）
    allow_click_positive: bool = False  # 是否允许点“同意/允许/登录”等正向按钮


class ClickTextGoal(BaseGoal):
    intent: Literal[GoalIntent.CLICK_TEXT] = GoalIntent.CLICK_TEXT
    target_text: str = Field(min_length=1)
    # 可选同义词，降低文案变动成本：用例给 key，框架映射后填 synonyms
    synonyms: List[str] = Field(default_factory=list)
    # 提示 AI：优先区域（比如底部 tab）
    ui_hints: List[str] = Field(default_factory=list)


class ReachStateGoal(BaseGoal):
    intent: Literal[GoalIntent.REACH_STATE] = GoalIntent.REACH_STATE
    target_state: str = Field(min_length=1)
    must_contain_text: List[str] = Field(default_factory=list)
    avoid_text: List[str] = Field(default_factory=list)


class WaitSafeGoal(BaseGoal):
    intent: Literal[GoalIntent.WAIT_SAFE] = GoalIntent.WAIT_SAFE
    seconds: float = Field(default=1.0, ge=0.0, le=10.0)
    reason: str = Field(default="stabilize")


Goal = Annotated[
    Union[CloseOverlayGoal, ClickTextGoal, ReachStateGoal, WaitSafeGoal],
    Field(discriminator="intent")
]


class GoalEnvelope(BaseModel):
    """
    你对外只传 GoalEnvelope，保证结构统一。
    """
    goal: Goal

    model_config = {"extra": "forbid"}

    @staticmethod
    def from_any(obj: Dict[str, Any]) -> "GoalEnvelope":
        # 允许你直接传 {"intent": "...", ...}，自动包一层
        if "goal" in obj:
            return GoalEnvelope.model_validate(obj)
        return GoalEnvelope.model_validate({"goal": obj})