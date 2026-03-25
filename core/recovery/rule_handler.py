# core/recovery/rule_handler.py
from __future__ import annotations

from typing import Optional

from core.executor.action_schema import ActionPlan
from core.perception.perception import PerceptionPack
from core.recovery.popup_handlers import PopupHandlers


class PopupRuleHandler:
    """
    将 Day8 PopupHandlers 适配为 Day7 Runner 的 rule_handler：
    - 输入：(state_res, pack)
    - 输出：ActionPlan 或 None
    """

    def __init__(
        self,
        avoid_login: bool = True,
        allow_positive: bool = True,
        prefer_close: bool = True,
        only_when_popup_state: bool = False,
    ):
        """
        only_when_popup_state:
        - False（推荐收官版）：不依赖 state，看到弹窗关键词就处理（更稳）
        - True：只有 state_res.state 以 'Popup.' 开头才处理（更谨慎）
        """
        self.handlers = PopupHandlers(
            avoid_login=avoid_login,
            allow_positive=allow_positive,
            prefer_close=prefer_close,
        )
        self.only_when_popup_state = only_when_popup_state

    def __call__(self, state_res, pack: PerceptionPack) -> Optional[ActionPlan]:
        # 可选：只在 StateMachine 判定是弹窗时才处理
        if self.only_when_popup_state:
            s = getattr(state_res, "state", "") or ""
            if not s.startswith("Popup."):
                return None

        res = self.handlers.handle(pack)
        if res.handled and res.plan is not None:
            return res.plan
        return None