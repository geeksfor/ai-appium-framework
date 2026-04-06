from __future__ import annotations

from typing import Optional

from core.executor.action_schema import ActionPlan
from core.perception.perception import PerceptionPack
from core.policy.policy_runner import PolicyRunner
from core.recovery.click_resolver import ClickResolver
from core.recovery.popup_handlers import PopupHandlers


class PopupRuleHandler:
    """
    将 PopupHandlers 适配为 Runner 的 rule_handler：
    - 输入：(state_res, pack)
    - 输出：ActionPlan 或 None

    收口版增强：
    - 支持直接传 policy_runner，用于 ClickResolver 的 AI 兜底
    - 也支持直接传 click_resolver，便于外部完全接管解析策略
    """

    def __init__(
        self,
        avoid_login: bool = True,
        allow_positive: bool = True,
        prefer_close: bool = True,
        only_when_popup_state: bool = False,
        policy_runner: Optional[PolicyRunner] = None,
        click_resolver: Optional[ClickResolver] = None,
    ):
        self.handlers = PopupHandlers(
            avoid_login=avoid_login,
            allow_positive=allow_positive,
            prefer_close=prefer_close,
            click_resolver=click_resolver,
            policy_runner=policy_runner,
        )
        self.only_when_popup_state = only_when_popup_state

    def __call__(self, state_res, pack: PerceptionPack) -> Optional[ActionPlan]:
        if self.only_when_popup_state:
            state = getattr(state_res, "state", "") or ""
            if not str(state).startswith("Popup."):
                return None

        result = self.handlers.handle(pack)
        if not result.handled or result.plan is None:
            return None
        return result.plan
