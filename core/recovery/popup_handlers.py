# core/recovery/popup_handlers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from core.perception.perception import PerceptionPack
from core.executor.action_schema import ActionPlan


@dataclass
class PopupMatch:
    name: str
    score: float
    hits: List[str]
    reason: str


@dataclass
class PopupHandleResult:
    handled: bool
    handler: str
    match: Optional[PopupMatch]
    plan: Optional[ActionPlan]
    meta: Dict[str, Any]


def _norm(text: str) -> str:
    # 合并换行/多空格，提升包含匹配稳定性
    return " ".join((text or "").split())


def _contains_any(t: str, words: List[str]) -> List[str]:
    tl = t.lower()
    hits = []
    for w in words:
        if w and w.lower() in tl:
            hits.append(w)
    return hits


def _plan_click_pct(name: str, x_pct: float, y_pct: float) -> Dict[str, Any]:
    return {"type": "CLICK", "name": name, "x_pct": float(x_pct), "y_pct": float(y_pct)}


def _plan_back(name: str = "back") -> Dict[str, Any]:
    return {"type": "BACK", "name": name}


def _plan_wait(sec: float = 0.6, name: str = "wait") -> Dict[str, Any]:
    return {"type": "WAIT", "name": name, "seconds": float(sec)}


class PopupHandlers:
    """
    Day8 收官版弹窗库：
    - 只用 OCR 文本做匹配
    - 输出最小 ActionPlan（1~3 步）
    - 默认不做无限重试，保持收敛
    """

    def __init__(
        self,
        avoid_login: bool = True,
        allow_positive: bool = True,
        prefer_close: bool = True,
    ):
        """
        avoid_login: True 时尽量不点“登录/注册/去登录”
        allow_positive: True 时允许点“同意/允许/继续/下一步”（用于继续流程）
        prefer_close: True 时优先尝试关闭/跳过/稍后/我知道了
        """
        self.avoid_login = avoid_login
        self.allow_positive = allow_positive
        self.prefer_close = prefer_close

        # 关键词库（收官版：够用即可）
        self.kw_permission = ["仅在使用期间", "始终允许", "不允许", "允许", "权限", "授权", "访问"]
        self.kw_privacy = ["隐私", "隐私保护指引", "隐私政策", "服务协议", "会员服务协议", "已阅读并同意", "同意"]
        self.kw_guide = ["下一步", "上一步", "完成", "点击跳过", "跳过", "1/5", "2/5", "活动时间", "新手", "引导"]
        self.kw_close = ["关闭", "我知道了", "知道了", "稍后", "以后再说", "取消", "返回"]
        self.kw_positive = ["同意", "允许", "继续", "确定", "下一步", "立即前往", "开始使用"]
        self.kw_negative_risky = ["登录", "注册", "去登录", "绑定", "开通", "购买", "支付"]

    # ---- public ----
    def handle(self, pack: PerceptionPack) -> PopupHandleResult:
        t = _norm(pack.ocr_text)

        # 快速过滤：如果连常见弹窗词都没有，直接不处理
        quick_hits = _contains_any(t, self.kw_permission + self.kw_privacy + self.kw_guide + self.kw_close)
        if not quick_hits:
            return PopupHandleResult(
                handled=False,
                handler="none",
                match=None,
                plan=None,
                meta={"reason": "no popup keywords"},
            )

        # 1) risk gate：避免登录/付费诱导（收官版：只做避免点击）
        risky_hits = _contains_any(t, self.kw_negative_risky)

        # 2) match handlers（按优先级）
        m_perm = self._match_permission(t)
        m_priv = self._match_privacy(t)
        m_guide = self._match_guide(t)
        m_close = self._match_generic_close(t)

        # 选择得分最高的命中（但 permission/privacy/guide 互斥优先）
        candidates = [m for m in [m_perm, m_priv, m_guide, m_close] if m is not None]
        if not candidates:
            return PopupHandleResult(False, "none", None, None, {"reason": "no handler matched"})

        # 按 score 取 best（同分按优先级：permission > privacy > guide > close）
        def _prio(name: str) -> int:
            return {"permission": 0, "privacy": 1, "guide": 2, "close": 3}.get(name, 9)

        candidates.sort(key=lambda m: (-m.score, _prio(m.name)))
        best = candidates[0]

        # 3) build plan
        plan = self._build_plan(best, t, risky_hits)

        if plan is None:
            return PopupHandleResult(
                handled=False,
                handler=best.name,
                match=best,
                plan=None,
                meta={"reason": "no safe plan", "risky_hits": risky_hits},
            )

        return PopupHandleResult(
            handled=True,
            handler=best.name,
            match=best,
            plan=plan,
            meta={"risky_hits": risky_hits},
        )

    # ---- matchers ----
    def _match_permission(self, t: str) -> Optional[PopupMatch]:
        hits = _contains_any(t, self.kw_permission)
        # permission 必须命中强特征（仅在使用期间/始终允许/权限/允许）
        strong = _contains_any(t, ["仅在使用期间", "始终允许", "权限"])
        if not hits and not strong:
            return None
        score = 0.6 + 0.1 * len(strong) + 0.03 * len(hits)
        return PopupMatch("permission", min(score, 0.99), hits, "system permission keywords")

    def _match_privacy(self, t: str) -> Optional[PopupMatch]:
        hits = _contains_any(t, self.kw_privacy)
        # privacy 必须至少命中 “隐私/政策/协议/已阅读并同意/同意”之一
        core = _contains_any(t, ["隐私", "隐私政策", "服务协议", "已阅读并同意"])
        if not hits or not core:
            return None
        score = 0.65 + 0.08 * len(core) + 0.02 * len(hits)
        return PopupMatch("privacy", min(score, 0.99), hits, "privacy consent keywords")

    def _match_guide(self, t: str) -> Optional[PopupMatch]:
        hits = _contains_any(t, self.kw_guide)
        # guide 至少命中 “下一步/跳过/1/5/活动时间”之一
        core = _contains_any(t, ["下一步", "跳过", "点击跳过", "1/5", "活动时间"])
        if not hits or not core:
            return None
        score = 0.62 + 0.08 * len(core) + 0.02 * len(hits)
        return PopupMatch("guide", min(score, 0.99), hits, "guide/activity keywords")

    def _match_generic_close(self, t: str) -> Optional[PopupMatch]:
        hits = _contains_any(t, self.kw_close)
        if not hits:
            return None
        score = 0.6 + 0.05 * len(hits)
        return PopupMatch("close", min(score, 0.95), hits, "generic close keywords")

    # ---- plan builder ----
    def _build_plan(self, m: PopupMatch, t: str, risky_hits: List[str]) -> Optional[ActionPlan]:
        """
        收官版：只输出 1~3 步的模板坐标
        - 右上角关闭： (0.92, 0.18)
        - 底部右按钮： (0.78, 0.84)  （同意/允许/下一步常在右侧）
        - 底部左按钮： (0.22, 0.84)  （拒绝/取消常在左侧）
        - 遮罩空白：   (0.50, 0.10)
        """
        actions: List[Dict[str, Any]] = []

        # 1) 如果 prefer_close，优先尝试关闭型模板（对 guide/close 特别有效）
        if self.prefer_close and m.name in ("guide", "close"):
            actions.append(_plan_click_pct("tap_top_right_close", 0.92, 0.18))
            actions.append(_plan_wait(0.4))
            # 遮罩空白
            actions.append(_plan_click_pct("tap_mask", 0.50, 0.10))
            return ActionPlan.model_validate({"actions": actions[:3]})

        # 2) Permission/Privacy：允许/同意通常在底部右侧按钮
        if m.name in ("permission", "privacy"):
            if not self.allow_positive:
                # 不允许点正向按钮，只能尝试关闭或 back
                actions.append(_plan_click_pct("tap_top_right_close", 0.92, 0.18))
                actions.append(_plan_wait(0.4))
                actions.append(_plan_back("back_close_popup"))
                return ActionPlan.model_validate({"actions": actions[:3]})

            # 风险词存在时，尽量不要点“登录/购买”等，仍只点底部右按钮（多为同意/允许）
            actions.append(_plan_click_pct("tap_bottom_right_positive", 0.78, 0.84))
            actions.append(_plan_wait(0.4))
            return ActionPlan.model_validate({"actions": actions[:2]})

        # 3) Guide：下一步/跳过 — 默认点右下（下一步）或右上 close
        if m.name == "guide":
            actions.append(_plan_click_pct("tap_bottom_right_next", 0.78, 0.84))
            actions.append(_plan_wait(0.4))
            return ActionPlan.model_validate({"actions": actions[:2]})

        # 4) Generic close：优先右上角，再 back
        if m.name == "close":
            actions.append(_plan_click_pct("tap_top_right_close", 0.92, 0.18))
            actions.append(_plan_wait(0.4))
            actions.append(_plan_back("back_close_popup"))
            return ActionPlan.model_validate({"actions": actions[:3]})

        # fallback：不处理
        return None