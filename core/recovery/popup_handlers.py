from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.executor.action_schema import ActionPlan
from core.perception.perception import PerceptionPack
from core.policy.policy_runner import PolicyRunner
from core.recovery.click_resolver import ClickResolver


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
    return " ".join((text or "").split())


def _contains_any(t: str, words: List[str]) -> List[str]:
    tl = t.lower()
    hits = []
    for w in words:
        if w and w.lower() in tl:
            hits.append(w)
    return hits


def _plan_click_pct(name: str, x_pct: float, y_pct: float) -> Dict[str, Any]:
    return {
        "type": "CLICK",
        "name": name,
        "x_pct": float(x_pct),
        "y_pct": float(y_pct),
        "allow_heal": True,
    }


def _plan_back(name: str = "back") -> Dict[str, Any]:
    return {"type": "BACK", "name": name}


def _plan_wait(sec: float = 0.6, name: str = "wait") -> Dict[str, Any]:
    return {"type": "WAIT", "name": name, "seconds": float(sec)}


class PopupHandlers:
    """
    弹窗规则处理器（今天的收敛修正版）：
    - OCR 关键词判断弹窗类型
    - 统一通过 ClickResolver 解析点击点（bbox 优先，AI 兜底）
    - 命中业务风险词时，优先保守关闭/返回，避免误点“登录/开通/支付”
    """

    def __init__(
        self,
        avoid_login: bool = True,
        allow_positive: bool = True,
        prefer_close: bool = True,
        click_resolver: ClickResolver | None = None,
        policy_runner: Optional[PolicyRunner] = None,
    ):
        self.avoid_login = avoid_login
        self.allow_positive = allow_positive
        self.prefer_close = prefer_close
        self.click_resolver = click_resolver or ClickResolver(policy_runner=policy_runner)
        self._last_pack: Optional[PerceptionPack] = None

        self.kw_permission = ["仅在使用期间", "始终允许", "不允许", "允许", "权限", "授权", "访问"]
        self.kw_privacy = ["隐私", "隐私保护指引", "隐私政策", "服务协议", "会员服务协议", "已阅读并同意", "同意"]
        self.kw_guide = ["下一步", "上一步", "完成", "点击跳过", "跳过", "1/5", "2/5", "活动时间", "新手", "引导"]
        self.kw_close = ["关闭", "我知道了", "知道了", "稍后", "以后再说", "取消", "返回"]
        self.kw_positive = ["同意", "允许", "继续", "确定", "下一步", "立即前往", "开始使用"]
        self.kw_negative_risky = [
            "登录",
            "注册",
            "去登录",
            "绑定",
            "开通",
            "购买",
            "支付",
            "认证",
            "实名",
            "续费",
            "升级",
            "办理",
        ]

    def handle(self, pack: PerceptionPack) -> PopupHandleResult:
        self._last_pack = pack
        t = _norm(pack.ocr_text)

        quick_hits = _contains_any(t, self.kw_permission + self.kw_privacy + self.kw_guide + self.kw_close)
        if not quick_hits:
            return PopupHandleResult(
                handled=False,
                handler="none",
                match=None,
                plan=None,
                meta={"reason": "no popup keywords"},
            )

        risky_hits = _contains_any(t, self.kw_negative_risky)

        m_perm = self._match_permission(t)
        m_priv = self._match_privacy(t)
        m_guide = self._match_guide(t)
        m_close = self._match_generic_close(t)

        candidates = [m for m in [m_perm, m_priv, m_guide, m_close] if m is not None]
        if not candidates:
            return PopupHandleResult(False, "none", None, None, {"reason": "no handler matched"})

        def _prio(name: str) -> int:
            return {"permission": 0, "privacy": 1, "guide": 2, "close": 3}.get(name, 9)

        candidates.sort(key=lambda m: (-m.score, _prio(m.name)))
        best = candidates[0]
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

    def _match_permission(self, t: str) -> Optional[PopupMatch]:
        hits = _contains_any(t, self.kw_permission)
        strong = _contains_any(t, ["仅在使用期间", "始终允许", "权限"])
        if not hits and not strong:
            return None
        score = 0.6 + 0.1 * len(strong) + 0.03 * len(hits)
        return PopupMatch("permission", min(score, 0.99), hits, "system permission keywords")

    def _match_privacy(self, t: str) -> Optional[PopupMatch]:
        hits = _contains_any(t, self.kw_privacy)
        core = _contains_any(t, ["隐私", "隐私政策", "服务协议", "已阅读并同意"])
        if not hits or not core:
            return None
        score = 0.65 + 0.08 * len(core) + 0.02 * len(hits)
        return PopupMatch("privacy", min(score, 0.99), hits, "privacy consent keywords")

    def _match_guide(self, t: str) -> Optional[PopupMatch]:
        hits = _contains_any(t, self.kw_guide)
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

    def _build_plan(self, m: PopupMatch, t: str, risky_hits: List[str]) -> Optional[ActionPlan]:
        actions: List[Dict[str, Any]] = []
        pack = self._last_pack
        risky_context = self.avoid_login and bool(risky_hits)

        def click_text_or_template(
            targets: List[str],
            template: Dict[str, Any],
            ui_hints: Optional[List[str]] = None,
            logical_name: str = "",
        ) -> Dict[str, Any]:
            if pack is not None:
                cp = self.click_resolver.resolve(pack, targets, ui_hints=ui_hints)
                if cp:
                    primary = targets[0] if targets else ""
                    return {
                        "type": "CLICK",
                        "name": f"tap_{primary}_{cp.source}",
                        "x_pct": cp.x_pct,
                        "y_pct": cp.y_pct,
                        "target": primary,
                        "target_text": primary,
                        "selector": f"text={primary}" if primary else "",
                        "logical_name": logical_name,
                        "allow_heal": True,
                    }
            return template

        top_right_close = {
            **_plan_click_pct("tap_top_right_close_tpl", 0.92, 0.18),
            "target": "关闭",
            "target_text": "关闭",
            "selector": "text=关闭",
            "logical_name": "close",
        }
        bottom_right = {
            **_plan_click_pct("tap_bottom_right_tpl", 0.78, 0.84),
            "target": "确定",
            "target_text": "确定",
            "selector": "text=确定",
            "logical_name": "agree",
        }
        mask = {
            **_plan_click_pct("tap_mask_tpl", 0.50, 0.10),
            "allow_heal": False,
        }

        if self.prefer_close and m.name in ("guide", "close"):
            act1 = click_text_or_template(
                targets=["关闭", "跳过", "点击跳过", "我知道了", "知道了"],
                template=top_right_close,
                ui_hints=["优先右上角关闭或跳过按钮"],
                logical_name="close",
            )
            actions += [act1, _plan_wait(0.4), mask]
            return ActionPlan.model_validate({"actions": actions[:3]})

        if m.name in ("permission", "privacy"):
            if risky_context:
                actions += [top_right_close, _plan_wait(0.4), _plan_back("back_risky_popup")]
                return ActionPlan.model_validate({"actions": actions[:3]})

            if not self.allow_positive:
                actions += [top_right_close, _plan_wait(0.4), _plan_back("back_close_popup")]
                return ActionPlan.model_validate({"actions": actions[:3]})

            act = click_text_or_template(
                targets=["同意", "允许", "仅在使用期间", "始终允许", "确定", "继续", "开始使用"],
                template=bottom_right,
                ui_hints=["优先底部右侧主要按钮（同意/允许/确定）"],
                logical_name="agree",
            )
            actions += [act, _plan_wait(0.4)]
            return ActionPlan.model_validate({"actions": actions[:2]})

        if m.name == "guide":
            act = click_text_or_template(
                targets=["下一步", "完成", "跳过", "点击跳过"],
                template=bottom_right,
                ui_hints=["引导弹窗优先下一步/跳过"],
                logical_name="next",
            )
            actions += [act, _plan_wait(0.4)]
            return ActionPlan.model_validate({"actions": actions[:2]})

        if m.name == "close":
            actions += [top_right_close, _plan_wait(0.4), _plan_back("back_close_popup")]
            return ActionPlan.model_validate({"actions": actions[:3]})

        return None
