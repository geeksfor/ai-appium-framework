# core/state/rules/popup_onboarding_consent.py
from __future__ import annotations
from core.state.rules.base import Rule, Region

class PopupOnboardingConsentRule(Rule):
    """
    小程序高频首次弹窗：欢迎加入 + 勾选隐私/个人信息 + 已阅读并同意 + 立即前往/稍后了解
    """
    state_name = "Popup.OnboardingConsent"
    region = Region(name="center")

    # ✅ 强锚点（建议 all 用 2 个即可，稳且不误判）
    keywords_all = ["欢迎加入", "已阅读并同意"]

    # ✅ 辅助锚点（随便命中几个都会提高分）
    keywords_any = [
        "隐私政策", "个人信息",
        "立即前往", "稍后了解",
        "我已阅读并同意", "我同意", "已知晓",
    ]

    # ✅ 排除系统权限（避免抢）
    negative_keywords = [
        "仅在使用期间", "始终允许", "权限", "授权", "允许", "不允许"
    ]

    min_score = 0.8

    # ✅ 二次确认：命中任一“按钮锚点”就更稳（有就过）
    confirm_any = ["立即前往", "稍后了解", "隐私政策"]
    confirm_min_any_hits = 1