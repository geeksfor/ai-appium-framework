from __future__ import annotations
from core.state.rules.base import Rule, Region

class PopupPrivacyConsentRule(Rule):
    state_name = "Popup.PrivacyConsent"
    region = Region(name="center")

    # 强特征：隐私类关键词 + 同意/拒绝按钮
    keywords_all = ["隐私", "同意"]
    keywords_any = [
        "隐私保护指引",
        "会员服务协议",
        "服务协议",
        "在使用当前小程序服务之前",
        "请仔细阅读",
        "拒绝",
        "开始使用",
    ]

    # 排除系统权限弹窗（避免误归类）
    negative_keywords = ["允许", "仅在使用期间", "始终允许", "权限"]

    min_score = 0.8

    # 页面二次确认（可选，但更稳）
    confirm_any = ["拒绝 同意", "隐私保护指引", "会员服务协议"]
    confirm_min_any_hits = 1