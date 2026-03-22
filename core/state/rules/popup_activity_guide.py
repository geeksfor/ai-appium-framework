# core/state/rules/popup_activity_guide.py
from __future__ import annotations
from core.state.rules.base import Rule, Region

class PopupActivityGuideRule(Rule):
    state_name = "Popup.Activity/Guide"
    region = Region(name="center")

    # ✅ 引导浮层强锚点：命中任意一个通常就很靠谱
    keywords_any = [
        "下一步",
        "上一步",
        "完成",
        "跳过",
        "点击跳过",
        "1/5", "2/5", "3/5", "4/5", "5/5",
        "新手引导",
        "指引",
        "活动时间",
    ]
    keywords_all = []

    # ✅ 排除：系统权限 / 隐私协议（避免抢状态）
    negative_keywords = [
        "允许", "不允许", "仅在使用期间", "始终允许", "权限", "授权",
        "隐私", "隐私保护指引", "服务协议", "会员服务协议", "同意", "拒绝",
    ]

    min_score = 0.6

    # ✅ 页面二次确认（建议启用）：至少满足一个 “引导特征组合”
    # 方式1：命中任意一个 confirm_any
    confirm_any = [
        "下一步",
        "点击跳过",
        "活动时间",
    ]
    # 方式2：keywords_any 至少命中 2 个（更稳）
    confirm_min_any_hits = 2