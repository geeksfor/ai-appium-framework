# core/state/rules/popup_permission.py
from __future__ import annotations
from core.state.rules.base import Rule, Region

class PopupPermissionRule(Rule):
    state_name = "Popup.Permission"
    region = Region(name="center")

    # ✅ 系统权限更稳定的词：允许/权限/仅在使用期间/始终允许/不允许
    # 注意：不要把“拒绝”当成单独的触发条件，否则会误伤隐私协议弹窗
    keywords_any = [
        "允许",
        "不允许",
        "仅在使用期间",
        "始终允许",
        "权限",
        "授权",
        "访问",
    ]
    keywords_all = []

    # ✅ 排除隐私协议/业务弹窗
    negative_keywords = [
        "隐私",
        "隐私保护指引",
        "服务协议",
        "会员服务协议",
        "小程序",
        "同意",   # 隐私协议弹窗强特征
    ]

    min_score = 0.6