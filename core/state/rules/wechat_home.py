# core/state/rules/wechat_home.py
from __future__ import annotations
from core.state.rules.base import Rule, Region

class WeChatHomeRule(Rule):
    state_name = "WeChat.Home"
    region = Region(name="bottom")  # 未来可用底部 ROI 更稳
    keywords_any = ["微信", "通讯录", "发现", "我"]
    keywords_all = []  # 不强制 all
    negative_keywords = ["小程序", "允许", "拒绝"]  # 弹窗/小程序入口常见词
    min_score = 0.6