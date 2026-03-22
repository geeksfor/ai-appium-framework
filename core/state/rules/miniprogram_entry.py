# core/state/rules/miniprogram_entry.py
from __future__ import annotations
from core.state.rules.base import Rule, Region

class MiniProgramEntryRule(Rule):
    state_name = "MiniProgram.Entry"
    region = Region(name="full")
    keywords_any = ["小程序", "服务", "更多", "关闭"]
    keywords_all = []
    negative_keywords = ["允许", "拒绝"]  # 权限弹窗优先级更高
    min_score = 0.6