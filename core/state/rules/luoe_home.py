from __future__ import annotations

from core.state.rules.base import Rule, Region


class LuoeHomeRule(Rule):
    state_name = "Luoe.Home"
    region = Region(name="full")

    keywords_any = [
        "首页",
        "我的",
        "欢迎您",
        "视频报修",
        "资质查询",
        "易问e答",
        "文件导出",
    ]
    keywords_all = []
    negative_keywords = [
        "用户名",
        "请输入6~20位密码",
        "记住密码",
        "登录",
    ]
    min_score = 0.6
    confirm_any = ["首页", "我的", "欢迎您", "视频报修", "资质查询", "文件导出"]
    confirm_min_any_hits = 2
