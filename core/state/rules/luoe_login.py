from __future__ import annotations

from core.state.rules.base import Rule, Region


class LuoeLoginRule(Rule):
    state_name = "Luoe.Login"
    region = Region(name="full")

    keywords_any = [
        "罗e联",
        "用户名",
        "请输入6~20位密码",
        "记住密码",
        "登录",
    ]
    keywords_all = []
    negative_keywords = [
        "欢迎您",
        "视频报修",
        "资质查询",
        "首页",
        "我的",
    ]
    min_score = 0.6
    confirm_any = ["用户名", "登录", "请输入6~20位密码", "记住密码"]
    confirm_min_any_hits = 2
