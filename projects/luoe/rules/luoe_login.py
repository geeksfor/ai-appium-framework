from __future__ import annotations

from core.state.rules.base import Region, Rule


class LuoeLoginRule(Rule):
    state_name = "Luoe.Login"
    region = Region(name="full")

    keywords_all = ["用户名", "登录"]
    keywords_any = ["罗e联", "请输入6~20位密码", "记住密码", "关爱如e", "服务智联"]
    negative_keywords = ["欢迎您", "首页", "我的", "视频报修", "资质查询"]

    min_score = 0.82
    confirm_any = ["请输入6~20位密码", "记住密码", "罗e联"]
    confirm_min_any_hits = 1
