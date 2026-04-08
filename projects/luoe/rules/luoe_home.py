from __future__ import annotations

from core.state.rules.base import Region, Rule


class LuoeHomeRule(Rule):
    state_name = "Luoe.Home"
    region = Region(name="full")

    keywords_all = ["首页", "我的"]
    keywords_any = ["欢迎您", "视频报修", "资质查询", "易问e答", "文件导出", "16条未读消息"]
    negative_keywords = ["用户名", "请输入6~20位密码", "记住密码", "登录"]

    min_score = 0.82
    confirm_any = ["欢迎您", "视频报修", "资质查询", "易问e答"]
    confirm_min_any_hits = 1
