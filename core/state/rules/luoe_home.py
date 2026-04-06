from __future__ import annotations

from core.state.rules.base import Region, Rule


class LuoeHomeRule(Rule):
    """
    罗e联首页：
    - 主锚点：底部导航 首页 + 我的
    - 辅助锚点：欢迎区 / 功能卡片 / 顶部文件导出
    - 排除：登录表单词，避免与登录页互相抢状态
    """

    state_name = "Luoe.Home"
    region = Region(name="full")

    keywords_all = ["首页", "我的"]
    keywords_any = [
        "欢迎您",
        "视频报修",
        "资质查询",
        "易问e答",
        "文件导出",
    ]
    negative_keywords = [
        "用户名",
        "请输入6~20位密码",
        "记住密码",
        "登录",
    ]

    min_score = 0.8

    confirm_any = [
        "欢迎您",
        "视频报修",
        "资质查询",
        "易问e答",
        "文件导出",
    ]
    confirm_min_any_hits = 1
    confirm_not_contains = [
        "用户名",
        "请输入6~20位密码",
        "记住密码",
    ]
