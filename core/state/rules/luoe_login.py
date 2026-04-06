from __future__ import annotations

from core.state.rules.base import Region, Rule


class LuoeLoginRule(Rule):
    """
    罗e联登录页：
    - 主锚点：用户名 + 登录
    - 辅助锚点：密码占位 / 记住密码 / 罗e联品牌词
    - 排除：首页类词，避免把首页或其他表单页误识别为登录页
    """

    state_name = "Luoe.Login"
    region = Region(name="center")

    keywords_all = ["用户名", "登录"]
    keywords_any = [
        "请输入6~20位密码",
        "记住密码",
        "罗e联",
    ]
    negative_keywords = [
        "首页",
        "我的",
        "欢迎您",
        "视频报修",
        "资质查询",
        "文件导出",
    ]

    min_score = 0.8

    confirm_any = [
        "请输入6~20位密码",
        "记住密码",
        "罗e联",
    ]
    confirm_min_any_hits = 1
    confirm_not_contains = [
        "首页",
        "我的",
        "欢迎您",
    ]
