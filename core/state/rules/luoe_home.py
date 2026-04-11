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
        "扫一扫",
        "消息",
    ]
    keywords_all = []
    negative_keywords = [
        "用户名",
        "请输入6~20位密码",
        "记住密码",
        "登录",
        # 关键：加载态出现时，不要把背景页提前判成 Home
        "正在加载",
        "请稍后",
        "加载中",
        "处理中",
        "提交中",
    ]
    min_score = 0.6

    # 背景页露出时只要命中 1 个首页词太容易误判。
    # 提高到至少 2 个首页信号，且不含 loading 负词。
    confirm_any = ["首页", "我的", "欢迎您", "视频报修", "资质查询", "文件导出", "易问e答"]
    confirm_min_any_hits = 2
    confirm_not_contains = ["正在加载", "请稍后", "加载中", "处理中", "提交中"]
