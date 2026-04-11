from __future__ import annotations

from core.state.rules.base import Rule, Region


class BusyLoadingRule(Rule):
    """
    高优先级加载态：只要页面上仍出现“正在加载/请稍后/加载中”等前景阻塞信号，
    就不应该把页面判成 Home / Detail 这类“可操作完成态”。
    """

    state_name = "Busy.Loading"
    region = Region(name="center_dialog")

    # 允许命中其一即可进入候选，避免漏掉不同文案。
    keywords_any = [
        "正在加载",
        "请稍后",
        "加载中",
        "请耐心等待",
        "数据加载中",
        "拼命加载中",
        "处理中",
        "提交中",
    ]
    keywords_all = []
    negative_keywords = [
        "登录失败",
        "加载失败",
        "重试",
    ]
    min_score = 0.85

    # 至少需要一个强 loading 词，避免背景页误判。
    confirm_any = [
        "正在加载",
        "请稍后",
        "加载中",
        "请耐心等待",
        "处理中",
        "提交中",
    ]
    confirm_min_any_hits = 1
