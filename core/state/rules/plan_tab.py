# core/state/rules/plan_tab.py
from __future__ import annotations
from core.state.rules.base import Rule, Region

class BloodSugarPlanTabRule(Rule):
    state_name = "BloodSugarReport.PlanTab"
    region = Region(name="full")

    # ✅ 强锚点：任取 2~3 个即可，越强越稳
    keywords_all = [
        "我的血糖报告",
        "测糖计划",
        "当周测糖计划",
    ]

    # ✅ 辅助：用于提高置信，但不强制
    keywords_any = [
        "测糖记录", "周报", "月报",
        "餐点说明", "我要调整",
        "测糖提示", "空腹", "早餐后", "睡前"
    ]

    confirm_any = ["测糖提示", "餐点说明", "我要调整"]

    # ✅ 排除弹窗类 / 非业务页
    negative_keywords = ["允许", "拒绝", "权限", "跳过", "知道了", "下一步"]

    # 这个 state 我建议阈值提高点，减少误判
    min_score = 0.8