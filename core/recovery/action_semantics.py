from __future__ import annotations

from typing import Dict, List

# 动作语义别名：用于把“具体文案”先归类成“动作类型”
ACTION_ALIASES: Dict[str, List[str]] = {
    "primary_submit": ["登录", "提交", "确认", "完成", "下一步", "继续"],
    "header_action": ["保存", "新增", "添加", "新建", "创建", "更多"],
    "dismiss_action": ["关闭", "取消", "跳过", "返回", "返回首页"],
    "row_action": ["编辑", "删除", "更新", "修改"],
    "input_field": ["用户名", "账号", "手机号", "密码", "验证码", "搜索"],
}

# 默认区域优先级：不是点击坐标，而是“先去哪片区域搜”
# default = 普通整屏页面
# dialog = 弹窗/浮层/center_dialog 场景
REGION_PRIORITY_BY_ACTION: Dict[str, Dict[str, List[str]]] = {
    "input_field": {
        "default": ["form_input_area", "center_dialog", "full"],
        "dialog": ["dialog_form_area", "center_dialog", "full"],
    },
    "primary_submit": {
        "default": ["bottom_primary_area", "center_dialog", "full"],
        "dialog": ["dialog_footer_primary_area", "center_dialog", "full"],
    },
    "header_action": {
        "default": ["header_action_area", "top_right", "full"],
        "dialog": ["dialog_footer_primary_area", "dialog_top_right_close_area", "full"],
    },
    "dismiss_action": {
        "default": ["top_right", "top_left", "center_dialog", "full"],
        "dialog": ["dialog_top_right_close_area", "dialog_footer_secondary_area", "center_dialog", "full"],
    },
    "row_action": {
        "default": ["list_row_action_area", "list_content_area", "full"],
        "dialog": ["dialog_body_area", "full"],
    },
}

DIALOG_REGION_PREFIXES = (
    "dialog_",
    "center_dialog",
    "bottom_sheet_area",
    "sheet_",
)


def infer_page_mode(explicit_regions: List[str]) -> str:
    """
    通过已知区域 hint 粗略判断当前是否更像弹窗/浮层模式。
    """
    for region in explicit_regions or []:
        name = str(region or "").strip()
        if not name:
            continue
        if name.startswith("dialog_") or name in {"center_dialog", "bottom_sheet_area"} or name.startswith("sheet_"):
            return "dialog"
    return "default"


def infer_action_semantic(alias_pool: List[str], target_role: str) -> str:
    """
    先把具体按钮/输入文案归类成动作语义，再由语义决定默认搜索区域。
    """
    if target_role == "input":
        return "input_field"

    merged = " ".join(str(x or "") for x in (alias_pool or []))

    for semantic, aliases in ACTION_ALIASES.items():
        if any(alias in merged for alias in aliases):
            return semantic

    return "default"


def default_regions_for_action(
    alias_pool: List[str],
    target_role: str,
    explicit_regions: List[str] | None = None,
) -> List[str]:
    """
    动作语义 -> 默认搜索区域优先级
    """
    page_mode = infer_page_mode(explicit_regions or [])
    semantic = infer_action_semantic(alias_pool=alias_pool, target_role=target_role)

    if semantic == "default":
        # 兜底策略：输入优先表单区；按钮优先中心再整屏
        return ["form_input_area", "full"] if target_role == "input" else ["center", "full"]

    by_mode = REGION_PRIORITY_BY_ACTION.get(semantic, {})
    return by_mode.get(page_mode) or by_mode.get("default") or (["form_input_area", "full"] if target_role == "input" else ["center", "full"])
