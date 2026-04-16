from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PctBox = Tuple[float, float, float, float]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass(frozen=True)
class RegionSpec:
    name: str
    box: PctBox
    desc: str = ""
    scope: str = "screen"  # screen | dialog

    def normalized(self) -> PctBox:
        x1, y1, x2, y2 = self.box
        x1 = _clamp01(x1)
        y1 = _clamp01(y1)
        x2 = _clamp01(x2)
        y2 = _clamp01(y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid region box for {self.name}: {self.box}")
        return x1, y1, x2, y2


# -------------------------
# Screen-level regions
# -------------------------
REGIONS: Dict[str, RegionSpec] = {
    "full": RegionSpec("full", (0.0, 0.0, 1.0, 1.0), "整屏区域"),
    "top_left": RegionSpec("top_left", (0.0, 0.0, 0.32, 0.20), "左上角：返回、抽屉入口、关闭返回"),
    "top_right": RegionSpec("top_right", (0.68, 0.0, 1.0, 0.20), "右上角：保存、新增、更多、关闭"),
    "header_action_area": RegionSpec("header_action_area", (0.60, 0.0, 1.0, 0.22), "页头操作区：新增/保存/更多"),
    "center": RegionSpec("center", (0.15, 0.18, 0.85, 0.82), "中间主体区域"),
    "center_dialog": RegionSpec("center_dialog", (0.12, 0.18, 0.88, 0.82), "中心弹窗大致区域"),
    "form_input_area": RegionSpec("form_input_area", (0.14, 0.28, 0.86, 0.82), "普通页面表单输入区域"),
    "bottom_primary_area": RegionSpec("bottom_primary_area", (0.18, 0.72, 0.82, 0.96), "普通页面底部主按钮区域"),
    "bottom_sheet_area": RegionSpec("bottom_sheet_area", (0.0, 0.60, 1.0, 1.0), "底部弹层 / BottomSheet 大致区域"),
    "list_content_area": RegionSpec("list_content_area", (0.02, 0.16, 0.98, 0.92), "列表主体区域"),
    "list_row_action_area": RegionSpec("list_row_action_area", (0.74, 0.18, 1.0, 0.92), "列表行右侧操作区：编辑/删除/更新"),
    "checkbox_area": RegionSpec("checkbox_area", (0.42, 0.68, 0.78, 0.90), "勾选框、记住密码、协议区域"),
}

# -------------------------
# Dialog-relative regions
# relative to dialog container bounds
# -------------------------
DIALOG_REGIONS: Dict[str, RegionSpec] = {
    "dialog_full": RegionSpec("dialog_full", (0.0, 0.0, 1.0, 1.0), "整个弹窗容器", scope="dialog"),
    "dialog_top_left": RegionSpec("dialog_top_left", (0.0, 0.0, 0.32, 0.22), "弹窗左上角", scope="dialog"),
    "dialog_top_right_close_area": RegionSpec("dialog_top_right_close_area", (0.70, 0.0, 1.0, 0.22), "弹窗右上角关闭/叉号/关闭", scope="dialog"),
    "dialog_body_area": RegionSpec("dialog_body_area", (0.06, 0.14, 0.94, 0.78), "弹窗主体内容区", scope="dialog"),
    "dialog_form_area": RegionSpec("dialog_form_area", (0.08, 0.18, 0.92, 0.76), "弹窗表单输入区", scope="dialog"),
    "dialog_footer_primary_area": RegionSpec("dialog_footer_primary_area", (0.45, 0.76, 0.96, 1.0), "弹窗底部主按钮区：保存/提交/确认", scope="dialog"),
    "dialog_footer_secondary_area": RegionSpec("dialog_footer_secondary_area", (0.04, 0.76, 0.56, 1.0), "弹窗底部次按钮区：取消/关闭/返回", scope="dialog"),
    "dialog_footer_full_area": RegionSpec("dialog_footer_full_area", (0.0, 0.74, 1.0, 1.0), "弹窗底部全按钮区", scope="dialog"),
}

ROLE_DEFAULT_REGION_PRIORITY: Dict[str, List[str]] = {
    "button": ["bottom_primary_area", "header_action_area", "center_dialog", "center"],
    "input": ["form_input_area", "center", "list_content_area"],
    "dialog_button": ["dialog_footer_primary_area", "dialog_top_right_close_area", "dialog_footer_secondary_area"],
    "dialog_input": ["dialog_form_area", "dialog_body_area"],
    "checkbox": ["checkbox_area", "dialog_body_area", "center_dialog"],
    "icon_close": ["top_right", "dialog_top_right_close_area", "center_dialog"],
    "icon_back": ["top_left", "dialog_top_left"],
    "row_action": ["list_row_action_area", "list_content_area"],
}

# 动作语义 -> 默认搜索区域优先级（弱先验，不是硬编码坐标）
TARGET_DEFAULT_REGION_PRIORITY: Dict[str, List[str]] = {
    # primary CTA
    "登录": ["bottom_primary_area", "dialog_footer_primary_area", "center_dialog"],
    "立即登录": ["bottom_primary_area", "dialog_footer_primary_area"],
    "去登录": ["bottom_primary_area", "dialog_footer_primary_area"],
    "提交": ["bottom_primary_area", "dialog_footer_primary_area", "center_dialog"],
    "下一步": ["bottom_primary_area", "dialog_footer_primary_area", "center_dialog"],
    "确定": ["dialog_footer_primary_area", "bottom_primary_area", "center_dialog"],
    "确认": ["dialog_footer_primary_area", "bottom_primary_area", "center_dialog"],
    "完成": ["header_action_area", "dialog_footer_primary_area", "bottom_primary_area"],
    "保存": ["header_action_area", "dialog_footer_primary_area", "bottom_primary_area"],
    "新增": ["header_action_area", "top_right", "dialog_footer_primary_area"],
    "发布": ["header_action_area", "dialog_footer_primary_area", "bottom_primary_area"],
    # close / nav
    "关闭": ["dialog_top_right_close_area", "top_right", "dialog_footer_secondary_area", "center_dialog"],
    "取消": ["dialog_footer_secondary_area", "dialog_top_right_close_area", "center_dialog"],
    "跳过": ["top_right", "dialog_top_right_close_area"],
    "返回": ["top_left", "dialog_top_left"],
    # row actions
    "编辑": ["list_row_action_area", "dialog_footer_primary_area", "header_action_area"],
    "删除": ["list_row_action_area", "dialog_footer_secondary_area", "center_dialog"],
    "更新": ["list_row_action_area", "dialog_footer_primary_area", "header_action_area"],
    "修改": ["list_row_action_area", "dialog_footer_primary_area", "header_action_area"],
    # inputs
    "用户名": ["form_input_area", "dialog_form_area"],
    "账号": ["form_input_area", "dialog_form_area"],
    "手机号": ["form_input_area", "dialog_form_area"],
    "密码": ["form_input_area", "dialog_form_area"],
    "验证码": ["form_input_area", "dialog_form_area"],
    "搜索": ["top_left", "header_action_area", "dialog_form_area"],
}

DIALOG_TRIGGER_TEXTS: List[str] = [
    "新增", "编辑", "保存", "取消", "关闭", "请选择", "确定", "确认", "弹窗", "提示", "新增页面", "编辑页面",
]


def list_region_names(include_dialog: bool = True) -> List[str]:
    names = list(REGIONS.keys())
    if include_dialog:
        names.extend(DIALOG_REGIONS.keys())
    return names


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def get_region_spec(name: str) -> RegionSpec:
    if name in REGIONS:
        return REGIONS[name]
    if name in DIALOG_REGIONS:
        return DIALOG_REGIONS[name]
    raise KeyError(name)


def get_region_box(name: str) -> PctBox:
    return get_region_spec(name).normalized()


def is_dialog_region(name: str) -> bool:
    try:
        return get_region_spec(name).scope == "dialog"
    except KeyError:
        return False


def project_region_box(region_name: str, container_box: Optional[PctBox]) -> PctBox:
    spec = get_region_spec(region_name)
    x1, y1, x2, y2 = spec.normalized()
    if spec.scope != "dialog" or container_box is None:
        return x1, y1, x2, y2
    cx1, cy1, cx2, cy2 = container_box
    cw = max(0.0, cx2 - cx1)
    ch = max(0.0, cy2 - cy1)
    if cw <= 0 or ch <= 0:
        return x1, y1, x2, y2
    return (
        _clamp01(cx1 + x1 * cw),
        _clamp01(cy1 + y1 * ch),
        _clamp01(cx1 + x2 * cw),
        _clamp01(cy1 + y2 * ch),
    )


def infer_page_mode(explicit_page_mode: Optional[str] = None, target_role: Optional[str] = None, region_hints: Optional[Sequence[str]] = None, raw_target: Optional[str] = None) -> str:
    mode = str(explicit_page_mode or "").strip().lower()
    if mode in {"dialog", "sheet", "page"}:
        return mode
    for name in region_hints or []:
        if is_dialog_region(name):
            return "dialog"
    text = str(raw_target or "")
    if any(t in text for t in DIALOG_TRIGGER_TEXTS):
        return "dialog"
    role = str(target_role or "").strip().lower()
    if role.startswith("dialog_"):
        return "dialog"
    return "page"


def default_regions_for_target(raw_target: Optional[str], target_role: Optional[str], page_mode: str = "page") -> List[str]:
    out: List[str] = []
    if raw_target:
        out.extend(TARGET_DEFAULT_REGION_PRIORITY.get(str(raw_target).strip(), []))
    if target_role:
        out.extend(ROLE_DEFAULT_REGION_PRIORITY.get(str(target_role).strip(), []))

    if page_mode == "dialog":
        promoted = [x for x in out if is_dialog_region(x)]
        fallback = [x for x in out if not is_dialog_region(x)]
        out = promoted + fallback
        if not promoted:
            if target_role == "input":
                out = ["dialog_form_area", "dialog_body_area"] + out
            else:
                out = ["dialog_footer_primary_area", "dialog_top_right_close_area", "dialog_body_area"] + out

    if not out:
        out = ["form_input_area"] if str(target_role or "") == "input" else ["bottom_primary_area", "center_dialog", "center"]
    return _dedupe_keep_order(out)


def merge_region_hints(
    raw_target: Optional[str] = None,
    target_role: Optional[str] = None,
    explicit_hints: Optional[Sequence[str]] = None,
    page_mode: Optional[str] = None,
) -> List[str]:
    mode = infer_page_mode(explicit_page_mode=page_mode, target_role=target_role, region_hints=explicit_hints, raw_target=raw_target)
    ordered: List[str] = []
    ordered.extend(_dedupe_keep_order(explicit_hints or []))
    ordered.extend(default_regions_for_target(raw_target=raw_target, target_role=target_role, page_mode=mode))
    return _dedupe_keep_order(ordered)
