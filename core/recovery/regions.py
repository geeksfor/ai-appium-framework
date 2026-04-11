from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

PctBox = Tuple[float, float, float, float]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass(frozen=True)
class RegionSpec:
    name: str
    box: PctBox
    desc: str = ""

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    def normalized(self) -> PctBox:
        x1, y1, x2, y2 = self.box
        x1 = _clamp01(x1)
        y1 = _clamp01(y1)
        x2 = _clamp01(x2)
        y2 = _clamp01(y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid region box for {self.name}: {self.box}")
        return x1, y1, x2, y2


# 通用交互区域：不是按页面定制，而是按控件/交互模式定义。
REGIONS: Dict[str, RegionSpec] = {
    "top_left": RegionSpec(
        "top_left", (0.0, 0.0, 0.35, 0.22), "返回、侧边返回入口、页面左上角操作"
    ),
    "top_right": RegionSpec(
        "top_right", (0.65, 0.0, 1.0, 0.22), "关闭、跳过、右上角菜单"
    ),
    "center": RegionSpec(
        "center", (0.15, 0.18, 0.85, 0.82), "中心主体内容区"
    ),
    "center_dialog": RegionSpec(
        "center_dialog", (0.12, 0.18, 0.88, 0.82), "弹窗主体区"
    ),
    "form_input_area": RegionSpec(
        "form_input_area", (0.22, 0.42, 0.78, 0.80), "表单输入区：用户名/密码/验证码等"
    ),
    "bottom_primary_area": RegionSpec(
        "bottom_primary_area", (0.20, 0.72, 0.80, 0.95), "页面底部主按钮区：登录/提交/下一步/确认"
    ),
    "bottom_sheet_area": RegionSpec(
        "bottom_sheet_area", (0.0, 0.62, 1.0, 1.0), "底部抽屉、ActionSheet、底部弹层"
    ),
    "checkbox_area": RegionSpec(
        "checkbox_area", (0.45, 0.70, 0.72, 0.88), "协议勾选、记住密码等复选区域"
    ),
}


ROLE_DEFAULT_REGIONS = {
    "button": ["bottom_primary_area", "center_dialog", "center"],
    "input": ["form_input_area", "center"],
    "checkbox": ["checkbox_area", "bottom_sheet_area", "center_dialog"],
    "icon_close": ["top_right", "center_dialog"],
    "icon_back": ["top_left"],
}


TARGET_DEFAULT_REGIONS = {
    "登录": ["bottom_primary_area"],
    "立即登录": ["bottom_primary_area"],
    "去登录": ["bottom_primary_area"],
    "提交": ["bottom_primary_area"],
    "下一步": ["bottom_primary_area"],
    "确定": ["bottom_primary_area", "center_dialog"],
    "保存": ["bottom_primary_area"],
    "关闭": ["top_right", "center_dialog"],
    "跳过": ["top_right"],
    "返回": ["top_left"],
    "用户名": ["form_input_area"],
    "密码": ["form_input_area"],
    "验证码": ["form_input_area"],
}


def get_region_box(name: str) -> PctBox:
    spec = REGIONS[name]
    return spec.normalized()


def list_region_names() -> list[str]:
    return list(REGIONS.keys())


def merge_region_hints(
    raw_target: str | None = None,
    target_role: str | None = None,
    explicit_hints: list[str] | None = None,
) -> list[str]:
    ordered: list[str] = []

    def _add(name: str) -> None:
        if name in REGIONS and name not in ordered:
            ordered.append(name)

    for name in explicit_hints or []:
        _add(name)

    if raw_target:
        for name in TARGET_DEFAULT_REGIONS.get(str(raw_target).strip(), []):
            _add(name)

    if target_role:
        for name in ROLE_DEFAULT_REGIONS.get(str(target_role).strip(), []):
            _add(name)

    return ordered
