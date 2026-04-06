from __future__ import annotations

from core.perception.perception import PerceptionPack
from core.recovery.click_resolver import ClickResolver


def test_click_resolver_resolve_by_bbox_prefers_semantic_match():
    pack = PerceptionPack(
        image_path="",
        ocr_text="关闭 同意并继续",
        meta={
            "ocr_boxes": [
                {"text": "关闭", "x1": 0.84, "y1": 0.10, "x2": 0.94, "y2": 0.18},
                {"text": "同意并继续", "x1": 0.60, "y1": 0.82, "x2": 0.92, "y2": 0.90},
            ]
        },
    )

    resolver = ClickResolver(policy_runner=None)
    cp = resolver.resolve(pack, ["同意", "允许"], ui_hints=["优先底部右侧主要按钮（同意/允许/确定）"])

    assert cp is not None
    assert cp.source == "bbox"
    assert abs(cp.x_pct - 0.76) < 1e-6
    assert abs(cp.y_pct - 0.86) < 1e-6


def test_click_resolver_supports_bbox_list_format():
    pack = PerceptionPack(
        image_path="",
        ocr_text="点击跳过",
        meta={"ocr_boxes": [{"text": "点击跳过", "bbox": [0.80, 0.10, 0.94, 0.18]}]},
    )

    resolver = ClickResolver(policy_runner=None)
    cp = resolver.resolve_by_bbox(pack, ["跳过"], ui_hints=["优先右上角关闭或跳过按钮"])

    assert cp is not None
    assert cp.source == "bbox"
    assert abs(cp.x_pct - 0.87) < 1e-6
    assert abs(cp.y_pct - 0.14) < 1e-6
