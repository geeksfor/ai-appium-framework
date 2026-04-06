from __future__ import annotations

from core.perception.perception import PerceptionPack
from core.recovery.popup_handlers import PopupHandlers


def test_popup_handlers_builds_click_plan_with_semantic_fields():
    pack = PerceptionPack(
        image_path="",
        ocr_text="隐私政策 服务协议 同意",
        meta={
            "ocr_boxes": [
                {"text": "同意", "x1": 0.66, "y1": 0.82, "x2": 0.86, "y2": 0.90}
            ]
        },
    )

    handlers = PopupHandlers()
    res = handlers.handle(pack)

    assert res.handled is True
    assert res.plan is not None
    a0 = res.plan.actions[0]
    assert getattr(a0, "type").value == "CLICK"
    assert getattr(a0, "target") == "同意"
    assert getattr(a0, "selector") == "text=同意"
    assert getattr(a0, "allow_heal") is True
