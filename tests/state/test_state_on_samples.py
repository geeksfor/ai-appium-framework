import os
import glob
import pytest

from core.perception.ocr import QwenVisionOCRProvider
from core.perception.perception import Perception
from core.state.state_machine import StateMachine


def _get_sample(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample not found: {path}")
    return path


@pytest.mark.skipif(not os.getenv("DASHSCOPE_API_KEY"), reason="DASHSCOPE_API_KEY not set")
def test_state_on_plan_tab_sample():
    img = _get_sample("tests/samples/WechatIMG169.jpeg")

    ocr = QwenVisionOCRProvider(
        model=os.getenv("DASHSCOPE_MODEL", "qwen-vl-ocr-latest"),
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    perception = Perception(ocr_provider=ocr)

    pack = perception.perceive_image(img)
    sm = StateMachine()

    res = sm.detect_state(pack)

    print("\nOCR:\n", pack.ocr_text)
    print("\nBEST:\n", res.best)
    print("\nALL MATCHES:")
    for m in res.matches:
        print(m.state, m.score, m.hits, m.reason)

    print("\n=== DETECTED STATE ===", res.state, res.score)
    print("\n=== CANDIDATES (>=min_score) ===")
    for c in res.meta["candidates"]:
        print(c)

    print("\n=== TOPK (ALL) ===")
    for t in res.meta["topk"]:
        print(t)

    assert res.state == "Popup.PrivacyConsent"