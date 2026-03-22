# tests/perception/test_ocr_on_samples.py
import os
import glob
import pytest

from core.perception.ocr import QwenVisionOCRProvider
from core.perception.perception import Perception


def _find_sample_image() -> str:
    # 兼容你可能的命名：测糖计划.png / cetang.png ...
    patterns = [
        # "tests/samples/*测糖*.*",
        # "tests/samples/*cetang*.*",
        # "tests/samples/*.png",
        # "tests/samples/*.jpg",
        "tests/samples/yindaotishi.jpeg",
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise FileNotFoundError("No sample image found under tests/samples/")


@pytest.mark.skipif(not os.getenv("DASHSCOPE_API_KEY"), reason="DASHSCOPE_API_KEY not set")
def test_ocr_on_samples():
    img = _find_sample_image()

    ocr = QwenVisionOCRProvider(
        # 推荐先用专用 OCR 模型；你想用 qwen3-vl-plus 也可替换
        model="qwen-vl-ocr-latest",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    p = Perception(ocr_provider=ocr)
    pack = p.perceive_image(img)

    print("\n===== OCR TEXT =====\n", pack.ocr_text)
    print("\n===== META =====\n", pack.meta)

    assert pack.meta["available"] is True
    # 你可以按“测糖计划”图里必然出现的关键字做弱断言，比如：
    # assert "测糖" in pack.ocr_text