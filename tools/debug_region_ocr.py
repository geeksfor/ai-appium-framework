from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from core.perception.ocr import build_ocr_provider_from_env
from core.recovery.regions import get_region_box, list_region_names, project_region_box

PctBox = Tuple[float, float, float, float]


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def crop_region(image_path: str, region_pct: PctBox, out_crop_path: str) -> Dict[str, Any]:
    x1p, y1p, x2p, y2p = region_pct
    with Image.open(image_path) as im:
        w, h = im.size
        x1 = int(round(w * x1p))
        y1 = int(round(h * y1p))
        x2 = int(round(w * x2p))
        y2 = int(round(h * y2p))
        crop = im.crop((x1, y1, x2, y2))
        crop.save(out_crop_path)
    return {
        "image_width": w,
        "image_height": h,
        "crop_left": x1,
        "crop_top": y1,
        "crop_width": max(1, x2 - x1),
        "crop_height": max(1, y2 - y1),
        "region_pct": {"x1": x1p, "y1": y1p, "x2": x2p, "y2": y2p},
    }


def map_crop_boxes_to_full_image(crop_boxes: List[Dict[str, Any]], crop_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    crop_left = crop_meta["crop_left"]
    crop_top = crop_meta["crop_top"]
    crop_w = crop_meta["crop_width"]
    crop_h = crop_meta["crop_height"]
    full_w = crop_meta["image_width"]
    full_h = crop_meta["image_height"]
    full_boxes: List[Dict[str, Any]] = []
    for b in crop_boxes:
        try:
            bx1 = float(b["x1"])
            by1 = float(b["y1"])
            bx2 = float(b["x2"])
            by2 = float(b["y2"])
        except Exception:
            continue
        full_boxes.append(
            {
                "text": b.get("text", ""),
                "x1": clamp01((crop_left + bx1 * crop_w) / full_w),
                "y1": clamp01((crop_top + by1 * crop_h) / full_h),
                "x2": clamp01((crop_left + bx2 * crop_w) / full_w),
                "y2": clamp01((crop_top + by2 * crop_h) / full_h),
                **({"score": b["score"]} if "score" in b else {}),
            }
        )
    return full_boxes


def draw_debug(image_path: str, region_pct: PctBox, full_boxes: List[Dict[str, Any]], out_debug_path: str) -> None:
    with Image.open(image_path) as im:
        w, h = im.size
        draw = ImageDraw.Draw(im)
        rx1 = int(round(w * region_pct[0]))
        ry1 = int(round(h * region_pct[1]))
        rx2 = int(round(w * region_pct[2]))
        ry2 = int(round(h * region_pct[3]))
        draw.rectangle((rx1, ry1, rx2, ry2), outline="blue", width=4)
        for b in full_boxes:
            x1 = int(round(w * float(b["x1"])))
            y1 = int(round(h * float(b["y1"])))
            x2 = int(round(w * float(b["x2"])))
            y2 = int(round(h * float(b["y2"])))
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        im.save(out_debug_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--region", default="bottom_primary_area", choices=list_region_names())
    parser.add_argument("--out-dir", default="tmp_region_ocr")
    parser.add_argument("--container", default="", help="optional dialog container pct box: x1,y1,x2,y2")
    return parser.parse_args()


def parse_container(raw: str) -> Optional[PctBox]:
    s = str(raw or "").strip()
    if not s:
        return None
    nums = [float(x.strip()) for x in s.split(",")]
    if len(nums) != 4:
        raise ValueError("container must be x1,y1,x2,y2")
    return clamp01(nums[0]), clamp01(nums[1]), clamp01(nums[2]), clamp01(nums[3])


def main() -> None:
    args = parse_args()
    container = parse_container(args.container)
    region_pct = project_region_box(args.region, container)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_path = str(out_dir / "region_crop.png")
    debug_path = str(out_dir / "region_ocr_debug.png")
    json_path = str(out_dir / "region_ocr.json")

    crop_meta = crop_region(args.image, region_pct, crop_path)
    provider = build_ocr_provider_from_env()
    if provider is None or not hasattr(provider, "recognize_with_boxes"):
        raise RuntimeError("No OCR boxes provider available")
    result = provider.recognize_with_boxes(crop_path)  # type: ignore[attr-defined]
    full_boxes = map_crop_boxes_to_full_image(result.boxes or [], crop_meta)
    payload = {
        "provider": result.provider,
        "model": result.model,
        "elapsed_ms": result.elapsed_ms,
        "error": result.error,
        "crop_meta": crop_meta,
        "crop_text": result.text,
        "crop_boxes": result.boxes or [],
        "full_boxes": full_boxes,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    draw_debug(args.image, region_pct, full_boxes, debug_path)
    print(f"[OK] region={args.region} json={json_path}")
    print(f"[OK] debug={debug_path}")


if __name__ == "__main__":
    main()
