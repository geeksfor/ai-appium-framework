from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont


def render_click_debug(
    image_path: str,
    boxes: Iterable[Dict[str, Any]],
    chosen: Optional[Dict[str, Any]],
    out_path: str,
    note: str = "",
) -> Optional[str]:
    if not image_path or not Path(image_path).exists():
        return None

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path).convert("RGB") as im:
        draw = ImageDraw.Draw(im)
        w, h = im.size
        font = ImageFont.load_default()

        for box in boxes:
            try:
                x1 = float(box.get("x1", 0.0)) * w
                y1 = float(box.get("y1", 0.0)) * h
                x2 = float(box.get("x2", 0.0)) * w
                y2 = float(box.get("y2", 0.0)) * h
            except Exception:
                continue
            draw.rectangle((x1, y1, x2, y2), outline=(64, 190, 255), width=2)
            label = str(box.get("text", "")).strip()
            if label:
                draw.text((x1 + 2, max(0, y1 - 12)), label[:20], fill=(64, 190, 255), font=font)

        if chosen:
            cp_x = float(chosen.get("x_pct", 0.5)) * w
            cp_y = float(chosen.get("y_pct", 0.5)) * h
            r = 14
            draw.line((cp_x - r, cp_y, cp_x + r, cp_y), fill=(255, 64, 64), width=3)
            draw.line((cp_x, cp_y - r, cp_x, cp_y + r), fill=(255, 64, 64), width=3)
            draw.ellipse((cp_x - 6, cp_y - 6, cp_x + 6, cp_y + 6), outline=(255, 64, 64), width=3)

            bbox = chosen.get("bbox") or {}
            if bbox:
                bx1 = float(bbox.get("x1", 0.0)) * w
                by1 = float(bbox.get("y1", 0.0)) * h
                bx2 = float(bbox.get("x2", 0.0)) * w
                by2 = float(bbox.get("y2", 0.0)) * h
                draw.rectangle((bx1, by1, bx2, by2), outline=(255, 150, 64), width=3)

            reason = str(chosen.get("reason", "")).strip()
            text = reason[:80]
            if text:
                draw.text((10, 10), text, fill=(255, 64, 64), font=font)

        if note:
            draw.text((10, h - 18), note[:120], fill=(255, 255, 0), font=font)

        im.save(out)
    return str(out)
