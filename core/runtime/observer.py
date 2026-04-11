from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.perception.ocr import PageSourceTextProvider
from core.perception.perception import PerceptionPack


@dataclass
class ObservedPage:
    pack: PerceptionPack
    step_dir: Path
    step_id: str


class Observer:
    def __init__(self, step_runner):
        self.step_runner = step_runner

    def observe(self, name: str = "observe", action: str = "OBSERVE") -> ObservedPage:
        self.step_runner.run(name=name, action=action, fn=lambda: None)
        return self.load_latest_observation()

    def load_latest_observation(self) -> ObservedPage:
        run_dir = Path(self.step_runner.evidence.run.run_dir)
        step_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()])
        if not step_dirs:
            raise RuntimeError(f"No evidence steps found under: {run_dir}")

        step_dir = step_dirs[-1]
        ocr_path = step_dir / "ocr.txt"
        perception_path = step_dir / "perception.json"
        image_path = step_dir / "screenshot_after.png"
        if not image_path.exists():
            image_path = step_dir / "screenshot.png"

        page_source_path: Optional[Path] = None
        for name in ["page_source_after.xml", "page_source.xml", "page_source_fail.xml"]:
            p = step_dir / name
            if p.exists():
                page_source_path = p
                break

        meta = {}
        if perception_path.exists():
            try:
                meta = json.loads(perception_path.read_text(encoding="utf-8")) or {}
            except Exception:
                meta = {"available": False, "error": "invalid perception.json"}
        elif image_path.exists():
            meta = {"available": False, "reason": "perception.json missing"}
        else:
            meta = {"available": False, "reason": "no screenshot/perception available"}

        ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
        page_text = ""
        if page_source_path and page_source_path.exists():
            try:
                xml = page_source_path.read_text(encoding="utf-8", errors="ignore")
                page_text = PageSourceTextProvider().recognize_from_page_source(xml).text or ""
            except Exception:
                page_text = ""

        merged_lines = []
        seen = set()
        for block in [ocr_text, page_text]:
            for line in (block or "").splitlines():
                s = line.strip()
                if s and s not in seen:
                    seen.add(s)
                    merged_lines.append(s)
        merged_text = "\n".join(merged_lines)

        meta = dict(meta or {})
        meta["page_source_path"] = str(page_source_path) if page_source_path else ""
        meta["step_dir"] = str(step_dir)
        meta["image_path"] = str(image_path) if image_path.exists() else ""
        meta["ocr_text_raw"] = ocr_text
        meta["page_text_raw"] = page_text
        meta["merged_text_enabled"] = True

        pack = PerceptionPack(image_path=str(image_path) if image_path.exists() else "", ocr_text=merged_text, meta=meta)
        step_id = step_dir.name.split('_', 1)[0]
        return ObservedPage(pack=pack, step_dir=step_dir, step_id=step_id)
