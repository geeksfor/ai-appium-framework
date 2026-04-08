from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.perception.perception import PerceptionPack


@dataclass
class ObservedPage:
    pack: PerceptionPack
    step_dir: Path
    step_id: str


class Observer:
    """
    统一“观察当前页面”的入口。
    对测试用例隐藏 evidence/perception 文件细节。
    """

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

        text = ocr_path.read_text(encoding="utf-8", errors="ignore") if ocr_path.exists() else ""
        pack = PerceptionPack(image_path=str(image_path) if image_path.exists() else "", ocr_text=text, meta=meta)
        step_id = step_dir.name.split('_', 1)[0]
        return ObservedPage(pack=pack, step_dir=step_dir, step_id=step_id)
