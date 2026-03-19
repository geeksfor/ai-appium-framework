# core/report/step_logger.py
from __future__ import annotations
import json
from typing import Optional

from core.utils.logger import get_logger


class StepLogger:
    """
    统一 step 日志格式：
    - start: step/action
    - end: result/ms
    - fail: error
    输出为单行 JSON，方便 grep / ELK / Loki
    """
    def __init__(self, name: str = "STEP"):
        self.logger = get_logger(name)

    def start(self, run_id: str, step_id: str, action: str, step_name: str):
        payload = {
            "event": "step_start",
            "run_id": run_id,
            "step_id": step_id,
            "action": action,
            "step": step_name,
        }
        self.logger.info(json.dumps(payload, ensure_ascii=False))

    def end(self, run_id: str, step_id: str, action: str, step_name: str, result: str, ms: int, error: Optional[dict] = None):
        payload = {
            "event": "step_end",
            "run_id": run_id,
            "step_id": step_id,
            "action": action,
            "step": step_name,
            "result": result,
            "ms": ms,
        }
        if error:
            payload["error"] = error
        self.logger.info(json.dumps(payload, ensure_ascii=False))