from __future__ import annotations

import concurrent.futures
import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def new_run_id() -> str:
    return f"{_now_str()}_{uuid.uuid4().hex[:6]}"


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s).strip("_")


@dataclass
class StepMeta:
    run_id: str
    step_id: str
    name: str
    action: str
    start_ts: float
    end_ts: float
    duration_ms: int
    result: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    extra: Optional[dict] = None


class EvidenceRun:
    def __init__(self, base_dir: str = "evidence", run_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.run_id = run_id or new_run_id()
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._step_counter = 0
        self._write_run_json()

    def _write_run_json(self) -> None:
        info = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_dir": str(self.base_dir),
        }
        (self.run_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def next_step_id(self) -> str:
        self._step_counter += 1
        return f"{self._step_counter:04d}"

    def step_dir(self, step_id: str, step_name: str) -> Path:
        name = _safe_name(step_name)
        d = self.run_dir / f"{step_id}_{name}"
        d.mkdir(parents=True, exist_ok=True)
        return d


class EvidenceStep:
    def __init__(self, run: EvidenceRun, step_id: str, name: str, action: str):
        self.run = run
        self.step_id = step_id
        self.name = name
        self.action = action
        self.dir = run.step_dir(step_id, name)
        self.start_ts = time.time()
        self.end_ts = self.start_ts
        self.result = "OK"
        self.error_type = None
        self.error_message = None
        self.error_stack = None
        self.extra: dict = {}

    def attach_json(self, filename: str, data: Any) -> str:
        if not filename.endswith('.json'):
            filename += '.json'
        p = self.dir / filename
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        return str(p)

    def attach_text(self, filename: str, text: str) -> str:
        p = self.dir / filename
        p.write_text(text, encoding='utf-8')
        return str(p)

    def attach_screenshot(self, driver, filename: str = 'screenshot.png') -> Optional[str]:
        try:
            p = self.dir / filename
            ok = driver.save_screenshot(str(p))
            if not ok:
                self.add_extra('screenshot_available', False)
                self.add_extra('screenshot_reason', 'save_screenshot returned False')
                return None
            self.add_extra('screenshot_available', True)
            return str(p)
        except Exception as e:
            self.add_extra('screenshot_available', False)
            self.add_extra('screenshot_reason', f"{type(e).__name__}: {e}")
            return None

    @staticmethod
    def get_page_source_with_timeout(driver, timeout_sec: int = 3) -> Optional[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(lambda: driver.page_source)
            try:
                return fut.result(timeout=timeout_sec)
            except Exception:
                return None

    def attach_page_source(self, driver, filename: str = 'page_source.xml') -> Optional[str]:
        try:
            p = self.dir / filename
            src = self.get_page_source_with_timeout(driver)
            if not src or not src.strip():
                self.add_extra('page_source_available', False)
                self.add_extra('page_source_reason', 'empty')
                return None
            p.write_text(src, encoding='utf-8')
            self.add_extra('page_source_available', True)
            return str(p)
        except Exception as e:
            self.add_extra('page_source_available', False)
            self.add_extra('page_source_reason', f"{type(e).__name__}: {e}")
            return None

    def add_extra(self, key: str, value: Any) -> None:
        self.extra[key] = value

    def mark_fail(self, err: BaseException, stack: Optional[str] = None) -> None:
        self.result = 'FAIL'
        self.error_type = type(err).__name__
        self.error_message = str(err)
        self.error_stack = stack

    def finalize(self) -> StepMeta:
        self.end_ts = time.time()
        duration_ms = int((self.end_ts - self.start_ts) * 1000)
        meta = StepMeta(
            run_id=self.run.run_id,
            step_id=self.step_id,
            name=self.name,
            action=self.action,
            start_ts=self.start_ts,
            end_ts=self.end_ts,
            duration_ms=duration_ms,
            result=self.result,
            error_type=self.error_type,
            error_message=self.error_message,
            error_stack=self.error_stack,
            extra=self.extra or None,
        )
        (self.dir / 'meta.json').write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding='utf-8')
        return meta


class EvidenceManager:
    def __init__(self, base_dir: str = 'evidence', run_id: Optional[str] = None):
        self.run = EvidenceRun(base_dir=base_dir, run_id=run_id)

    def new_step(self, name: str, action: str) -> EvidenceStep:
        step_id = self.run.next_step_id()
        return EvidenceStep(self.run, step_id=step_id, name=name, action=action)
