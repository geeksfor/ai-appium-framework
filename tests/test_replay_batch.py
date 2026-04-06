from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.report.replay import FailureCategory, replay_plan_batch


@dataclass
class _RunObj:
    run_id: str
    run_dir: Path


@dataclass
class _EvidenceObj:
    run: _RunObj


@dataclass
class _StepRunnerObj:
    evidence: _EvidenceObj


class _ExecOk:
    def __init__(self, run_dir: Path, run_id: str):
        self.step_runner = _StepRunnerObj(_EvidenceObj(_RunObj(run_id=run_id, run_dir=run_dir)))

    def run_plan(self, plan):
        class _R:
            executed = 2
        return _R()


class _ExecPopupFail:
    def __init__(self, run_dir: Path, run_id: str):
        self.step_runner = _StepRunnerObj(_EvidenceObj(_RunObj(run_id=run_id, run_dir=run_dir)))

    def run_plan(self, plan):
        step_dir = self.step_runner.evidence.run.run_dir / "0001_tap_save"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "meta.json").write_text(
            json.dumps(
                {
                    "action": "ActionType.CLICK",
                    "result": "FAIL",
                    "error_type": "RuntimeError",
                    "error_message": "click failed",
                    "extra": {"no_progress": True},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (step_dir / "ocr.txt").write_text("允许 跳过 稍后 权限提示", encoding="utf-8")
        (step_dir / "perception.json").write_text(
            json.dumps({"overlay_suspected": True}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError("click failed")


def test_replay_plan_batch_aggregates_results(tmp_path: Path):
    plan_path = tmp_path / "actions.json"
    plan_path.write_text('{"actions": []}', encoding="utf-8")

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    states = ["ok", "popup", "popup"]
    counter = {"i": 0}

    def factory():
        idx = counter["i"]
        counter["i"] += 1
        run_id = f"run_{idx + 1}"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if states[idx] == "ok":
            return _ExecOk(run_dir, run_id)
        return _ExecPopupFail(run_dir, run_id)

    report = replay_plan_batch(
        factory,
        plan_path,
        times=3,
        source_run_id="src_run",
        batch_base_dir=tmp_path / "batches",
    )

    assert report.total_runs == 3
    assert report.passed_runs == 1
    assert report.failed_runs == 2
    assert report.unstable is True
    assert report.category_counts[FailureCategory.POPUP_UNCOVERED.value] == 2
    assert Path(report.json_report_path).exists()
    assert Path(report.md_report_path).exists()
    assert Path(report.source_actions_copy).exists()
    assert len(report.items or []) == 3


def test_replay_plan_batch_stop_on_first_failure(tmp_path: Path):
    plan_path = tmp_path / "actions.json"
    plan_path.write_text('{"actions": []}', encoding="utf-8")

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    counter = {"i": 0}

    def factory():
        idx = counter["i"]
        counter["i"] += 1
        run_id = f"run_{idx + 1}"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return _ExecPopupFail(run_dir, run_id)

    report = replay_plan_batch(
        factory,
        plan_path,
        times=10,
        source_run_id="src_run",
        batch_base_dir=tmp_path / "batches",
        stop_on_first_failure=True,
    )

    assert report.total_runs == 1
    assert report.failed_runs == 1
