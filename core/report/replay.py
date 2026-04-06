from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Optional

from core.executor.plan_loader import load_plan


class FailureCategory(str, Enum):
    LOCATOR_FAILED = "定位失败"
    POPUP_UNCOVERED = "弹窗未覆盖"
    LOAD_TIMEOUT = "加载超时"
    AI_OUTPUT_INVALID = "AI输出非法"
    UNKNOWN = "未知失败"


@dataclass
class FailureAnalysis:
    category: FailureCategory
    reason: str
    evidence: dict[str, Any]


@dataclass
class ReplayResult:
    ok: bool
    source_run_id: str
    replay_run_id: str
    actions_path: str
    executed: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    failure: Optional[FailureAnalysis] = None
    summary_path: Optional[str] = None


@dataclass
class BatchReplayItem:
    run_index: int
    ok: bool
    replay_run_id: str
    executed: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    failure_category: Optional[str] = None
    failure_reason: Optional[str] = None
    summary_path: Optional[str] = None


@dataclass
class StabilityReport:
    batch_id: str
    source_run_id: str
    actions_path: str
    total_runs: int
    passed_runs: int
    failed_runs: int
    success_rate: float
    unstable: bool
    category_counts: dict[str, int]
    avg_executed: float
    min_executed: int
    max_executed: int
    started_at: str
    finished_at: str
    elapsed_ms: int
    report_dir: str
    source_actions_copy: Optional[str] = None
    json_report_path: Optional[str] = None
    md_report_path: Optional[str] = None
    items: list[BatchReplayItem] | None = None


FactoryReturn = Any
ExecutorFactory = Callable[[], FactoryReturn]


def _sorted_run_dirs(base_dir: str | Path) -> list[Path]:
    root = Path(base_dir)
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _new_batch_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"batch_{ts}_{uuid.uuid4().hex[:6]}"


def find_latest_run_with_actions(base_dir: str | Path = "evidence", actions_filename: str = "actions.json") -> Path:
    for run_dir in _sorted_run_dirs(base_dir):
        if (run_dir / actions_filename).exists():
            return run_dir
    raise FileNotFoundError(f"No run with {actions_filename} found under: {base_dir}")


def find_run_dir(base_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(base_dir) / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    return run_dir


def load_actions_from_run(run_dir: str | Path, actions_filename: str = "actions.json"):
    path = Path(run_dir) / actions_filename
    if not path.exists():
        raise FileNotFoundError(f"actions file not found: {path}")
    return load_plan(path), path


def _iter_step_dirs(run_dir: Path) -> list[Path]:
    step_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name[:4].isdigit()]
    step_dirs.sort(key=lambda p: p.name)
    return step_dirs


def _last_step_dir(run_dir: Path) -> Path | None:
    steps = _iter_step_dirs(run_dir)
    return steps[-1] if steps else None


def _last_failed_step_dir(run_dir: Path) -> Path | None:
    for step_dir in reversed(_iter_step_dirs(run_dir)):
        meta = _read_json(step_dir / "meta.json") or {}
        if meta.get("result") == "FAIL":
            return step_dir
    return None


def _popup_signal(step_dir: Path | None) -> tuple[bool, list[str]]:
    if step_dir is None:
        return False, []

    evidence: list[str] = []
    popup_keywords = [
        "允许",
        "同意",
        "稍后",
        "关闭",
        "跳过",
        "取消",
        "我知道了",
        "权限",
        "登录",
        "授权",
        "privacy",
        "permission",
        "allow",
        "deny",
        "skip",
        "close",
        "popup",
        "overlay",
    ]

    ocr_text = _read_text(step_dir / "ocr.txt")
    ocr_hit = [kw for kw in popup_keywords if kw.lower() in ocr_text.lower()]
    if len(ocr_hit) >= 2:
        evidence.append(f"ocr_keywords={ocr_hit[:6]}")

    perception = _read_json(step_dir / "perception.json") or {}
    if bool(perception.get("overlay_suspected")):
        evidence.append("perception.overlay_suspected=true")

    meta = _read_json(step_dir / "meta.json") or {}
    extra = meta.get("extra") or {}
    if extra.get("no_progress") is True:
        evidence.append("meta.extra.no_progress=true")

    return bool(evidence), evidence


def classify_failure(run_dir: str | Path) -> FailureAnalysis:
    run_dir = Path(run_dir)

    policy = _read_json(run_dir / "policy" / "policy_parsed.json") or {}
    policy_reason = str(policy.get("reason") or "")
    policy_raw = json.dumps(policy, ensure_ascii=False)
    ai_invalid_keywords = [
        "PolicyParseError",
        "Invalid JSON from model",
        "ActionPlan validation failed",
        "Unknown policy JSON schema",
        "validation failed",
        "extra_forbidden",
    ]
    ai_blob = f"{policy_reason}\n{policy_raw}".lower()
    if any(k.lower() in ai_blob for k in ai_invalid_keywords):
        return FailureAnalysis(
            category=FailureCategory.AI_OUTPUT_INVALID,
            reason=policy_reason or "policy 输出无法解析或 schema 校验失败",
            evidence={
                "policy_parsed": str(run_dir / "policy" / "policy_parsed.json"),
                "policy_raw": str(run_dir / "policy" / "policy_raw.txt"),
            },
        )

    step_dir = _last_failed_step_dir(run_dir) or _last_step_dir(run_dir)
    meta = _read_json(step_dir / "meta.json") if step_dir else {}
    meta = meta or {}
    action = str(meta.get("action") or "")
    err_type = str(meta.get("error_type") or "")
    err_msg = str(meta.get("error_message") or "")
    extra = meta.get("extra") or {}
    combined = "\n".join([action, err_type, err_msg, json.dumps(extra, ensure_ascii=False)])

    timeout_keywords = ["timeout", "timed out", "loading", "load timeout", "wait timeout"]
    if any(k in combined.lower() for k in timeout_keywords):
        return FailureAnalysis(
            category=FailureCategory.LOAD_TIMEOUT,
            reason=err_msg or "检测到 timeout / loading 相关异常",
            evidence={
                "step_dir": str(step_dir) if step_dir else None,
                "meta": str((step_dir / "meta.json") if step_dir else ""),
            },
        )

    popup_suspected, popup_evidence = _popup_signal(step_dir)
    if popup_suspected:
        return FailureAnalysis(
            category=FailureCategory.POPUP_UNCOVERED,
            reason=err_msg or "失败前画面仍有明显弹窗/遮罩信号，说明规则未覆盖或恢复策略不足",
            evidence={
                "step_dir": str(step_dir) if step_dir else None,
                "signals": popup_evidence,
                "ocr": str((step_dir / "ocr.txt") if step_dir else ""),
            },
        )

    locator_keywords = [
        "normal click failed",
        "heal failed",
        "target not found",
        "not found",
        "locator",
        "bbox",
        "click failed",
        "tap",
        "coordinates",
        "clickgesture",
        "assert failed: text not found",
    ]
    if action.endswith("CLICK") or any(k in combined.lower() for k in locator_keywords):
        return FailureAnalysis(
            category=FailureCategory.LOCATOR_FAILED,
            reason=err_msg or "点击/定位链路失败，未找到合适目标或重试后仍失败",
            evidence={
                "step_dir": str(step_dir) if step_dir else None,
                "meta": str((step_dir / "meta.json") if step_dir else ""),
                "ocr": str((step_dir / "ocr.txt") if step_dir else ""),
            },
        )

    return FailureAnalysis(
        category=FailureCategory.UNKNOWN,
        reason=err_msg or "未命中已知分类规则，请结合最后一步证据人工分析",
        evidence={
            "step_dir": str(step_dir) if step_dir else None,
            "meta": str((step_dir / "meta.json") if step_dir else ""),
        },
    )


def write_failure_classification(run_dir: str | Path, analysis: FailureAnalysis) -> str:
    run_dir = Path(run_dir)
    out = run_dir / "failure_classification.json"
    out.write_text(json.dumps(asdict(analysis), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def write_replay_summary(run_dir: str | Path, result: ReplayResult) -> str:
    run_dir = Path(run_dir)
    payload = asdict(result)
    if result.failure is not None:
        payload["failure"] = asdict(result.failure)
    out = run_dir / "replay_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def replay_plan_file(executor, plan_path: str | Path, *, source_run_id: str = "external") -> ReplayResult:
    plan_path = Path(plan_path)
    plan = load_plan(plan_path)
    replay_run_id = executor.step_runner.evidence.run.run_id

    try:
        exec_result = executor.run_plan(plan)
        result = ReplayResult(
            ok=True,
            source_run_id=source_run_id,
            replay_run_id=replay_run_id,
            actions_path=str(plan_path),
            executed=exec_result.executed,
        )
    except Exception as e:
        run_dir = Path(executor.step_runner.evidence.run.run_dir)
        failure = classify_failure(run_dir)
        write_failure_classification(run_dir, failure)
        result = ReplayResult(
            ok=False,
            source_run_id=source_run_id,
            replay_run_id=replay_run_id,
            actions_path=str(plan_path),
            error_type=type(e).__name__,
            error_message=str(e),
            failure=failure,
        )

    result.summary_path = write_replay_summary(Path(executor.step_runner.evidence.run.run_dir), result)
    return result


def replay_run(executor, source_run_dir: str | Path, *, actions_filename: str = "actions.json") -> ReplayResult:
    source_run_dir = Path(source_run_dir)
    _, actions_path = load_actions_from_run(source_run_dir, actions_filename=actions_filename)
    return replay_plan_file(executor, actions_path, source_run_id=source_run_dir.name)


def replay_latest_run(executor, *, base_dir: str | Path = "evidence", actions_filename: str = "actions.json") -> ReplayResult:
    latest = find_latest_run_with_actions(base_dir=base_dir, actions_filename=actions_filename)
    return replay_run(executor, latest, actions_filename=actions_filename)


def _normalize_factory_output(factory_output: FactoryReturn):
    if isinstance(factory_output, tuple) and len(factory_output) == 2:
        return factory_output[0], factory_output[1]
    if isinstance(factory_output, dict) and "executor" in factory_output:
        cleanup = factory_output.get("cleanup") or (lambda: None)
        return factory_output["executor"], cleanup
    return factory_output, (lambda: None)


def _copy_actions_to_report(plan_path: Path, report_dir: Path) -> str:
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "source_actions.json"
    shutil.copy2(plan_path, out)
    return str(out)


def _category_counts(items: list[BatchReplayItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        if not item.ok:
            key = item.failure_category or FailureCategory.UNKNOWN.value
            counts[key] = counts.get(key, 0) + 1
    return counts


def _build_markdown(report: StabilityReport) -> str:
    lines = [
        "# Stability Report",
        "",
        f"- batch_id: `{report.batch_id}`",
        f"- source_run_id: `{report.source_run_id}`",
        f"- actions_path: `{report.actions_path}`",
        f"- total_runs: {report.total_runs}",
        f"- passed_runs: {report.passed_runs}",
        f"- failed_runs: {report.failed_runs}",
        f"- success_rate: {report.success_rate:.2f}%",
        f"- unstable: {'yes' if report.unstable else 'no'}",
        f"- avg_executed: {report.avg_executed:.2f}",
        f"- min_executed: {report.min_executed}",
        f"- max_executed: {report.max_executed}",
        f"- elapsed_ms: {report.elapsed_ms}",
        "",
        "## Failure Categories",
        "",
    ]
    if report.category_counts:
        for key, value in sorted(report.category_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Per Run",
        "",
        "| # | ok | replay_run_id | executed | category | error |",
        "|---|---|---|---:|---|---|",
    ])
    for item in report.items or []:
        lines.append(
            f"| {item.run_index} | {'Y' if item.ok else 'N'} | `{item.replay_run_id}` | {item.executed} | {item.failure_category or ''} | {item.error_type or ''}: {item.error_message or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_stability_report(report_dir: str | Path, report: StabilityReport) -> tuple[str, str]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "stability_report.json"
    md_path = report_dir / "stability_report.md"

    payload = asdict(report)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    return str(json_path), str(md_path)


def _aggregate_batch(
    *,
    batch_id: str,
    source_run_id: str,
    plan_path: Path,
    started_at: float,
    report_dir: Path,
    items: list[BatchReplayItem],
) -> StabilityReport:
    total_runs = len(items)
    passed_runs = sum(1 for item in items if item.ok)
    failed_runs = total_runs - passed_runs
    success_rate = round((passed_runs / total_runs) * 100.0, 2) if total_runs else 0.0
    executed_values = [item.executed for item in items] or [0]
    finished = time.time()

    report = StabilityReport(
        batch_id=batch_id,
        source_run_id=source_run_id,
        actions_path=str(plan_path),
        total_runs=total_runs,
        passed_runs=passed_runs,
        failed_runs=failed_runs,
        success_rate=success_rate,
        unstable=(passed_runs > 0 and failed_runs > 0),
        category_counts=_category_counts(items),
        avg_executed=round(mean(executed_values), 2),
        min_executed=min(executed_values),
        max_executed=max(executed_values),
        started_at=datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        finished_at=datetime.fromtimestamp(finished).isoformat(timespec="seconds"),
        elapsed_ms=int((finished - started_at) * 1000),
        report_dir=str(report_dir),
        source_actions_copy=str(report_dir / "source_actions.json"),
        items=items,
    )
    json_path, md_path = write_stability_report(report_dir, report)
    report.json_report_path = json_path
    report.md_report_path = md_path
    write_stability_report(report_dir, report)
    return report


def replay_plan_batch(
    executor_factory: ExecutorFactory,
    plan_path: str | Path,
    *,
    times: int = 10,
    source_run_id: str = "external",
    batch_base_dir: str | Path = "evidence/_batches",
    sleep_s: float = 0.0,
    stop_on_first_failure: bool = False,
) -> StabilityReport:
    if times <= 0:
        raise ValueError("times must be > 0")

    plan_path = Path(plan_path)
    if not plan_path.exists():
        raise FileNotFoundError(f"plan file not found: {plan_path}")

    batch_id = _new_batch_id()
    report_dir = Path(batch_base_dir) / batch_id
    report_dir.mkdir(parents=True, exist_ok=True)
    _copy_actions_to_report(plan_path, report_dir)

    started_at = time.time()
    items: list[BatchReplayItem] = []

    for idx in range(1, times + 1):
        cleanup = lambda: None
        factory_output = executor_factory()
        executor, cleanup = _normalize_factory_output(factory_output)
        try:
            result = replay_plan_file(executor, plan_path, source_run_id=source_run_id)
        finally:
            try:
                cleanup()
            except Exception:
                pass

        items.append(
            BatchReplayItem(
                run_index=idx,
                ok=result.ok,
                replay_run_id=result.replay_run_id,
                executed=result.executed,
                error_type=result.error_type,
                error_message=result.error_message,
                failure_category=(result.failure.category.value if result.failure else None),
                failure_reason=(result.failure.reason if result.failure else None),
                summary_path=result.summary_path,
            )
        )

        if stop_on_first_failure and not result.ok:
            break
        if sleep_s > 0 and idx < times:
            time.sleep(sleep_s)

    return _aggregate_batch(
        batch_id=batch_id,
        source_run_id=source_run_id,
        plan_path=plan_path,
        started_at=started_at,
        report_dir=report_dir,
        items=items,
    )


def replay_run_batch(
    executor_factory: ExecutorFactory,
    source_run_dir: str | Path,
    *,
    times: int = 10,
    actions_filename: str = "actions.json",
    batch_base_dir: str | Path = "evidence/_batches",
    sleep_s: float = 0.0,
    stop_on_first_failure: bool = False,
) -> StabilityReport:
    source_run_dir = Path(source_run_dir)
    _, actions_path = load_actions_from_run(source_run_dir, actions_filename=actions_filename)
    return replay_plan_batch(
        executor_factory,
        actions_path,
        times=times,
        source_run_id=source_run_dir.name,
        batch_base_dir=batch_base_dir,
        sleep_s=sleep_s,
        stop_on_first_failure=stop_on_first_failure,
    )


def replay_latest_run_batch(
    executor_factory: ExecutorFactory,
    *,
    base_dir: str | Path = "evidence",
    actions_filename: str = "actions.json",
    times: int = 10,
    batch_base_dir: str | Path = "evidence/_batches",
    sleep_s: float = 0.0,
    stop_on_first_failure: bool = False,
) -> StabilityReport:
    latest = find_latest_run_with_actions(base_dir=base_dir, actions_filename=actions_filename)
    return replay_run_batch(
        executor_factory,
        latest,
        times=times,
        actions_filename=actions_filename,
        batch_base_dir=batch_base_dir,
        sleep_s=sleep_s,
        stop_on_first_failure=stop_on_first_failure,
    )
