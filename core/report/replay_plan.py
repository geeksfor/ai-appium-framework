#!/usr/bin/env python3
# tools/replay_plan.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from core.driver.appium_adapter import AppiumAdapter
from core.executor.executor import Executor
from core.heal.heal_policy import HealPolicy, PolicyRunnerHealAIProvider
from core.perception.ocr import QwenOCRBoxesProvider
from core.perception.perception import Perception
from core.report.evidence import EvidenceManager
from core.report.replay import (
    find_latest_run_with_actions,
    replay_latest_run,
    replay_latest_run_batch,
    replay_plan_batch,
    replay_plan_file,
    replay_run,
    replay_run_batch,
)
from core.report.step_logger import StepLogger
from core.report.step_runner import StepRunner
from core.utils.config_loader import load_yaml


def build_adapter_from_device_yaml(device_yaml: str) -> AppiumAdapter:
    cfg = load_yaml(device_yaml)

    server_url = cfg["appium"]["server_url"]
    capabilities = {}
    capabilities.update(cfg["device"])
    capabilities.update(cfg["app"])

    implicit_wait = cfg.get("runtime", {}).get("implicit_wait", 10)
    evidence_dir = cfg.get("runtime", {}).get("evidence_dir", "evidence")

    return AppiumAdapter(
        server_url=server_url,
        capabilities=capabilities,
        implicit_wait=implicit_wait,
        evidence_dir=evidence_dir,
    )


def build_optional_perception() -> Perception | None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        return None
    ocr = QwenOCRBoxesProvider(
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model=os.getenv("DASHSCOPE_BBOX_MODEL", "qwen-vl-ocr-latest"),
    )
    return Perception(ocr_provider=ocr)


def build_executor_factory(device_yaml: str):
    cfg = load_yaml(device_yaml)
    evidence_dir = cfg.get("runtime", {}).get("evidence_dir", "evidence")

    def _factory():
        adapter = build_adapter_from_device_yaml(device_yaml)
        evidence = EvidenceManager(base_dir=evidence_dir)
        step_logger = StepLogger()
        perception = build_optional_perception()
        step_runner = StepRunner(
            evidence=evidence,
            step_logger=step_logger,
            driver_getter=lambda: adapter.driver,
            perception=perception,
        )
        ai_provider = PolicyRunnerHealAIProvider(None) if os.getenv("DASHSCOPE_API_KEY") else None
        click_healer = HealPolicy(
            locator_store_path="core/heal/locator_store.yaml",
            ai_provider=ai_provider,
            accept_threshold=0.72,
        )
        adapter.start()
        executor = Executor(
            adapter=adapter,
            step_runner=step_runner,
            perception=perception,
            click_healer=click_healer,
        )
        return executor, adapter.quit

    return _factory


def main():
    parser = argparse.ArgumentParser(description="Replay an ActionPlan file or the latest run actions.json and collect evidence.")
    parser.add_argument("--plan", default="", help="Path to plan file (json/yaml).")
    parser.add_argument("--source-run-id", default="", help="Replay from evidence/<run_id>/actions.json")
    parser.add_argument("--latest", action="store_true", help="Replay the latest run that contains actions.json")
    parser.add_argument("--actions-filename", default="actions.json", help="Actions filename under run dir. Default: actions.json")
    parser.add_argument("--device-yaml", default="configs/device.yaml", help="Device config yaml path. Default: configs/device.yaml")
    parser.add_argument("--times", type=int, default=1, help="How many times to replay. Default: 1")
    parser.add_argument("--batch-base-dir", default="", help="Batch report output dir. Default: <evidence_dir>/_batches")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Sleep seconds between replay runs. Default: 0")
    parser.add_argument("--stop-on-first-failure", action="store_true", help="Stop batch replay when the first failure happens.")
    args = parser.parse_args()

    cfg = load_yaml(args.device_yaml)
    evidence_dir = cfg.get("runtime", {}).get("evidence_dir", "evidence")
    batch_base_dir = args.batch_base_dir or str(Path(evidence_dir) / "_batches")

    source_desc = ""
    plan_path: Path | None = None
    source_run_id = "external"
    if args.plan:
        plan_path = Path(args.plan)
        if not plan_path.exists():
            print(f"[replay] plan not found: {plan_path}", file=sys.stderr)
            return 2
        source_desc = str(plan_path)
    elif args.source_run_id:
        source_run_id = args.source_run_id
        source_desc = f"evidence/{source_run_id}/{args.actions_filename}"
    else:
        args.latest = True
        try:
            latest = find_latest_run_with_actions(base_dir=evidence_dir, actions_filename=args.actions_filename)
        except Exception as e:
            print(f"[replay] latest run resolve failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        source_run_id = latest.name
        source_desc = str(latest / args.actions_filename)

    executor_factory = build_executor_factory(args.device_yaml)

    try:
        if args.times > 1:
            if plan_path is not None:
                report = replay_plan_batch(
                    executor_factory,
                    plan_path,
                    times=args.times,
                    source_run_id=source_run_id,
                    batch_base_dir=batch_base_dir,
                    sleep_s=args.sleep_s,
                    stop_on_first_failure=args.stop_on_first_failure,
                )
            elif args.source_run_id:
                report = replay_run_batch(
                    executor_factory,
                    Path(evidence_dir) / args.source_run_id,
                    actions_filename=args.actions_filename,
                    times=args.times,
                    batch_base_dir=batch_base_dir,
                    sleep_s=args.sleep_s,
                    stop_on_first_failure=args.stop_on_first_failure,
                )
            else:
                report = replay_latest_run_batch(
                    executor_factory,
                    base_dir=evidence_dir,
                    actions_filename=args.actions_filename,
                    times=args.times,
                    batch_base_dir=batch_base_dir,
                    sleep_s=args.sleep_s,
                    stop_on_first_failure=args.stop_on_first_failure,
                )

            print(
                f"[replay-batch] source={source_desc} total={report.total_runs} passed={report.passed_runs} "
                f"failed={report.failed_runs} success_rate={report.success_rate:.2f}% unstable={report.unstable}"
            )
            print(f"[replay-batch] report_json={report.json_report_path}")
            print(f"[replay-batch] report_md={report.md_report_path}")
            return 0 if report.failed_runs == 0 else 1

        executor, cleanup = executor_factory()
        try:
            if plan_path is not None:
                outcome = replay_plan_file(executor, plan_path, source_run_id=source_run_id)
            elif args.source_run_id:
                outcome = replay_run(executor, Path(evidence_dir) / args.source_run_id, actions_filename=args.actions_filename)
            else:
                outcome = replay_latest_run(executor, base_dir=evidence_dir, actions_filename=args.actions_filename)
        finally:
            cleanup()

        if outcome.ok:
            print(f"[replay] OK source={source_desc} replay_run_id={outcome.replay_run_id} executed={outcome.executed}")
            print(f"[replay] summary={outcome.summary_path}")
            return 0

        print(f"[replay] FAIL source={source_desc} replay_run_id={outcome.replay_run_id} error={outcome.error_type}: {outcome.error_message}", file=sys.stderr)
        if outcome.failure is not None:
            print(f"[replay] category={outcome.failure.category} reason={outcome.failure.reason}", file=sys.stderr)
        print(f"[replay] summary={outcome.summary_path}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"[replay] FAIL error={type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
