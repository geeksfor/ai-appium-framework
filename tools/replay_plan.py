# 用法 python tools/replay_plan.py --plan evidence/20260319_232617_9a470c/ai_plan.json
#!/usr/bin/env python3
# tools/replay_plan.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.driver.appium_adapter import AppiumAdapter
from core.executor.executor import Executor
from core.executor.plan_loader import load_plan
from core.report.evidence import EvidenceManager
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


def main():
    parser = argparse.ArgumentParser(description="Replay an ActionPlan file and collect evidence.")
    parser.add_argument(
        "--plan",
        required=True,
        help="Path to plan file (json/yaml). Example: evidence/<run_id>/ai_plan.json",
    )
    parser.add_argument(
        "--device-yaml",
        default="configs/device.yaml",
        help="Device config yaml path. Default: configs/device.yaml",
    )
    parser.add_argument(
        "--reuse-run-id",
        default="",
        help="If provided, reuse this run_id instead of creating a new one (NOT recommended).",
    )

    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"[replay] plan not found: {plan_path}", file=sys.stderr)
        return 2

    # 1) load & validate plan
    plan = load_plan(plan_path)

    # 2) build adapter
    adapter = build_adapter_from_device_yaml(args.device_yaml)

    # 3) evidence manager
    # 默认创建新 run_id，避免覆盖旧 evidence
    evidence_dir = load_yaml(args.device_yaml).get("runtime", {}).get("evidence_dir", "evidence")
    evidence = EvidenceManager(base_dir=evidence_dir, run_id=args.reuse_run_id or None)

    # 4) step runner + executor
    step_logger = StepLogger()
    step_runner = StepRunner(evidence=evidence, step_logger=step_logger, driver_getter=lambda: adapter.driver)

    try:
        adapter.start()
        exe = Executor(adapter=adapter, step_runner=step_runner)

        result = exe.run_plan(plan)
        print(f"[replay] OK executed={result.executed} run_id={evidence.run.run_id}")
        return 0

    except Exception as e:
        print(f"[replay] FAIL run_id={evidence.run.run_id} error={type(e).__name__}: {e}", file=sys.stderr)
        return 1

    finally:
        adapter.quit()


if __name__ == "__main__":
    raise SystemExit(main())