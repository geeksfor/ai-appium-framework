# core/executor/plan_archive.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Union
import json

from core.executor.action_schema import ActionPlan
from core.executor.plan_loader import dump_plan


def archive_plan_to_evidence(
    evidence_manager,
    plan: Union[ActionPlan, dict, str],
    filename: str = "ai_plan.json",
) -> str:
    """
    将 plan 归档到 evidence/<run_id>/<filename>
    - evidence_manager: Day2 的 EvidenceManager 实例（含 run.run_dir）
    - plan: ActionPlan | dict | json_str
    返回：写入文件的绝对路径字符串
    """
    run_dir: Path = evidence_manager.run.run_dir
    out_path = run_dir / filename
    return dump_plan(plan, out_path)


def archive_raw_text_to_evidence(
    evidence_manager,
    text: str,
    filename: str = "ai_raw_output.txt",
) -> str:
    """
    将 AI 原始文本（可能不是合法 JSON）归档到 evidence/<run_id>/<filename>
    返回：写入文件绝对路径字符串
    """
    run_dir: Path = evidence_manager.run.run_dir
    out_path = run_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return str(out_path.resolve())


def archive_json_to_evidence(
    evidence_manager,
    data: Any,
    filename: str = "ai_meta.json",
    indent: int = 2,
) -> str:
    """
    将任意 JSON 数据归档到 evidence/<run_id>/<filename>
    """
    run_dir: Path = evidence_manager.run.run_dir
    out_path = run_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    return str(out_path.resolve())