# core/executor/plan_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  

from core.executor.action_schema import ActionPlan
from typing import Any, Union



class PlanLoadError(RuntimeError):
    pass


def load_plan(path: str | Path) -> ActionPlan:
    """
    从文件读取 ActionPlan（JSON/YAML），并做 schema 校验。
    - path: e.g. tests/plans/wechat_smoke.json
    """
    p = Path(path)
    if not p.exists():
        raise PlanLoadError(f"Plan file not found: {p}")

    suffix = p.suffix.lower()
    try:
        if suffix in [".json"]:
            data = json.loads(p.read_text(encoding="utf-8"))
        elif suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            raise PlanLoadError(f"Unsupported plan format: {suffix} (only .json/.yaml/.yml)")
    except Exception as e:
        raise PlanLoadError(f"Failed to parse plan file: {p} ({type(e).__name__}: {e})") from e

    if not isinstance(data, dict):
        raise PlanLoadError(f"Plan root must be an object/dict, got: {type(data).__name__}")

    try:
        # pydantic v2 校验
        return ActionPlan.model_validate(data)
    except Exception as e:
        raise PlanLoadError(f"Plan schema validation failed: {p} ({type(e).__name__}: {e})") from e

# core/executor/plan_loader.py (追加内容)

def dump_plan(plan: Union[ActionPlan, dict, str], path: str | Path, *, indent: int = 2) -> str:
    """
    将 ActionPlan / dict / JSON字符串 保存到文件（json 格式）
    - plan: ActionPlan | dict | json_str
    - path: 输出路径
    返回：最终写入的文件绝对路径字符串
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(plan, ActionPlan):
        data: Any = plan.model_dump()
    elif isinstance(plan, dict):
        data = plan
    elif isinstance(plan, str):
        # 允许直接传 AI 产出的 JSON 文本
        data = json.loads(plan)
    else:
        raise TypeError(f"Unsupported plan type: {type(plan).__name__}")

    p.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    return str(p.resolve())