from __future__ import annotations

import importlib
from typing import List, Optional

from core.runtime.project_loader import load_project_profile
from core.state.rules.base import Rule


class StateRegistryError(RuntimeError):
    pass


def _import_object(path: str):
    module_name, _, attr = path.rpartition('.')
    if not module_name or not attr:
        raise StateRegistryError(f"Invalid rule import path: {path}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise StateRegistryError(f"Rule class not found: {path}") from e


def build_rules_for_project(project_id: Optional[str] = None) -> List[Rule]:
    profile = load_project_profile(project_id)
    rules: List[Rule] = []
    for path in profile.rule_paths:
        obj = _import_object(path)
        try:
            inst = obj()
        except Exception as e:
            raise StateRegistryError(f"Failed to instantiate rule: {path}: {type(e).__name__}: {e}") from e
        if not isinstance(inst, Rule):
            raise StateRegistryError(f"Imported object is not a Rule: {path}")
        rules.append(inst)
    return rules
