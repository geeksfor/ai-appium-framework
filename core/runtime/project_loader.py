from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.utils.config_loader import load_yaml


@dataclass
class ProjectProfile:
    project_id: str
    app_package: str = ""
    default_home_state: str = ""
    state_aliases: Dict[str, str] = field(default_factory=dict)
    rule_paths: List[str] = field(default_factory=list)
    semantic_targets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    risk_words: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def resolve_state(self, name: str) -> str:
        key = str(name or "").strip()
        if not key:
            return key
        return self.state_aliases.get(key, key)

    def semantic_target_aliases(self, logical_name: str) -> List[str]:
        item = self.semantic_targets.get(logical_name, {}) or {}
        aliases = item.get("aliases", []) or []
        out: List[str] = []
        for x in aliases:
            s = str(x).strip()
            if s:
                out.append(s)
        return out


DEFAULT_PROJECT_ID = "wechat"


def _project_file(project_id: str, base_dir: str = "projects") -> Path:
    return Path(base_dir) / project_id / "project.yaml"


def load_project_profile(project_id: Optional[str] = None, base_dir: str = "projects") -> ProjectProfile:
    pid = (project_id or os.getenv("TEST_PROJECT") or DEFAULT_PROJECT_ID).strip()
    p = _project_file(pid, base_dir=base_dir)
    if not p.exists():
        raise FileNotFoundError(f"Project profile not found: {p}")

    data = load_yaml(str(p))
    return ProjectProfile(
        project_id=str(data.get("project_id") or pid),
        app_package=str(data.get("app_package") or "").strip(),
        default_home_state=str(data.get("default_home_state") or "").strip(),
        state_aliases={str(k): str(v) for k, v in (data.get("state_aliases") or {}).items()},
        rule_paths=[str(x).strip() for x in (data.get("rules") or []) if str(x).strip()],
        semantic_targets=dict(data.get("semantic_targets") or {}),
        risk_words=[str(x).strip() for x in (data.get("risk_words") or []) if str(x).strip()],
        raw=data,
    )
