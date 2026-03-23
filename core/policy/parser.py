# core/policy/parser.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from core.executor.action_schema import ActionPlan


@dataclass
class ParsedPolicy:
    kind: str  # "plan" | "ask_human" | "stop"
    plan: Optional[ActionPlan] = None
    reason: Optional[str] = None
    raw_json: Optional[Dict[str, Any]] = None


class PolicyParseError(RuntimeError):
    pass


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_FIRST_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _extract_json_text(s: str) -> str:
    s = (s or "").strip()

    # 1) code fence ```json ... ```
    m = _JSON_BLOCK_RE.search(s)
    if m:
        return m.group(1).strip()

    # 2) try find first { ... } block
    m2 = _FIRST_JSON_OBJ_RE.search(s)
    if m2:
        return m2.group(1).strip()

    # 3) fallback: whole string
    return s


def parse_policy_output(model_text: str) -> ParsedPolicy:
    """
    Accept either:
    A) {"actions":[...]}  -> ActionPlan
    B) {"type":"ASK_HUMAN","reason":"..."}
    C) {"type":"SCREENSHOT_AND_STOP","reason":"..."}
    """
    jtxt = _extract_json_text(model_text)

    try:
        obj = json.loads(jtxt)
    except Exception as e:
        raise PolicyParseError(f"Invalid JSON from model: {type(e).__name__}: {e}\nraw={model_text[:500]}") from e

    if not isinstance(obj, dict):
        raise PolicyParseError(f"Model JSON must be an object/dict, got {type(obj).__name__}")

    # Case A: plan
    if "actions" in obj:
        try:
            plan = ActionPlan.model_validate(obj)
            return ParsedPolicy(kind="plan", plan=plan, raw_json=obj)
        except Exception as e:
            raise PolicyParseError(f"ActionPlan validation failed: {type(e).__name__}: {e}\njson={obj}") from e

    # Case B/C: special
    t = (obj.get("type") or "").strip()
    reason = (obj.get("reason") or "").strip()

    if t == "ASK_HUMAN":
        return ParsedPolicy(kind="ask_human", reason=reason or "model asked human", raw_json=obj)

    if t in ("SCREENSHOT_AND_STOP", "STOP"):
        return ParsedPolicy(kind="stop", reason=reason or "model requested stop", raw_json=obj)

    # If unknown format, treat as error
    raise PolicyParseError(f"Unknown policy JSON schema: {obj}")