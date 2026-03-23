# core/policy/policy_runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from core.perception.perception import PerceptionPack
from core.policy.qwen_client import QwenClient, QwenResponse
from core.policy.parser import ParsedPolicy, parse_policy_output, PolicyParseError
from core.policy.goal_schema import GoalEnvelope, Goal


@dataclass
class PolicyRunnerConfig:
    prompt_path: str = "assets/prompts/policy_v1.txt"
    max_tokens: int = 512
    temperature: float = 0.0


class PolicyRunner:
    """
    - goal 强校验（GoalSchema）
    - 归档 policy 输入/输出到 evidence/<run_id>/policy/
    """

    def __init__(
        self,
        qwen: QwenClient,
        evidence_manager,  # EvidenceManager（Day2），只用它的 run.run_dir
        cfg: Optional[PolicyRunnerConfig] = None,
    ):
        self.qwen = qwen
        self.evidence = evidence_manager
        self.cfg = cfg or PolicyRunnerConfig()

    def _read_prompt(self) -> str:
        p = Path(self.cfg.prompt_path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        return p.read_text(encoding="utf-8")

    def _policy_dir(self) -> Path:
        d = Path(self.evidence.run.run_dir) / "policy"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "", encoding="utf-8")

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def decide_next(
        self,
        pack: PerceptionPack,
        goal: Union[Goal, Dict[str, Any]],
        recent: Optional[Dict[str, Any]] = None,
        hints: Optional[Dict[str, Any]] = None,
    ) -> ParsedPolicy:
        """
        goal：强制使用 GoalSchema
        - 你可以传 Goal 对象
        - 也可以传 dict（会被 GoalEnvelope 强校验）
        """
        prompt = self._read_prompt()

        # ===== validate/normalize goal =====
        if isinstance(goal, dict):
            env = GoalEnvelope.from_any(goal)
        else:
            env = GoalEnvelope.model_validate({"goal": goal})
        goal_norm = env.goal.model_dump()

        payload = {
            "goal": goal_norm,
            "perception": {
                "image_path": pack.image_path,
                "ocr_text": pack.ocr_text,
                "meta": pack.meta,
            },
            "recent": recent or {},
            "hints": hints or {},
        }
        user_payload = json.dumps(payload, ensure_ascii=False)

        resp: Optional[QwenResponse] = None
        parsed: Optional[ParsedPolicy] = None
        err: Optional[str] = None

        try:
            resp = self.qwen.chat(
                system_prompt=prompt,
                user_payload=user_payload,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            parsed = parse_policy_output(resp.content)
        except PolicyParseError as e:
            err = f"PolicyParseError: {e}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        # ===== archive =====
        out_dir = self._policy_dir()
        self._write_text(out_dir / "policy_prompt.txt", prompt)
        self._write_json(out_dir / "policy_input.json", payload)

        if resp is not None:
            self._write_text(out_dir / "policy_raw.txt", resp.content)
            self._write_json(out_dir / "policy_response_meta.json", {
                "elapsed_ms": resp.elapsed_ms,
                "model": resp.raw.get("model", None),
                "id": resp.raw.get("id", None),
            })
        else:
            self._write_text(out_dir / "policy_raw.txt", "")

        if parsed is not None:
            parsed_json: Dict[str, Any] = {
                "kind": parsed.kind,
                "reason": parsed.reason,
                "raw_json": parsed.raw_json,
            }
            if parsed.plan is not None:
                parsed_json["plan"] = parsed.plan.model_dump()
            self._write_json(out_dir / "policy_parsed.json", parsed_json)
            return parsed

        # 失败默认 stop（安全）
        self._write_json(out_dir / "policy_parsed.json", {
            "kind": "stop",
            "reason": err or "unknown error",
        })
        return ParsedPolicy(kind="stop", reason=err or "unknown error")