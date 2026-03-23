# core/policy/ui_vocab.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class UiVocabError(RuntimeError):
    pass


@dataclass
class UiVocab:
    mapping: Dict[str, List[str]]

    @staticmethod
    def load(path: str = "assets/ui_vocab.yaml") -> "UiVocab":
        p = Path(path)
        if not p.exists():
            raise UiVocabError(f"ui_vocab file not found: {p}")

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise UiVocabError("ui_vocab.yaml root must be a mapping/dict")

        mapping: Dict[str, List[str]] = {}
        for k, v in data.items():
            if not isinstance(k, str) or not k.strip():
                continue
            if isinstance(v, list):
                texts = [str(x).strip() for x in v if str(x).strip()]
                if texts:
                    mapping[k.strip()] = texts
            elif isinstance(v, str) and v.strip():
                mapping[k.strip()] = [v.strip()]

        return UiVocab(mapping=mapping)

    def expand_key(self, key: str) -> List[str]:
        key = (key or "").strip()
        if not key:
            return []
        return self.mapping.get(key, [])


class UiVocabExpander:
    """
    将 goal dict 中的 target_key / open_key / option_key 等展开成 target_text+synonyms。
    设计为“可选”：goal 不写 key 时不做任何处理。
    """

    def __init__(self, vocab: UiVocab):
        self.vocab = vocab

    def expand_goal(self, goal_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        输入：可能包含 target_key 的 goal dict
        输出：展开后的 goal dict（不修改原 dict）
        目前只实现 CLICK_TEXT 的 target_key（保持最小闭环）
        """
        goal = dict(goal_dict)

        intent = goal.get("intent")
        if intent == "CLICK_TEXT":
            self._expand_click_text(goal)

        return goal

    def _expand_click_text(self, goal: Dict[str, Any]) -> None:
        # 用例可以写：target_key: "测糖入口"
        key = (goal.get("target_key") or "").strip()
        if not key:
            return

        texts = self.vocab.expand_key(key)
        if not texts:
            # 不抛错也行，但建议抛错，避免 silently 失效
            raise UiVocabError(f"target_key not found in ui_vocab: {key}")

        # 约定：第一个作为主 target_text，其余作为 synonyms
        goal.setdefault("target_text", texts[0])
        syn = list(goal.get("synonyms") or [])
        # 合并去重（保持顺序）
        for t in texts[1:]:
            if t not in syn and t != goal["target_text"]:
                syn.append(t)
        goal["synonyms"] = syn

        # 清理 key（可选）
        # goal.pop("target_key", None)