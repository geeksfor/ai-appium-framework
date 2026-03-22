# core/state/rules/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class MatchResult:
    state: str
    score: float
    hits: List[str]
    misses: List[str]
    reason: str
    meta: Dict[str, Any]


@dataclass
class Region:
    """
    预留：区域（百分比坐标），后续如果你做 ROI OCR / bbox OCR，这里可生效
    """
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0
    name: str = "full"


class Rule:
    """
    每条规则 = 关键字特征（all/any/negative） + 阈值 +（可选）二次确认配置
    """

    # ====== 必填/常用 ======
    state_name: str = "Unknown"
    region: Region = Region()

    keywords_all: List[str] = []
    keywords_any: List[str] = []
    negative_keywords: List[str] = []

    # 达到阈值才算“命中候选”
    min_score: float = 0.6

    # ====== Day5.1：页面类二次确认（默认不启用，兼容旧规则）======
    # 命中其中任意一个词即可通过二次确认
    confirm_any: List[str] = []
    # keywords_any 至少命中 N 个才通过二次确认
    confirm_min_any_hits: int = 0
    # 命中这些词则二次确认失败
    confirm_not_contains: List[str] = []

    def score(self, text: str) -> MatchResult:
        """
        对 OCR 文本做关键词匹配打分（Day5：全文匹配；Day6 可扩展 ROI 匹配）
        - hits：命中的关键词（all + any）
        - misses：未命中的 keywords_all（用于解释）
        - meta：包含 all_hits/any_hits/neg_hits 等，方便 state_machine 做二次确认与调参
        """
        # 1) normalize：合并换行/多空格，提升包含匹配稳定性
        t = " ".join((text or "").split())
        t_low = t.lower()

        # 2) negative 命中统计（命中则强扣分）
        neg_hits: List[str] = []
        for nk in self.negative_keywords:
            if nk and nk.lower() in t_low:
                neg_hits.append(nk)

        # ==========================================================
        # ✅ 你问的三行，就放在这里（negative 之后、all/any 逻辑之前）
        # ==========================================================
        all_hits = [kw for kw in self.keywords_all if kw and kw.lower() in t_low]
        any_hits = [kw for kw in self.keywords_any if kw and kw.lower() in t_low]
        hits = all_hits + any_hits
        # ==========================================================

        # 3) misses（用于解释：哪些 all 没命中）
        misses: List[str] = [kw for kw in self.keywords_all if kw and kw.lower() not in t_low]

        # 4) all/any 逻辑判定（命中条件）
        all_ok = True
        if self.keywords_all:
            all_ok = len(all_hits) == len([k for k in self.keywords_all if k])

        any_ok = True
        any_hit = True
        if self.keywords_any:
            any_hit = len(any_hits) > 0
            any_ok = any_hit

        # 5) 计算分数：all 占 0.7，any 占 0.3，neg 命中强扣
        score = 0.0

        # all 部分
        if self.keywords_all:
            denom = max(1, len([k for k in self.keywords_all if k]))
            score += 0.7 * (len(all_hits) / denom)
        else:
            score += 0.7

        # any 部分
        if self.keywords_any:
            score += 0.3 * (1.0 if any_hit else 0.0)
        else:
            score += 0.3

        # negative 强扣
        if neg_hits:
            score *= 0.2

        # 6) 是否“命中候选”（注意：最终是否短路由 StateMachine 决定）
        ok_candidate = all_ok and any_ok and score >= self.min_score

        reason = f"all_ok={all_ok} any_ok={any_ok} neg_hits={neg_hits}"
        meta: Dict[str, Any] = {
            "region": self.region.__dict__,
            "neg_hits": neg_hits,
            "all_hits": all_hits,
            "any_hits": any_hits,
            "ok_candidate": ok_candidate,
            # 二次确认配置也带出去，便于 debug
            "confirm_any": self.confirm_any,
            "confirm_min_any_hits": self.confirm_min_any_hits,
            "confirm_not_contains": self.confirm_not_contains,
        }

        return MatchResult(
            state=self.state_name,
            score=score,
            hits=hits,
            misses=misses,
            reason=reason,
            meta=meta,
        )