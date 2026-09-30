"""
judge.py — LLM-as-judge scoring, with caching.

"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("judge")


JUDGE_PROMPT = """You are grading an answer to a question about a long conversation.

QUESTION: {question}
CORRECT ANSWER: {gold}
STUDENT ANSWER: {prediction}

Mark the student answer CORRECT if it conveys the same information as the
correct answer. Ignore differences in wording, phrasing, extra detail, date
format, or verbosity. A correct answer that adds true context is still CORRECT.

Mark it WRONG if it states something different, contradicts the correct answer,
omits a required part of it, or declines to answer.

Output ONLY JSON: {{"verdict": "CORRECT" or "WRONG", "reason": "<few words>"}}"""


#: Token budgets tried in order. See Judge.score for why this cannot be small.
JUDGE_BUDGETS = (512, 1536)
JUDGE_VERSION = "v2"


class JudgeUnavailable(RuntimeError):
    """The judge produced no verdict. Never to be scored as WRONG."""


class Judge:
    """LLM judge with an on-disk cache."""

    def __init__(self, llm: Any, cache_path: Optional[str] = None,
                 model_tag: str = "judge"):
        self.llm = llm
        self.cache_path = cache_path
        self.model_tag = model_tag
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._dirty = 0
        self.unjudged = 0
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    # -- cache -----------------------------------------------------------

    @staticmethod
    def _key(question: str, gold: str, prediction: str, tag: str) -> str:
        raw = f"{JUDGE_VERSION}␟{tag}␟{question}␟{gold}␟{prediction}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def save(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with self._lock:
            # Atomic: other processes read these caches while this one writes.
            tmp = f"{self.cache_path}.tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(self._cache, f)
            os.replace(tmp, self.cache_path)
            self._dirty = 0

    def absorb(self, path: str) -> int:
        """Merge another cache file's verdicts in (read-only). Returns entries added."""
        try:
            with open(path) as f:
                other = json.load(f)
        except Exception:
            return 0
        with self._lock:
            before = len(self._cache)
            for k, v in other.items():
                self._cache.setdefault(k, v)
            return len(self._cache) - before

    # -- scoring ---------------------------------------------------------

    def _caller(self):
        llm = self.llm
        if llm is None:
            return None
        if hasattr(llm, "get_completion"):
            return llm
        if hasattr(llm, "llm") and hasattr(llm.llm, "get_completion"):
            return llm.llm
        return None

    @staticmethod
    def _parse(raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        t = raw.strip()
        if "```" in t:
            t = re.sub(r"```(?:json)?\s*", "", t).replace("```", "").strip()
        start = t.find("{")
        if start >= 0:
            depth = 0
            for i, ch in enumerate(t[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except json.JSONDecodeError:
                            break
        # Fall back to looking for the verdict word directly.
        up = t.upper()
        if "CORRECT" in up and "WRONG" not in up:
            return {"verdict": "CORRECT", "reason": "parsed from text"}
        if "WRONG" in up:
            return {"verdict": "WRONG", "reason": "parsed from text"}
        return None

    def score(self, question: str, gold: Any, prediction: Any) -> float:
        gold = "" if gold is None else str(gold)
        prediction = "" if prediction is None else str(prediction)

        if not prediction.strip():
            return 0.0                      # a refusal is never correct
        if not gold.strip():
            return 0.0

        key = self._key(question, gold, prediction, self.model_tag)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return float(hit["score"])

        caller = self._caller()
        if caller is None:
            return 0.0
        parsed = None
        raw = ""
        for budget in JUDGE_BUDGETS:
            raw = caller.get_completion(
                JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction),
                temperature=0.0, max_tokens=budget, role="judge",
            )
            parsed = self._parse(raw or "")
            if parsed is not None:
                break
        if parsed is None:
            self.unjudged += 1
            raise JudgeUnavailable(
                f"judge returned no parseable verdict after budgets "
                f"{JUDGE_BUDGETS}; raw={str(raw)[:80]!r}")

        verdict = parsed.get("verdict", "")
        score = 1.0 if str(verdict).upper().startswith("CORRECT") else 0.0

        with self._lock:
            self._cache[key] = {"score": score,
                                "reason": str((parsed or {}).get("reason", ""))[:120]}
            self._dirty += 1
            dirty = self._dirty
        if self.cache_path and dirty >= 50:
            self.save()
        return score

    def score_many(self, items: List[Tuple[str, Any, Any]],
                   progress_every: int = 50) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for i, (q, g, p) in enumerate(items, start=1):
            try:
                out.append(self.score(q, g, p))
            except JudgeUnavailable as e:
                logger.warning("unjudged item %d: %s", i, e)
                out.append(None)
            if progress_every and i % progress_every == 0:
                logger.info("  judged %d/%d", i, len(items))
        self.save()
        return out

    @property
    def cache_size(self) -> int:
        return len(self._cache)
