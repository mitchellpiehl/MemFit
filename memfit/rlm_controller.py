"""
rlm_controller.py — MemFit retrieval 
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from indexes import minmax, rrf, get_default_reranker
from memory_layer import (
    LeafNode,
    MemoryConfig,
    MemoryObject,
    RecursiveMemoryLayer,
    RegionNode,
    _fuzzy_token_overlap,
)

logger = logging.getLogger("rlm_controller")

@dataclass
class RLMConfig:
    # -- retrieval breadth
    scan_top_k: int = 30
    top_k_episodes: int = 10     # episodes handed to the reader
    max_plan_steps: int = 3

    # -- scoring (Phase 2 step A)
    lexical_scorer: str = "bm25"
    fusion: str = "convex"
    alpha: float = 0.7
    rrf_k: int = 60

    # -- reranking
    rerank: bool = True
    rerank_pool: int = 40        # candidates scored by the cross-encoder

    # -- LLM
    planner_temperature: float = 0.1
    reader_temperature: float = 0.0
    planner_max_tokens: int = 512
    reader_max_tokens: int = 256

    # -- ablation switches
    enable_planner: bool = False
    planner_style: str = "conservative"
    reader_question: str = "original"
    reader_prompt: str = "uniform"
    reader_reason_max_tokens: int = 768
    context_window: int = 0
    pool_strategy: str = "roundrobin"
    semantic_query: str = "subquestion"   # "subquestion" | "original"
    resolve_relative_dates: bool = True
    system_prompt: str = ""
    temporal_grounding: bool = True
    # depth of the ranked list kept for retrieval measurement only
    # (never shown to the reader; see RLMController.retrieve)
    measure_depth: int = 50
    reader_confidence: bool = False
    rerank_normalise: bool = True

    # -- candidate expansion (LLM-free)
    expand_summaries: int = 0
    prf_docs: int = 0
    prf_terms: int = 12

    # -- metadata boosts (paper values)
    boost_speaker: float = 0.10
    boost_year: float = 0.05
    boost_month: float = 0.08

    # -- notes (Phase 3)
    use_notes: bool = False
    note_top_k: int = 10         # note candidates entering the pool
    note_slots: int = 0          # pack slots reserved for notes (0 = compete)
    note_mode: str = "compete"
    note_additive_max: int = 5

    # -- multimodal
    image_caption_top_k: int = 5
    caption_weight: float = 0.4

def _resolve_llm(llm: Any) -> Any:
    if llm is None:
        return None
    if hasattr(llm, "get_completion"):
        return llm
    if hasattr(llm, "llm") and hasattr(llm.llm, "get_completion"):
        return llm.llm
    return None


def _resolve_mllm(llm: Any) -> Any:
    if llm is None:
        return None
    if hasattr(llm, "get_image_completion"):
        return llm
    if hasattr(llm, "llm") and hasattr(llm.llm, "get_image_completion"):
        return llm.llm
    return None


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if "```" in t:
        t = re.sub(r"```(?:json)?\s*", "", t).replace("```", "").strip()
    return t


def _first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced {...} object from a model response."""
    t = _strip_code_fence(text)
    start = t.find("{")
    if start < 0:
        return None
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
                    return None
    return None


def _clean_answer(text: str) -> str:
    s = _strip_code_fence(text).strip()
    obj = _first_json_object(s) if s.startswith("{") else None
    if obj:
        for key in ("answer", "Answer", "ANSWER"):
            if key in obj and isinstance(obj[key], str):
                return obj[key].strip()
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    # Drop a leading "ANSWER:" label if the model echoed it.
    s = re.sub(r"^(answer|final answer)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    return s


def _final_answer_line(text: str) -> str:
    s = _strip_code_fence(text or "").strip()
    hits = list(re.finditer(r"(?:final\s+)?answer\s*:", s, flags=re.IGNORECASE))
    if hits:
        tail = s[hits[-1].end():].strip()
        tail = tail.splitlines()[0].strip() if tail else ""
        return _clean_answer(tail.strip("*").strip())
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    return _clean_answer(lines[-1]) if lines else ""


_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
_WORD_NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_session_date(session_date: Optional[str]) -> Optional[datetime]:
    """Parse an ISO date, or a LoCoMo stamp like '1:56 pm on 8 May, 2023'."""
    if not session_date:
        return None
    s = session_date.strip()
    try:
        return datetime.fromisoformat(s[:19])
    except (ValueError, TypeError):
        pass
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", s)
    if m:
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(
                    f"{m.group(1)} {m.group(2)} {m.group(3)}", fmt
                )
            except ValueError:
                continue
    return None


def format_date(d: datetime) -> str:
    """'7 May 2023' — the surface form LoCoMo gold answers use."""
    return f"{d.day} {d.strftime('%B %Y')}"


def days_before(d: datetime, now: datetime) -> str:
    """'10 days before the current date': elapsed time as a number to read."""
    n = (now.date() - d.date()).days
    if n == 0:
        return "the current date"
    unit = "day" if abs(n) == 1 else "days"
    return (f"{n} {unit} before the current date" if n > 0
            else f"{-n} {unit} after the current date")


def _resolve_relative_dates(text: str, session_date: str,
                            now: Optional[datetime] = None) -> str:
    anchor = parse_session_date(session_date)
    if anchor is None:
        return text

    def sub(pattern: str, repl) -> None:
        nonlocal text
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    def annotate(m: "re.Match", value: str) -> str:
        return f"{m.group(0)} ({value})"

    def day(d: datetime) -> str:
        return format_date(d) + (f", {days_before(d, now)}" if now is not None else "")

    # -- day level -------------------------------------------------------
    sub(r"\bthe day before yesterday\b",
        lambda m: annotate(m, day(anchor - timedelta(days=2))))
    sub(r"\byesterday\b",
        lambda m: annotate(m, day(anchor - timedelta(days=1))))
    sub(r"\btoday\b", lambda m: annotate(m, day(anchor)))
    sub(r"\btomorrow\b",
        lambda m: annotate(m, day(anchor + timedelta(days=1))))

    def _n(word: str) -> int:
        w = word.lower()
        if w in _WORD_NUM:
            return _WORD_NUM[w]
        try:
            return int(w)
        except ValueError:
            return 0

    _NUM = r"(\d+|two|three|four|five|six|seven|eight|nine|ten)"

    def days_ago(m: "re.Match") -> str:
        n = _n(m.group(1))
        return annotate(m, day(anchor - timedelta(days=n))) if n else m.group(0)

    sub(rf"\b{_NUM}\s+days?\s+ago\b", days_ago)

    def weeks_ago(m: "re.Match") -> str:
        n = _n(m.group(1))
        return annotate(m, day(anchor - timedelta(weeks=n))) if n else m.group(0)

    sub(rf"\b{_NUM}\s+weeks?\s+ago\b", weeks_ago)

    # -- week level ------------------------------------------------------
    sub(r"\blast week\b",
        lambda m: annotate(m, f"the week before {format_date(anchor)}"))
    sub(r"\bthis week\b",
        lambda m: annotate(m, f"the week of {format_date(anchor)}"))

    def last_weekday(m: "re.Match") -> str:
        target = _WEEKDAYS.get(m.group(1).lower())
        if target is None:
            return m.group(0)
        back = (anchor.weekday() - target) % 7 or 7
        resolved = anchor - timedelta(days=back)
        return annotate(
            m, f"the {m.group(1)} before {format_date(anchor)}, "
               f"i.e. {day(resolved)}"
        )

    sub(r"\blast\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        last_weekday)

    # -- month level -----------------------------------------------------
    def shift_months(d: datetime, k: int) -> datetime:
        month = d.month - k
        year = d.year
        while month <= 0:
            month += 12
            year -= 1
        while month > 12:
            month -= 12
            year += 1
        return d.replace(year=year, month=month, day=1)

    sub(r"\blast month\b",
        lambda m: annotate(m, shift_months(anchor, 1).strftime("%B %Y")))
    sub(r"\bthis month\b", lambda m: annotate(m, anchor.strftime("%B %Y")))
    sub(r"\bnext month\b",
        lambda m: annotate(m, shift_months(anchor, -1).strftime("%B %Y")))

    def months_ago(m: "re.Match") -> str:
        n = _n(m.group(1))
        return annotate(m, shift_months(anchor, n).strftime("%B %Y")) if n else m.group(0)

    sub(rf"\b{_NUM}\s+months?\s+ago\b", months_ago)

    # -- year level ------------------------------------------------------
    sub(r"\blast year\b", lambda m: annotate(m, str(anchor.year - 1)))
    sub(r"\bthis year\b", lambda m: annotate(m, str(anchor.year)))
    sub(r"\bnext year\b", lambda m: annotate(m, str(anchor.year + 1)))

    def years_ago(m: "re.Match") -> str:
        n = _n(m.group(1))
        return annotate(m, str(anchor.year - n)) if n else m.group(0)

    sub(rf"\b{_NUM}\s+years?\s+ago\b", years_ago)
    return text


_Q_NUMBERS = {"a": 1, "an": 1, "one": 1, "a couple of": 2, "two": 2, "three": 3,
              "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "eleven": 11, "twelve": 12}
_Q_NUM = (r"(\d+|a couple of|an|a|one|two|three|four|five|six|seven|eight|nine|ten"
          r"|eleven|twelve)")


def _months_back(d: datetime, k: int) -> datetime:
    """The same day k calendar months earlier (clamped to the month's end)."""
    import calendar
    month, year = d.month - k, d.year
    while month <= 0:
        month += 12
        year -= 1
    return d.replace(year=year, month=month,
                     day=min(d.day, calendar.monthrange(year, month)[1]))


def resolve_question_time(question: str, now: datetime
                          ) -> Tuple[str, List[Tuple[Any, Any]]]:
    windows: List[Tuple[Any, Any]] = []
    out = question

    def qn(word: str) -> int:
        w = word.lower().strip()
        return int(w) if w.isdigit() else _Q_NUMBERS.get(w, 0)

    def sub(pattern: str, repl) -> None:
        nonlocal out
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    def window(lo: datetime, hi: datetime) -> None:
        w = (lo.date(), hi.date())
        if w not in windows:
            windows.append(w)

    # -- point references --------------------------------------------------
    def ago(m: "re.Match") -> str:
        n, unit = qn(m.group(1)), m.group(2).lower()
        if n <= 0:
            return m.group(0)
        if unit == "day":
            d = now - timedelta(days=n)
            window(d - timedelta(days=1), d + timedelta(days=1))
            return f"{m.group(0)} ({format_date(d)})"
        if unit == "week":
            d = now - timedelta(weeks=n)
            window(d - timedelta(days=3), d + timedelta(days=3))
            return f"{m.group(0)} (around {format_date(d)})"
        if unit == "month":
            d = _months_back(now, n)
            window(d - timedelta(days=15), d + timedelta(days=15))
            return f"{m.group(0)} (around {format_date(d)})"
        # years: annotated, but a year-wide window would select nothing
        d = _months_back(now, 12 * n)
        return f"{m.group(0)} (around {d.strftime('%B %Y')})"

    sub(rf"\b{_Q_NUM}\s+(day|week|month|year)s?\s+ago\b", ago)

    def last_weekday(m: "re.Match") -> str:
        target = _WEEKDAYS.get(m.group(1).lower())
        if target is None:
            return m.group(0)
        d = now - timedelta(days=(now.weekday() - target) % 7 or 7)
        window(d - timedelta(days=1), d + timedelta(days=1))
        return f"{m.group(0)} ({format_date(d)})"

    sub(r"\blast\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        last_weekday)

    def yesterday(m: "re.Match") -> str:
        d = now - timedelta(days=1)
        window(d - timedelta(days=1), d + timedelta(days=1))
        return f"{m.group(0)} ({format_date(d)})"

    sub(r"\byesterday\b", yesterday)

    # "last week/weekend/month/year" name the previous calendar period; "the
    # last month" (as in "in the last month") is a range, handled below.
    def last_week(m: "re.Match") -> str:
        mon = now - timedelta(days=now.weekday() + 7)
        sun = mon + timedelta(days=6)
        window(mon, sun)
        return f"{m.group(0)} ({format_date(mon)} to {format_date(sun)})"

    sub(r"(?<!the )\blast week\b", last_week)

    def last_weekend(m: "re.Match") -> str:
        since_sat = (now.weekday() - 5) % 7
        sat = now - timedelta(days=since_sat + (7 if since_sat <= 1 else 0))
        window(sat, sat + timedelta(days=1))
        return f"{m.group(0)} ({format_date(sat)} to {format_date(sat + timedelta(days=1))})"

    sub(r"(?<!the )\blast weekend\b", last_weekend)

    def last_month(m: "re.Match") -> str:
        last = now.replace(day=1) - timedelta(days=1)
        window(last.replace(day=1), last)
        return f"{m.group(0)} ({last.strftime('%B %Y')})"

    sub(r"(?<!the )\blast month\b", last_month)
    sub(r"(?<!the )\blast year\b", lambda m: f"{m.group(0)} ({now.year - 1})")

    # -- ranges: for the reader only ----------------------------------------
    def since(m: "re.Match") -> str:
        n = qn(m.group(1)) if m.group(1) else 1
        unit = m.group(2).lower()
        if n <= 0:
            return m.group(0)
        start = (now - timedelta(days=n) if unit == "day" else
                 now - timedelta(weeks=n) if unit == "week" else
                 _months_back(now, n if unit == "month" else 12 * n))
        return f"{m.group(0)} (since {format_date(start)})"

    sub(rf"\bpast\s+(?:{_Q_NUM}\s+)?(day|week|month|year)s?\b", since)
    sub(rf"\bthe\s+last\s+(?:{_Q_NUM}\s+)?(day|week|month|year)s?\b", since)
    sub(rf"(?<!the )\blast\s+{_Q_NUM}\s+(day|week|month|year)s\b", since)

    def this_period(m: "re.Match") -> str:
        unit = m.group(1).lower()
        start = (now - timedelta(days=now.weekday()) if unit == "week" else
                 now.replace(day=1) if unit == "month" else now.replace(month=1, day=1))
        return f"{m.group(0)} (since {format_date(start)})"

    sub(r"\bthis\s+(week|month|year)\b", this_period)
    return out, windows

@dataclass
class PlanStep:
    id: str
    terms: str        # keywords -> lexical path
    question: str     # natural-language sub-question -> semantic path
                      # NEVER shown to the reader


@dataclass
class Plan:
    steps: List[PlanStep]
    raw: Optional[str] = None
    from_llm: bool = True
    needs_multiple: bool = False

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


class MemorySearch:
    """Multi-path scoring over all stored episodes. No LLM calls."""

    def __init__(self, layer: RecursiveMemoryLayer, config: RLMConfig):
        self._layer = layer
        self._config = config
        self._embedder = getattr(layer, "_embedder", None)
        self._emb_ids: Optional[List[str]] = None
        self._emb_matrix: Optional[np.ndarray] = None

    # -- metadata --------------------------------------------------------

    def _constraints(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        out: Dict[str, Any] = {}
        speakers = {s.lower() for s in self._layer.entity_stats.speakers}
        hit = {s for s in speakers if len(s) > 2 and re.search(rf"\b{re.escape(s)}\b", q)}
        if hit:
            out["speakers"] = hit
        years = {int(y) for y in re.findall(r"\b(20\d{2})\b", q)}
        if years:
            out["years"] = years
        months = {n for name, n in _MONTHS.items() if re.search(rf"\b{name}\b", q)}
        if months:
            out["months"] = months
        return out

    def _metadata_boost(self, m: MemoryObject, c: Dict[str, Any]) -> float:
        if not c:
            return 0.0
        boost = 0.0
        cfg = self._config
        if "speakers" in c:
            eps = {s.lower() for s in m.episode_speakers}
            low = m.raw_text.lower()
            if any(sp in eps or f"speaker {sp}" in low for sp in c["speakers"]):
                boost += cfg.boost_speaker
        head = m.raw_text[:80]
        if "years" in c:
            found = {int(y) for y in re.findall(r"\b(20\d{2})\b", f"{m.timestamp} {head}")}
            if found & c["years"]:
                boost += cfg.boost_year
        if "months" in c:
            nums = {int(x) for x in re.findall(r"-0?(\d{1,2})-", m.timestamp or "")
                    if 1 <= int(x) <= 12}
            low = f"{m.timestamp} {head}".lower()
            nums |= {n for name, n in _MONTHS.items() if re.search(rf"\b{name}\b", low)}
            if nums & c["months"]:
                boost += cfg.boost_month
        return boost

    # -- scoring ---------------------------------------------------------

    def _embedding_matrix(self) -> Tuple[List[str], Optional[np.ndarray]]:
        """Cached (episode_ids, stacked embeddings) for one matvec per query."""
        episodes = self._layer.all_episodes()
        if self._emb_ids is None or len(episodes) != len(self._emb_ids):
            ids, rows = [], []
            for m in episodes:
                if m.embedding is not None:
                    ids.append(m.id)
                    rows.append(m.embedding)
            self._emb_ids = ids
            self._emb_matrix = np.vstack(rows) if rows else None
        return self._emb_ids, self._emb_matrix

    def _lexical_scores(self, terms: str) -> Dict[str, float]:
        if self._config.lexical_scorer == "bm25":
            return self._layer.bm25.score_all(terms)
        query_tokens = set(re.findall(r"\w+", terms.lower()))
        if not query_tokens:
            return {}
        out: Dict[str, float] = {}
        for m in self._layer.all_episodes():
            s = _fuzzy_token_overlap(query_tokens, m.tokens, doc_stems=m.stem_tokens)
            if s > 0.0:
                out[m.id] = s
        return out

    def _semantic_scores(self, semantic_text: str) -> Dict[str, float]:
        if self._embedder is None:
            return {}
        qv = self._embedder.embed(semantic_text)
        if qv is None:
            return {}
        ids, mat = self._embedding_matrix()
        if mat is None:
            return {}
        sims = mat @ qv
        return {i: float(s) for i, s in zip(ids, sims) if s > 0.0}

    def search(self, terms: str, semantic_text: str,
               caption_keywords: str = "") -> List[Dict[str, Any]]:
        episodes = self._layer.all_episodes()
        if not episodes:
            return []

        by_id = {m.id: m for m in episodes}
        k = self._config.scan_top_k

        lex_raw = self._lexical_scores(terms)
        sem_raw = self._semantic_scores(semantic_text)

        top_lex = sorted(lex_raw.items(), key=lambda kv: kv[1], reverse=True)[:k]
        top_sem = sorted(sem_raw.items(), key=lambda kv: kv[1], reverse=True)[:k]
        candidates = list(dict.fromkeys([i for i, _ in top_sem] + [i for i, _ in top_lex]))
        if not candidates:
            return []

        # -- fuse over the candidate union
        if self._config.fusion == "rrf":
            fused = rrf([[i for i, _ in top_sem], [i for i, _ in top_lex]],
                        k=self._config.rrf_k)
            lex_n = {i: lex_raw.get(i, 0.0) for i in candidates}
            sem_n = {i: sem_raw.get(i, 0.0) for i in candidates}
        elif self._config.fusion == "max":
            fused = {i: max(lex_raw.get(i, 0.0), sem_raw.get(i, 0.0)) for i in candidates}
            lex_n = {i: lex_raw.get(i, 0.0) for i in candidates}
            sem_n = {i: sem_raw.get(i, 0.0) for i in candidates}
        else:  # convex
            lex_n = minmax({i: lex_raw.get(i, 0.0) for i in candidates})
            sem_n = minmax({i: sem_raw.get(i, 0.0) for i in candidates})
            a = self._config.alpha
            fused = {i: a * sem_n[i] + (1.0 - a) * lex_n[i] for i in candidates}

        constraints = self._constraints(f"{terms} {semantic_text}")
        multimodal = self._layer.config.enable_multimodal
        cap_emb = None
        if multimodal and caption_keywords.strip() and self._embedder:
            cap_emb = self._embedder.embed(caption_keywords)
        query_tokens = set(re.findall(r"\w+", terms.lower()))

        scored: List[Dict[str, Any]] = []
        for eid in candidates:
            m = by_id.get(eid)
            if m is None:
                continue

            cap = 0.0
            if multimodal and m.image_path:
                caption = f"{m.image_caption_concise} {m.image_caption_detailed}".strip()
                if caption:
                    ck = set(re.findall(r"\w+", caption_keywords.lower())) or query_tokens
                    ck = {t for t in ck if len(t) > 2}
                    lex_cap = (len(ck & set(re.findall(r"\w+", caption.lower()))) / len(ck)
                               if ck else 0.0)
                    sem_cap = 0.0
                    if cap_emb is not None and self._embedder:
                        ce = self._embedder.embed(caption)
                        if ce is not None:
                            sem_cap = max(0.0, float(np.dot(cap_emb, ce)))
                    cap = max(lex_cap, sem_cap) * self._config.caption_weight

            meta = self._metadata_boost(m, constraints)
            scored.append({
                "kind": "episode",
                "episode_id": eid,
                "global_seq": m.global_seq,
                "timestamp": m.timestamp,
                "evidence_ids": list(m.evidence_ids),
                "score": fused[eid] + meta + cap,
                "fused_score": round(fused[eid], 4),
                "lexical_score": round(lex_n.get(eid, 0.0), 4),
                "semantic_score": round(sem_n.get(eid, 0.0), 4),
                "lexical_raw": round(lex_raw.get(eid, 0.0), 4),
                "semantic_raw": round(sem_raw.get(eid, 0.0), 4),
                "metadata_boost": round(meta, 3),
                "caption_boost": round(cap, 3),
                "has_image": bool(m.image_path),
                "preview": m.raw_text[:120],
            })

        merged = sorted(scored, key=lambda e: e["score"], reverse=True)

        # Multimodal: fold in the dedicated caption index.
        if multimodal:
            idx = self._layer.get_image_caption_index()
            if idx is not None and idx.size:
                q = caption_keywords.strip() or terms
                by_eid = {e["episode_id"]: e for e in merged}
                for hit in idx.search(q, top_k=self._config.image_caption_top_k):
                    eid = hit["episode_id"]
                    if eid in by_eid:
                        by_eid[eid]["score"] = round(by_eid[eid]["score"] + hit["score"] * 0.3, 4)
                    else:
                        ep = self._layer.get_episode(eid)
                        if ep is None:
                            continue
                        merged.append({
                            "episode_id": eid,
                            "global_seq": ep.global_seq,
                            "timestamp": ep.timestamp,
                            "evidence_ids": list(ep.evidence_ids),
                            "score": round(hit["score"], 4),
                            "lexical_score": 0.0, "semantic_score": 0.0,
                            "metadata_boost": 0.0, "caption_boost": round(hit["score"], 3),
                            "has_image": True,
                            "preview": hit.get("caption", "")[:120],
                        })
                merged = sorted(merged, key=lambda e: e["score"], reverse=True)

        return merged

    # -- memory map for the planner --------------------------------------

    def memory_map(self, max_rows: int = 12) -> Dict[str, Any]:
        root = self._layer.nodes.get(self._layer.root_id)
        rows: List[Dict[str, Any]] = []
        if isinstance(root, RegionNode):
            for cid in root.children[:max_rows]:
                child = self._layer.nodes.get(cid)
                if not isinstance(child, RegionNode):
                    continue
                row = {
                    "anchors": child.anchor_terms[:6],
                    "entities": child.key_entities[:4],
                    "seq_range": list(child.seq_range),
                }
                if child.time_range:
                    row["dates"] = [child.time_range.get("start"),
                                    child.time_range.get("end")]
                if child.image_count:
                    row["images"] = child.image_count
                rows.append(row)

        episodes = self._layer.all_episodes()
        dates = sorted(m.timestamp for m in episodes if m.timestamp)
        return {
            "regions": rows,
            "entities": self._layer.entity_stats.top_entities(n=20),
            "n_episodes": len(episodes),
            "date_span": [dates[0], dates[-1]] if dates else None,
        }


# ==============================================================================
# Planner (LLM call 1)
# ==============================================================================

class Planner:
    """Turns the question into 1-3 retrieval sub-queries. Never gates memory."""

    FEW_SHOT = (
        'Question: "What is Caroline\'s job?"\n'
        '{"steps": [{"id": "q1", "terms": "Caroline job work career profession", '
        '"question": "What is Caroline\'s job or profession?"}]}\n\n'
        'Question: "What hobby did Caroline take up after the LGBTQ support group?"\n'
        '{"steps": [{"id": "q1", "terms": "Caroline LGBTQ support group attended", '
        '"question": "When did Caroline attend the LGBTQ support group?"}, '
        '{"id": "q2", "terms": "Caroline hobby started began new activity", '
        '"question": "What new hobby did Caroline start?"}]}'
    )

    FEW_SHOT_MULTIQUERY = (
        'Question: "What is Caroline\'s job?"\n'
        '{"steps": ['
        '{"id": "q1", "terms": "Caroline job work profession", '
        '"question": "What is Caroline\'s job?"}, '
        '{"id": "q2", "terms": "Caroline career role employer works as hired", '
        '"question": "What does Caroline do for a living?"}]}\n\n'
        'Question: "What hobby did Caroline take up after the LGBTQ support group?"\n'
        '{"steps": ['
        '{"id": "q1", "terms": "Caroline hobby took up started new", '
        '"question": "What new hobby did Caroline take up?"}, '
        '{"id": "q2", "terms": "Caroline began learning class activity picked up", '
        '"question": "What new activity did Caroline start doing?"}, '
        '{"id": "q3", "terms": "Caroline LGBTQ support group attended went", '
        '"question": "When did Caroline attend the LGBTQ support group?"}]}'
    )

    def __init__(self, config: RLMConfig):
        self._config = config

    def _map_text(self, mm: Dict[str, Any]) -> str:
        lines = []
        for i, r in enumerate(mm.get("regions") or []):
            parts = [f"  region {i}: anchors=[{', '.join(r.get('anchors') or [])}]"]
            if r.get("entities"):
                parts.append(f"entities=[{', '.join(r['entities'])}]")
            if r.get("dates"):
                d = r["dates"]
                parts.append(f"dates={d[0]} to {d[1]}")
            lines.append(" | ".join(parts))
        if not lines:
            lines = ["  (memory is a single region)"]
        return "\n".join(lines)

    CONSERVATIVE_RULES = (
        "RULES:\n"
        "- Use 1 sub-query when a single fact answers the question.\n"
        "- Use 2-3 sub-queries only when genuinely distinct facts are needed,\n"
        "  such as two events to compare or a bridge entity to resolve.\n"
    )

    MULTIQUERY_RULES = (
        "RULES:\n"
        "- ALWAYS output 2 or 3 sub-queries, never 1. Each must approach the\n"
        "  question from a DIFFERENT angle, because any single phrasing can miss\n"
        "  the turn that holds the answer.\n"
        "- Sub-query 1: the question's own wording and most literal keywords.\n"
        "- Sub-query 2: a paraphrase using different vocabulary the speakers\n"
        "  would plausibly have used, including synonyms for the key nouns.\n"
        "- Sub-query 3 (when the question depends on another fact, compares two\n"
        "  things, or asks what followed something): target that OTHER fact\n"
        "  directly, so both halves of the answer can be retrieved.\n"
    )

    def _few_shot(self) -> str:
        style = self._config.planner_style
        if style == "multiquery":
            return self.FEW_SHOT_MULTIQUERY
        if style == "adaptive":
            return self.FEW_SHOT_ADAPTIVE
        return self.FEW_SHOT

    ADAPTIVE_RULES = (
        "RULES:\n"
        "- First decide one thing: does answering this question need information\n"
        "  from MORE THAN ONE moment in the conversation? It does when the\n"
        "  question compares two things, asks what followed something else,\n"
        "  counts occurrences, or depends on a fact that must itself be looked\n"
        "  up first. It does NOT when a single statement would answer it, even\n"
        "  if that statement is hard to find. Report this as\n"
        "  \"needs_multiple_moments\": true or false.\n"
        "- If false: output EXACTLY ONE sub-query, worded as precisely as you\n"
        "  can. Extra sub-queries would only add distracting candidates.\n"
        "- If true: output ONE sub-query PER FACT needed, 2 or 3 in total, each\n"
        "  naming its own fact. Do not paraphrase the same fact twice.\n"
    )

    FEW_SHOT_ADAPTIVE = (
        'Question: "When did Caroline go to the LGBTQ support group?"\n'
        '{"needs_multiple_moments": false, "steps": ['
        '{"id": "q1", "terms": "Caroline LGBTQ support group went attended", '
        '"question": "When did Caroline attend the LGBTQ support group?"}]}\n\n'
        'Question: "What hobby did Caroline take up after the LGBTQ support group?"\n'
        '{"needs_multiple_moments": true, "steps": ['
        '{"id": "q1", "terms": "Caroline LGBTQ support group attended date", '
        '"question": "When did Caroline attend the LGBTQ support group?"}, '
        '{"id": "q2", "terms": "Caroline hobby started took up new activity", '
        '"question": "What new hobby did Caroline take up?"}]}\n\n'
        'Question: "How many times did Melanie go running?"\n'
        '{"needs_multiple_moments": true, "steps": ['
        '{"id": "q1", "terms": "Melanie running run jog went", '
        '"question": "When did Melanie go running?"}, '
        '{"id": "q2", "terms": "Melanie race marathon training miles", '
        '"question": "What running events did Melanie take part in?"}]}'
    )

    def _rules(self) -> str:
        style = self._config.planner_style
        if style == "multiquery":
            return self.MULTIQUERY_RULES
        if style == "adaptive":
            return self.ADAPTIVE_RULES
        return self.CONSERVATIVE_RULES

    def plan(self, query: str, memory_map: Dict[str, Any], llm: Any) -> Plan:
        caller = _resolve_llm(llm)
        fallback = Plan(steps=[PlanStep(id="q1", terms=query, question=query)],
                        from_llm=False)
        if caller is None or not self._config.enable_planner:
            return fallback

        span = memory_map.get("date_span")
        span_text = f"Conversation spans {span[0]} to {span[1]}.\n" if span else ""

        prompt = (
            "You plan searches over a conversational memory.\n"
            "Break the question into 1-3 sub-queries used ONLY to FIND relevant\n"
            "excerpts. A separate model will read those excerpts and answer the\n"
            "user's original question, so do not try to answer it here.\n\n"
            f"MEMORY MAP ({memory_map.get('n_episodes', 0)} stored excerpts):\n"
            f"{self._map_text(memory_map)}\n"
            f"{span_text}"
            f"People and entities mentioned: {', '.join(memory_map.get('entities') or []) or '(none)'}\n\n"
            + self._rules()
            + (
            "- \"terms\": 5-8 keywords likely to appear verbatim in the conversation.\n"
            "  Include names, places and concrete nouns. Do not include stopwords.\n"
            "- \"question\": a natural-language sentence stating what to look for.\n"
            "- Output ONLY a JSON object. No commentary.\n\n"
            f"EXAMPLES:\n{self._few_shot()}\n\n"
            f'Now plan for:\nQuestion: "{query}"\n\n'
            + ('Output: {"needs_multiple_moments": true|false, "steps": '
               '[{"id": "...", "terms": "...", "question": "..."}]}'
               if self._config.planner_style == "adaptive" else
               'Output: {"steps": [{"id": "...", "terms": "...", "question": "..."}]}')
            )
        )

        raw = caller.get_completion(
            prompt,
            temperature=self._config.planner_temperature,
            max_tokens=self._config.planner_max_tokens,
            role="planner",
        )
        parsed = _first_json_object(raw or "")
        if not parsed:
            logger.debug("Planner returned unparseable output; using raw query.")
            return fallback

        steps: List[PlanStep] = []
        for i, s in enumerate(parsed.get("steps") or []):
            if not isinstance(s, dict):
                continue
            terms = str(s.get("terms") or s.get("scan_terms") or "").strip()
            question = str(s.get("question") or s.get("inspect_question") or "").strip()
            if not terms and not question:
                continue
            steps.append(PlanStep(
                id=str(s.get("id") or f"q{i + 1}"),
                terms=terms or question,
                question=question or terms,
            ))
            if len(steps) >= self._config.max_plan_steps:
                break

        if not steps:
            return fallback
        # Honour the planner's own decomposition verdict: if it judged the
        # question answerable from one moment, keep one sub-query even if it
        # listed more, so single-fact questions keep a tight candidate pool.
        if (self._config.planner_style == "adaptive"
                and parsed.get("needs_multiple_moments") is False):
            steps = steps[:1]
        plan = Plan(steps=steps, raw=raw, from_llm=True)
        plan.needs_multiple = bool(parsed.get("needs_multiple_moments", len(steps) > 1))
        return plan

class Reader:
    """Answers the original question from the pooled evidence pack."""

    def __init__(self, config: RLMConfig):
        self._config = config

    def _excerpts(self, episodes: List[MemoryObject],
                  image_episode_ids: Optional[set] = None,
                  now: Optional[datetime] = None
                  ) -> Tuple[str, List[str], List[str]]:
        
        parts, images, labels = [], [], []
        seen_paths: set = set()
        for i, m in enumerate(episodes, 1):
            text = m.raw_text.strip()
            if m.image_caption_detailed:
                text += f"\n[Image description: {m.image_caption_detailed}]"
            session_stamp = m.metadata.get("session_datetime") or m.timestamp
            if self._config.resolve_relative_dates and session_stamp:
                text = _resolve_relative_dates(text, session_stamp, now=now)
            anchor = parse_session_date(session_stamp)
            recorded = format_date(anchor) if anchor else (m.timestamp or "unknown")
            if now is not None and anchor is not None:
                recorded += f", {days_before(anchor, now)}"
            parts.append(f"EXCERPT {i} (recorded {recorded}):\n{text}")
            if image_episode_ids is not None and m.id not in image_episode_ids:
                continue
            md = m.metadata or {}
            paths = list(md.get("image_paths") or ([m.image_path] if m.image_path else []))
            ids = list(md.get("image_ids") or [])
            for j, p in enumerate(paths):
                if not p or p in seen_paths:
                    continue
                seen_paths.add(p)
                images.append(p)
                labels.append(f"{ids[j]} (excerpt {i})" if j < len(ids) else f"excerpt {i}")
        return "\n\n".join(parts), images, labels

    def _prompt(self, question: str, excerpts: str, has_images: bool,
                question_image: bool,
                image_labels: Optional[List[str]] = None) -> str:
        if self._config.reader_prompt == "legacy":
            return (
                "Read these conversation excerpts and answer the question.\n\n"
                f"EXCERPTS:\n{excerpts}\n\n"
                f"QUESTION: {question}\n\n"
                "CRITICAL - Temporal questions:\n"
                "If the question asks WHEN and an excerpt contains relative time\n"
                "words, express the answer RELATIVE TO THE SESSION DATE (the\n"
                "'recorded' timestamp). If the excerpt states an absolute date,\n"
                "return it directly.\n"
                "Instructions:\n"
                "- Answer using ONLY information in the EXCERPTS above.\n"
                "- Give the shortest correct answer phrase. No preamble.\n"
                "- For listing questions, include ALL items.\n"
                "- If the answer is not present, respond: NOT_FOUND\n\n"
                "ANSWER:"
            )

        image_note = ""
        if has_images:
            image_note = ("Some excerpts include images. Use both the text and the "
                          "images.\n")
        # Which attached image is which. Without this a reader shown five
        # images and asked for an image id can only guess the mapping.
        order_note = ""
        if has_images and image_labels:
            order_note = ("The images are attached in this order: "
                          + "; ".join(f"{k}. {lab}" for k, lab in
                                      enumerate(image_labels, 1)) + ".\n")
            image_note += order_note
        if self._config.reader_prompt == "reason":
            return (
                "Answer the question using the conversation excerpts below.\n"
                f"{image_note}\n"
                f"EXCERPTS:\n{excerpts}\n\n"
                f"QUESTION: {question}\n\n"
                "Instructions:\n"
                "- First think briefly, in at most four sentences: which excerpts\n"
                "  matter, and how they combine (dates, order, several facts).\n"
                "- Base the answer on the excerpts, and reason from them when the\n"
                "  answer is not stated outright. If the question asks what someone\n"
                "  would likely do, prefer, choose or be, commit to the best-supported\n"
                "  answer instead of declining.\n"
                "- Then write the final answer on the LAST line, starting with\n"
                "  \"ANSWER:\" — the shortest phrase that fully answers the question,\n"
                "  in the wording and date style of the excerpts. Text in\n"
                "  parentheses is a resolved value you should use as written.\n"
                "- If the question asks for several things, include all of them.\n"
                "- Write \"ANSWER: NOT_FOUND\" only if the excerpts contain nothing\n"
                "  relevant to the question.\n"
            )
        if question_image:
            image_note = ("The first image belongs to the question; the rest come "
                          "from the conversation. Use both text and images.\n"
                          + order_note)

        return (
            "Answer the question using the conversation excerpts below.\n"
            f"{image_note}\n"
            f"EXCERPTS:\n{excerpts}\n\n"
            f"QUESTION: {question}\n\n"
            "Instructions:\n"
            "- Base the answer on the excerpts, and reason from them when the\n"
            "  answer is not stated outright. If the question asks what someone\n"
            "  would likely do, prefer, choose or be, commit to the best-supported\n"
            "  answer instead of declining.\n"
            "- Reply with the shortest phrase that fully answers the question.\n"
            "  No preamble, no explanation, no restatement of the question.\n"
            "- If the question asks for several things, include all of them.\n"
            "- Match the wording and date style of the excerpts. Text in\n"
            "  parentheses is a resolved value you should use as written.\n"
            "- Reply exactly NOT_FOUND only if the excerpts contain nothing\n"
            "  relevant to the question. Needing to infer is not a reason to\n"
            "  answer NOT_FOUND.\n\n"
            "ANSWER:"
        )

    @staticmethod
    def _notes_block(notes: List[Any], conflicts: List[Dict[str, Any]]) -> str:
        if not notes:
            return ""
        lines = [f"- {n.render()}" for n in notes]
        block = ("REMEMBERED FACTS (already distilled from earlier sessions; "
                 "a fact marked superseded is no longer current):\n"
                 + "\n".join(lines) + "\n\n")
        if conflicts:
            c_lines = [f"- \"{c['statement_a']}\" ({c['time_a']}) vs "
                       f"\"{c['statement_b']}\" ({c['time_b']})"
                       for c in conflicts[:5]]
            block += ("KNOWN CONTRADICTIONS in these facts:\n"
                      + "\n".join(c_lines) + "\n\n")
        return block

    def read(self, question: str, episodes: List[MemoryObject], llm: Any,
             question_image_path: Optional[str] = None,
             notes: Optional[List[Any]] = None,
             conflicts: Optional[List[Dict[str, Any]]] = None,
             image_episode_ids: Optional[set] = None,
             now: Optional[datetime] = None) -> Dict[str, Any]:
        if not episodes and not notes:
            return {"answer": "", "prompt_chars": 0, "n_images": 0}

        excerpts, images, labels = self._excerpts(episodes, image_episode_ids, now=now)
        excerpts = self._notes_block(notes or [], conflicts or []) + excerpts
        # The prompt says "the first image belongs to the question" only when
        # that image was actually attached: a path to a missing file used to
        # produce the claim with no image behind it.
        has_q_image = bool(question_image_path and os.path.exists(question_image_path))
        if has_q_image:
            images.insert(0, question_image_path)
            labels.insert(0, "the question's own image")

        prompt = self._prompt(question, excerpts, bool(images), has_q_image, labels)
        if self._config.system_prompt:
            prompt = f"{self._config.system_prompt.strip()}\n\n{prompt}"

        mllm = _resolve_mllm(llm) if images else None
        caller = _resolve_llm(llm)

        try:
            if mllm is not None:
                raw = mllm.get_image_completion(
                    prompt, images,
                    temperature=self._config.reader_temperature,
                    max_tokens=self._config.reader_max_tokens,
                    role="reader",
                )
                # what reached the model, not what was asked for
                images = images[:getattr(mllm, "last_images_sent", len(images))]
            elif caller is not None:
                raw = caller.get_completion(
                    prompt,
                    temperature=self._config.reader_temperature,
                    max_tokens=(self._config.reader_reason_max_tokens
                                if self._config.reader_prompt == "reason"
                                else self._config.reader_max_tokens),
                    role="reader",
                    logprobs=self._config.reader_confidence,
                )
            else:
                return {"answer": "", "prompt_chars": len(prompt),
                        "n_images": len(images)}
        except Exception as e:
            logger.warning("Reader call failed: %s", e)
            return {"answer": "", "prompt_chars": len(prompt),
                    "n_images": len(images)}

        # Free confidence: the backend returned these alongside the text of the
        # call just made, so reading them adds no call and no tokens.
        confidence: Dict[str, float] = {}
        if self._config.reader_confidence:
            try:
                confidence = (caller or mllm).last_confidence() or {}
            except Exception:
                confidence = {}

        reasoning = ""
        if self._config.reader_prompt == "reason":
            reasoning = raw or ""
            answer = _final_answer_line(raw or "")
        else:
            answer = _clean_answer(raw or "")
        # The reader declining is not the same event as the reader failing, but
        # both leave `answer` empty. `abstained` keeps them apart for benchmarks
        # that grade a refusal (LongMemEval's abstention questions); F1
        # benchmarks are unaffected, since the answer is empty either way.
        abstained = answer.upper().startswith("NOT_FOUND")
        if abstained:
            answer = ""
        out = {"answer": answer, "prompt_chars": len(prompt),
               "confidence": confidence, "n_images": len(images),
               "abstained": abstained}
        if reasoning:
            out["reasoning"] = reasoning[-600:]
        return out

class RLMController:
    """Two LLM calls per query: one planner, one reader."""

    def __init__(self, layer: RecursiveMemoryLayer,
                 rlm_config: Optional[RLMConfig] = None):
        self.layer = layer
        self.config = rlm_config or RLMConfig()
        self.search = MemorySearch(layer, self.config)
        self.planner = Planner(self.config)
        self.reader = Reader(self.config)
        self._memory_map_cache: Optional[Dict[str, Any]] = None
        self._memory_map_size: int = -1

    def _seq_index(self) -> Dict[int, Any]:
        """global_seq -> episode, rebuilt only when the layer grows."""
        eps = self.layer.all_episodes()
        if getattr(self, "_seq_cache_n", -1) != len(eps):
            self._seq_cache = {m.global_seq: m for m in eps}
            self._seq_cache_n = len(eps)
        return self._seq_cache

    def _with_neighbours(self, episodes: List[Any], w: int) -> List[Any]:
        """The pack plus the turns within `w` of each pack turn, same session."""
        idx = self._seq_index()
        chosen = {m.id: m for m in episodes}

        def session(m: Any) -> Any:
            md = getattr(m, "metadata", None) or {}
            return md.get("session_id") or md.get("session_datetime") or m.timestamp

        for m in list(episodes):
            for d in range(1, w + 1):
                for seq in (m.global_seq - d, m.global_seq + d):
                    n = idx.get(seq)
                    if n is None or n.id in chosen or session(n) != session(m):
                        continue
                    chosen[n.id] = n
        return sorted(chosen.values(), key=lambda m: m.global_seq)

    def _memory_map(self) -> Dict[str, Any]:
        n = len(self.layer.all_episodes())
        if self._memory_map_cache is None or self._memory_map_size != n:
            self._memory_map_cache = self.search.memory_map()
            self._memory_map_size = n
        return self._memory_map_cache

    def _pool(self, per_step: List[List[Dict[str, Any]]],
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = self.config.top_k_episodes if limit is None else limit
        chosen: List[Dict[str, Any]] = []
        seen: set = set()

        if self.config.pool_strategy == "score" or len(per_step) == 1:
            flat: Dict[str, Dict[str, Any]] = {}
            for lst in per_step:
                for ep in lst:
                    prev = flat.get(ep["episode_id"])
                    if prev is None or ep["score"] > prev["score"]:
                        flat[ep["episode_id"]] = ep
            return sorted(flat.values(), key=lambda e: e["score"], reverse=True)[:limit]

        depth = 0
        while len(chosen) < limit:
            progressed = False
            for lst in per_step:
                if depth >= len(lst):
                    continue
                ep = lst[depth]
                progressed = True
                if ep["episode_id"] in seen:
                    continue
                seen.add(ep["episode_id"])
                chosen.append(ep)
                if len(chosen) >= limit:
                    break
            if not progressed:
                break
            depth += 1
        return chosen

    def _build_pack(self, ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Take the top-k for the reader. With note_slots > 0, that many slots are
        reserved for notes so a distilled fact cannot be crowded out entirely;
        with note_slots = 0 notes and episodes compete purely on rerank score.
        """
        limit = self.config.top_k_episodes
        if self.config.note_mode == "additive":
            episodes = [r for r in ranked if r.get("kind") != "note"][:limit]
            notes = [r for r in ranked if r.get("kind") == "note"]
            return episodes + notes[: self.config.note_additive_max]
        if self.config.note_slots <= 0:
            return ranked[:limit]
        notes = [r for r in ranked if r.get("kind") == "note"][: self.config.note_slots]
        chosen = list(notes)
        for r in ranked:
            if len(chosen) >= limit:
                break
            if r not in chosen:
                chosen.append(r)
        return chosen[:limit]

    def _search_notes(self, query: str) -> List[Dict[str, Any]]:
        """Candidate notes for this query, shaped like episode candidates."""
        store = getattr(self.layer, "notes", None)
        if store is None or not store.size or not self.config.use_notes:
            return []
        embedder = getattr(self.layer, "_embedder", None)
        qv = embedder.embed(query) if embedder is not None else None
        out: List[Dict[str, Any]] = []
        for note_id, score in store.search(query, qv, self.config.note_top_k,
                                           alpha=self.config.alpha):
            note = store.get(note_id)
            if note is None:
                continue
            out.append({
                "kind": "note",
                "note_id": note_id,
                "episode_id": note_id,        # keeps pooling/dedup uniform
                "global_seq": -1,
                "timestamp": note.event_time or note.recorded_time,
                "evidence_ids": list(note.evidence_ids),
                "score": score,
                "status": note.status,
                "preview": note.statement[:120],
            })
        return out

    def _expand_via_summaries(self, note_candidates: List[Dict[str, Any]],
                              already: set) -> List[Dict[str, Any]]:
        store = getattr(self.layer, "notes", None)
        if store is None:
            return []
        out: List[Dict[str, Any]] = []
        seen = set(already)
        for cand in note_candidates[: self.config.expand_summaries]:
            note = store.get(cand.get("note_id", ""))
            if note is None or not note.provenance:
                continue
            for eid in note.provenance:
                if eid in seen:
                    continue
                m = self.layer.get_episode(eid)
                if m is None:
                    continue
                seen.add(eid)
                out.append({
                    "kind": "episode",
                    "episode_id": eid,
                    "global_seq": m.global_seq,
                    "timestamp": m.timestamp,
                    "evidence_ids": list(m.evidence_ids),
                    # Enters the pool below the first-stage hits; the
                    # cross-encoder decides whether it belongs in the pack.
                    "score": float(cand.get("score", 0.0)) * 0.5,
                    "from_summary": cand.get("note_id"),
                })
        return out

    def _expand_via_feedback(self, ranked: List[Dict[str, Any]],
                             already: set) -> List[Dict[str, Any]]:
        top = [r for r in ranked if r.get("kind") != "note"][: self.config.prf_docs]
        if not top:
            return []
        terms: List[str] = []
        for r in top:
            m = self.layer.get_episode(r["episode_id"])
            if m is not None:
                terms.extend(getattr(m, "tokens", None) or m.raw_text.split())
        if not terms:
            return []
        # Rarest terms carry the most signal; the index already has the counts.
        uniq = sorted(set(t for t in terms if len(t) > 3))
        try:
            idf = self.layer.bm25.idf
            uniq.sort(key=lambda t: -idf.get(t, 0.0))
        except Exception:
            pass
        expansion = " ".join(uniq[: self.config.prf_terms])
        if not expansion.strip():
            return []
        hits = self.search.search(terms=expansion, semantic_text=expansion)
        out = []
        for h in hits:
            if h["episode_id"] in already:
                continue
            h = dict(h)
            h["score"] = float(h.get("score", 0.0)) * 0.5
            h["from_prf"] = True
            out.append(h)
        return out

    def _rerank(self, query: str, ranked: List[Dict[str, Any]],
                extra: Optional[List[Dict[str, Any]]] = None
                ) -> List[Dict[str, Any]]:
        reranker = get_default_reranker()
        head = ranked[: self.config.rerank_pool] + list(extra or [])
        tail = ranked[self.config.rerank_pool:]
        passages = []
        for ep in head:
            if ep.get("kind") == "note":
                note = self.layer.notes.get(ep["note_id"])
                passages.append(note.render() if note is not None else ep.get("preview", ""))
            else:
                m = self.layer.get_episode(ep["episode_id"])
                passages.append(m.raw_text if m is not None else ep.get("preview", ""))
        scores = reranker.score(query, passages)
        if scores is None:                 # model unavailable: leave order as is
            return ranked
        for ep, s in zip(head, scores):
            ep["rerank_score"] = round(float(s), 4)
            ep["pre_rerank_score"] = ep["score"]

        if self.config.rerank_normalise:
            self._normalise_by_kind(head)
        head.sort(key=lambda e: e.get("rerank_sort", e.get("rerank_score",
                                                           float("-inf"))),
                  reverse=True)
        return head + tail

    @staticmethod
    def _normalise_by_kind(head: List[Dict[str, Any]]) -> None:
        import statistics as _st
        by_kind: Dict[str, List[Dict[str, Any]]] = {}
        for ep in head:
            by_kind.setdefault(ep.get("kind") or "episode", []).append(ep)
        for rows in by_kind.values():
            vals = [r.get("rerank_score", 0.0) for r in rows]
            if len(vals) < 3:
                for r in rows:
                    r["rerank_sort"] = r.get("rerank_score", 0.0)
                continue
            mu = _st.mean(vals)
            sd = _st.pstdev(vals) or 1.0
            for r in rows:
                r["rerank_sort"] = (r.get("rerank_score", 0.0) - mu) / sd

    def retrieve(self, query: str,
                 question_image_path: Optional[str] = None,
                 reader_suffix: Optional[str] = None,
                 now: Optional[datetime] = None,
                 **_ignored) -> Dict[str, Any]:
        llm = self.layer.llm
        trajectory: List[Dict[str, Any]] = []
        grounding = self.config.temporal_grounding and now is not None
        reader_query, windows = (resolve_question_time(query, now) if grounding
                                 else (query, []))

        # -- LLM call 1: plan -------------------------------------------
        plan = self.planner.plan(query, self._memory_map(), llm)
        trajectory.append({
            "action": "plan",
            "from_llm": plan.from_llm,
            "steps": [{"id": s.id, "terms": s.terms, "question": s.question}
                      for s in plan.steps],
            "needs_multiple": plan.needs_multiple,
        })

        # -- LLM-free search --------------------------------------------
        caption_kw = ""
        if self.layer.config.enable_multimodal:
            m = re.search(r"\[(?:Image description|Question image|Image caption):\s*([^\]]+)\]",
                          query, re.IGNORECASE)
            if m:
                caption_kw = m.group(1).strip()

        per_step: List[List[Dict[str, Any]]] = []
        for step in plan.steps:
            semantic_text = (step.question
                             if self.config.semantic_query == "subquestion" and step.question
                             else query)
            results = self.search.search(
                terms=step.terms or query,
                semantic_text=semantic_text,
                caption_keywords=caption_kw,
            )
            per_step.append(results)
            trajectory.append({
                "action": "search",
                "step_id": step.id,
                "n_results": len(results),
                "top_score": results[0]["score"] if results else 0.0,
            })

        note_candidates = self._search_notes(query)
        if note_candidates:
            per_step = per_step + [note_candidates]

        # "route": summaries are an index only — they pull their member turns in
        # (below) and never occupy a pool or pack slot themselves.
        if self.config.note_mode == "route":
            first_stage = [[r for r in step if r.get("kind") != "note"]
                           for step in per_step]
            first_stage = [step for step in first_stage if step]
        else:
            first_stage = per_step

        ranked = self._pool(first_stage, limit=max(self.config.top_k_episodes,
                                                   self.config.measure_depth))
        expansion: List[Dict[str, Any]] = []
        seen_ids = {r["episode_id"] for r in ranked}
        if self.config.expand_summaries and note_candidates:
            extra = self._expand_via_summaries(note_candidates, seen_ids)
            if extra:
                seen_ids.update(r["episode_id"] for r in extra)
                expansion.extend(extra)
                trajectory.append({"action": "expand_summaries", "added": len(extra)})
        if self.config.prf_docs:
            extra = self._expand_via_feedback(ranked, seen_ids)
            if extra:
                seen_ids.update(r["episode_id"] for r in extra)
                expansion.extend(extra)
                trajectory.append({"action": "expand_prf", "added": len(extra)})
        if self.config.rerank and ranked:
            ranked = self._rerank(query, ranked, extra=expansion)
        elif expansion:
            ranked = ranked + expansion

        pack = self._build_pack(ranked)

        episodes: List[MemoryObject] = []
        notes: List[Any] = []
        for ep in pack:
            if ep.get("kind") == "note":
                note = self.layer.notes.get(ep["note_id"])
                if note is not None:
                    notes.append(note)
            else:
                m = self.layer.get_episode(ep["episode_id"])
                if m is not None:
                    episodes.append(m)
        # Chronological order reads better than score order for a narrative.
        episodes.sort(key=lambda m: m.global_seq)
        # Episodes retrieval ranked, before context neighbours join. Images are
        # attached for these only (see ReaderStage._excerpts).
        ranked_ids = {m.id for m in episodes}
        if self.config.context_window > 0 and episodes:
            n_before = len(episodes)
            episodes = self._with_neighbours(episodes, self.config.context_window)
            trajectory.append({"action": "context_window",
                               "added": len(episodes) - n_before})

        def _evidence(rows: List[Dict[str, Any]]) -> List[str]:
            out: List[str] = []
            for ep in rows:
                for ev in ep.get("evidence_ids") or []:
                    if ev not in out:
                        out.append(ev)
            return out

        retrieved_evidence = _evidence([p for p in pack if p.get("kind") != "note"])
        candidate_evidence = _evidence([r for r in ranked if r.get("kind") != "note"])
        note_evidence = _evidence([p for p in pack if p.get("kind") == "note"])

        # -- LLM call 2: read -------------------------------------------
        reader_question = reader_query
        if self.config.reader_question == "plan" and plan.steps:
            reader_question = plan.steps[-1].question or query
        if reader_suffix:
            reader_question = f"{reader_question}\n\n{reader_suffix.strip()}"

        conflicts = []
        if notes:
            store = getattr(self.layer, "notes", None)
            if store is not None:
                conflicts = store.conflicts_for(n.id for n in notes)
                if hasattr(store, "flag_for_review"):
                    store.flag_for_review(n.id for n in notes)

        result = self.reader.read(reader_question, episodes, llm,
                                  question_image_path=question_image_path,
                                  notes=notes, conflicts=conflicts,
                                  image_episode_ids=ranked_ids,
                                  now=now if grounding else None)
        trajectory.append({
            "action": "read",
            "n_excerpts": len(episodes),
            "n_notes": len(notes),
            "n_images": result.get("n_images", 0),
            "prompt_chars": result["prompt_chars"],
        })

        confidence = dict(result.get("confidence") or {})
        pack_scores = [p.get("rerank_score") for p in pack
                       if p.get("rerank_score") is not None]
        if pack_scores:
            ordered = sorted(pack_scores, reverse=True)
            confidence["rr_top"] = round(float(ordered[0]), 4)
            confidence["rr_margin"] = round(
                float(ordered[0] - sum(ordered) / len(ordered)), 4)

        return {
            "query": query,
            "answer": result["answer"],
            "abstained": bool(result.get("abstained", False)),
            "confidence": confidence,
            "retrieved_episode_ids": [ep["episode_id"] for ep in pack],
            "retrieved_evidence_ids": retrieved_evidence,
            "candidate_evidence_ids": candidate_evidence,
            "note_evidence_ids": note_evidence,
            "retrieved_scores": [round(ep["score"], 4) for ep in pack],
            "retrieval_detail": pack,
            "plan_steps": len(plan.steps),
            "plan_from_llm": plan.from_llm,
            "n_notes_used": len(notes),
            "n_episodes_used": len(episodes),
            "n_images": result.get("n_images", 0),
            "reasoning": result.get("reasoning", ""),
            "question_resolved": reader_query if reader_query != query else "",
            "time_windows": [[str(lo), str(hi)] for lo, hi in windows],
            "trajectory": trajectory,
        }

class RMMWithRLM:
    """Storage + retrieval behind one object."""

    def __init__(self, mem_config: MemoryConfig, rlm_config: RLMConfig,
                 llm_controller: Any):
        self._layer = RecursiveMemoryLayer(mem_config, llm_controller)
        self._controller = RLMController(self._layer, rlm_config)

    @property
    def layer(self) -> RecursiveMemoryLayer:
        return self._layer

    @property
    def controller(self) -> RLMController:
        return self._controller

    def add_memory(self, *a, **kw):
        return self._layer.add_memory(*a, **kw)

    def add_episode_from_utterances(self, *a, **kw):
        return self._layer.add_episode_from_utterances(*a, **kw)

    def retrieve(self, query: str, **kw) -> Dict[str, Any]:
        return self._controller.retrieve(query, **kw)

    def get_stats(self) -> Dict[str, Any]:
        return self._layer.get_stats()

    def print_tree(self, *a, **kw) -> None:
        return self._layer.print_tree(*a, **kw)

    def save(self, path: str) -> None:
        self._layer.save(path)

    @classmethod
    def load(cls, path: str, mem_config: MemoryConfig, rlm_config: RLMConfig,
             llm_controller: Any) -> "RMMWithRLM":
        obj = cls.__new__(cls)
        obj._layer = RecursiveMemoryLayer.load(path, llm_controller)
        obj._layer.llm = llm_controller
        obj._controller = RLMController(obj._layer, rlm_config)
        return obj
