"""
consolidation.py — Phase 3: the evolving semantic layer.

Episodes are immutable and cheap to write. Notes are the layer that changes:
short, self-contained, date-anchored statements distilled from a session, each
carrying provenance back to the episodes it came from, and each able to be
superseded by a later note about the same thing.

Why this exists, from the measurements
--------------------------------------
Multi-hop is the weakest category (40 F1) and its retrieval is the binding
constraint: even with the Phase 2 retriever, only 72% of multi-hop questions get
any gold turn into the reader's pack, because the two halves of the answer sit in
different sessions and no single query retrieves both. A note that already joins
them is retrievable by one query. Knowledge-update and conflict questions have
the same shape: the answer depends on which of several statements is current,
which raw append-only episodes cannot express.

Cost discipline
---------------
Consolidation runs ONCE PER SESSION over only that session's episodes, off the
write path, with a hard cap on calls. It never runs per message. For a LoCoMo
conversation that is ~20-30 calls in total against Mem0's ~2 calls per message
pair (~600+), and the per-utterance insert stays LLM-free and near-instant.

Design choices and their sources
--------------------------------
* Episodes stay immutable; only notes evolve (HiMem's episode/note split).
* Supersession, never deletion: a superseded note keeps its text and gains a
  valid_until, so "what did they say before?" stays answerable (Zep/Graphiti's
  bi-temporal model; "Does Memory Need Graphs?" found UPDATE and NOOP help while
  DELETE is unnecessary).
* Candidate pairs for reconciliation are preselected without an LLM (same
  subject+attribute key, or embedding similarity), so the LLM only adjudicates a
  handful of pairs rather than scanning the store.
* Malformed model output degrades to NOOP rather than corrupting the store,
  because open-weight backbones produce format errors at a measurable rate.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from indexes import BM25Index, minmax

logger = logging.getLogger("consolidation")


# ==============================================================================
# Note
# ==============================================================================

@dataclass
class Note:
    """One distilled, self-contained fact with provenance and validity."""

    id: str
    subject: str                 # who/what the note is about
    attribute: str               # short key: job, location, pet, plan, ...
    statement: str               # self-contained sentence, absolute dates
    event_time: Optional[str] = None      # when the fact was true / happened
    recorded_time: Optional[str] = None   # session date it was said
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    status: str = "active"                # "active" | "superseded"
    superseded_by: Optional[str] = None
    provenance: List[str] = field(default_factory=list)   # episode ids
    evidence_ids: List[str] = field(default_factory=list)  # dataset turn ids
    session: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return f"{self.subject.strip().lower()}::{self.attribute.strip().lower()}"

    def render(self) -> str:
        """How the note appears to the reader."""
        bits = [self.statement.strip()]
        when = self.event_time or self.recorded_time
        if when and when not in self.statement:
            bits.append(f"(as of {when})")
        if self.status == "superseded" and self.valid_until:
            bits.append(f"[superseded on {self.valid_until}]")
        return " ".join(bits)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in vars(self).items() if k != "embedding"}
        return d


# ==============================================================================
# Note store
# ==============================================================================

class NoteStore:
    """Notes plus their indexes and the conflict register."""

    def __init__(self) -> None:
        self._notes: Dict[str, Note] = {}
        self._by_key: Dict[str, List[str]] = {}
        self.bm25 = BM25Index()
        self.conflicts: List[Dict[str, Any]] = []
        self._emb_ids: List[str] = []
        self._emb_matrix: Optional[np.ndarray] = None
        #: Pairs noticed at query time, awaiting adjudication at the next
        #: evolve. Held in `_review_queue` behind a lazy property so stores
        #: restored from an older pickle still work. See `review_queue`.
        self._review_queue: List[Tuple[str, str]] = []

    # -- mutation --------------------------------------------------------

    def add(self, note: Note) -> None:
        self._notes[note.id] = note
        self._by_key.setdefault(note.key, []).append(note.id)
        self.bm25.add(note.id, f"{note.subject} {note.attribute} {note.statement}")
        self._emb_matrix = None

    def supersede(self, old_id: str, new_id: str, when: Optional[str]) -> None:
        old = self._notes.get(old_id)
        if old is None or old.status == "superseded":
            return
        old.status = "superseded"
        old.superseded_by = new_id
        old.valid_until = when

    @property
    def review_queue(self) -> List[Tuple[str, str]]:
        q = self.__dict__.get("_review_queue")
        if q is None:
            q = self.__dict__["_review_queue"] = []
        return q

    def flag_for_review(self, note_ids: Iterable[str]) -> int:
        ids = [i for i in note_ids if i in self._notes]
        added = 0
        for a in range(len(ids)):
            na = self._notes[ids[a]]
            if na.status != "active":
                continue
            for b in range(a + 1, len(ids)):
                nb = self._notes[ids[b]]
                if nb.status != "active" or na.key != nb.key:
                    continue
                if na.statement.strip().lower() == nb.statement.strip().lower():
                    continue
                pair = (na.id, nb.id) if na.id < nb.id else (nb.id, na.id)
                if pair not in self.review_queue:
                    self.review_queue.append(pair)
                    added += 1
        return added

    def take_review_queue(self) -> List[Tuple[str, str]]:
        queued = list(self.review_queue)
        self.__dict__["_review_queue"] = []
        return queued

    def record_conflict(self, a: Note, b: Note, reason: str) -> None:
        self.conflicts.append({
            "note_a": a.id, "note_b": b.id,
            "statement_a": a.statement, "statement_b": b.statement,
            "time_a": a.event_time or a.recorded_time,
            "time_b": b.event_time or b.recorded_time,
            "reason": reason,
        })

    # -- access ----------------------------------------------------------

    def get(self, note_id: str) -> Optional[Note]:
        return self._notes.get(note_id)

    def all(self, include_superseded: bool = True) -> List[Note]:
        return [n for n in self._notes.values()
                if include_superseded or n.status == "active"]

    def by_key(self, key: str) -> List[Note]:
        return [self._notes[i] for i in self._by_key.get(key, []) if i in self._notes]

    def conflicts_for(self, note_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = set(note_ids)
        return [c for c in self.conflicts
                if c["note_a"] in ids or c["note_b"] in ids]

    @property
    def size(self) -> int:
        return len(self._notes)

    def stats(self) -> Dict[str, Any]:
        active = sum(1 for n in self._notes.values() if n.status == "active")
        return {
            "notes": len(self._notes),
            "active": active,
            "superseded": len(self._notes) - active,
            "conflicts": len(self.conflicts),
            "subjects": len({n.subject.lower() for n in self._notes.values()}),
        }

    # -- retrieval -------------------------------------------------------

    def _matrix(self) -> Tuple[List[str], Optional[np.ndarray]]:
        if self._emb_matrix is None:
            ids, rows = [], []
            for n in self._notes.values():
                if n.embedding is not None:
                    ids.append(n.id)
                    rows.append(n.embedding)
            self._emb_ids = ids
            self._emb_matrix = np.vstack(rows) if rows else None
        return self._emb_ids, self._emb_matrix

    def search(self, query: str, query_embedding: Optional[np.ndarray],
               top_k: int, alpha: float = 0.7) -> List[Tuple[str, float]]:
        """Hybrid search over notes, same fusion as the episode path."""
        if not self._notes:
            return []
        lex = self.bm25.score_all(query)
        sem: Dict[str, float] = {}
        if query_embedding is not None:
            ids, mat = self._matrix()
            if mat is not None:
                sims = mat @ query_embedding
                sem = {i: float(s) for i, s in zip(ids, sims) if s > 0}
        cand = list(dict.fromkeys(
            [i for i, _ in sorted(sem.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
            + [i for i, _ in sorted(lex.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]
        ))
        if not cand:
            return []
        ln = minmax({i: lex.get(i, 0.0) for i in cand})
        sn = minmax({i: sem.get(i, 0.0) for i in cand})
        scored = [(i, alpha * sn[i] + (1 - alpha) * ln[i]) for i in cand]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]


# ==============================================================================
# Consolidator
# ==============================================================================

EXTRACT_PROMPT = """You are maintaining a long-term memory about a conversation.

Read this session and write down the durable facts worth remembering. A fact is
worth a note if someone might later ask about it: what someone does, owns,
plans, prefers, decided, or experienced, and events with a date.

SESSION DATE: {date}

SESSION:
{transcript}
{existing_block}
Rules:
- Each note must stand alone. Resolve pronouns to names. Use absolute dates
  ("7 May 2023"), never "yesterday" or "last week".
- "subject" is the person or thing the note is about, usually a name.
- "attribute" is a short lowercase key such as job, pet, location, hobby,
  plan, health, relationship, opinion, event.
- Write at most 8 notes. Prefer few, informative notes over many trivial ones.
- Skip greetings, small talk, and anything already listed under KNOWN FACTS
  unless this session CHANGES it.
- If this session changes a known fact, still write the new note; say what is
  true now.

Output ONLY JSON:
{{"notes": [{{"subject": "...", "attribute": "...", "statement": "...",
"event_time": "<date or null>"}}]}}"""


RECONCILE_PROMPT = """For each numbered pair, decide how the LATER statement
relates to the EARLIER one about the same person.

{pairs}

Labels:
- "same": the later statement says the same thing as the earlier one.
- "update": the later statement REPLACES the earlier one because the same
  ongoing situation changed (a job, a home, a relationship status, a plan, a
  current opinion). Only use this when the earlier statement has stopped being
  true.
- "conflict": they cannot both be true and time passing does not explain it.
- "independent": both remain true. Two different events that both happened are
  ALWAYS independent, even when they involve the same person.

Most pairs are independent. Do not label two separate events as "update".

Output ONLY JSON, one entry per pair:
{{"relations": [{{"pair": 1, "relation": "...", "reason": "<few words>"}}]}}"""

# Attributes describing an ongoing state, where a later statement can genuinely
# replace an earlier one. Anything else (an event that happened, an opinion
# expressed at a moment) accumulates instead of superseding. Without this,
# blind recency supersession destroyed the store: in a first run 102 of 148
# notes were superseded, including "had a car accident" being replaced by
# "completed a family hike".
STATEFUL_ATTRIBUTES = {
    "job", "occupation", "work", "career", "employer", "role",
    "location", "home", "residence", "city", "address",
    "relationship", "status", "marital_status", "partner",
    "health", "condition", "plan", "goal", "current_project",
    "preference", "favourite", "favorite", "contact", "phone", "email",
    "pet", "vehicle", "car", "school", "study", "course",
}


class Consolidator:
    """
    Turns a session's episodes into notes, then reconciles them with what is
    already stored. Bounded to `max_calls_per_session` LLM calls.
    """

    def __init__(self, store: NoteStore, llm: Any, embedder: Any = None,
                 max_calls_per_session: int = 2,
                 max_notes_per_session: int = 8,
                 reconcile_threshold: float = 0.80,
                 same_key_threshold: float = 0.60,
                 max_reconcile_pairs: int = 4,
                 max_pairs_per_call: int = 20,
                 use_llm_reconcile: bool = True):
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.max_calls_per_session = max_calls_per_session
        self.max_notes_per_session = max_notes_per_session
        self.reconcile_threshold = reconcile_threshold
        self.same_key_threshold = same_key_threshold
        self.max_reconcile_pairs = max_reconcile_pairs
        self.max_pairs_per_call = max_pairs_per_call
        self.use_llm_reconcile = use_llm_reconcile
        self.ops: Dict[str, int] = {"add": 0, "update": 0, "noop": 0, "conflict": 0}

    # -- helpers ---------------------------------------------------------

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
    def _json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        t = text.strip()
        if "```" in t:
            t = re.sub(r"```(?:json)?\s*", "", t).replace("```", "").strip()
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

    def _known_facts_block(self, subjects: Iterable[str]) -> str:
        """Active notes about subjects appearing in this session."""
        wanted = {s.lower() for s in subjects}
        rows = [n for n in self.store.all(include_superseded=False)
                if n.subject.lower() in wanted]
        if not rows:
            return ""
        rows = rows[-25:]        # keep the prompt bounded
        lines = "\n".join(f"- [{n.subject}/{n.attribute}] {n.statement}" for n in rows)
        return f"\nKNOWN FACTS (already remembered):\n{lines}\n"

    # -- main entry point ------------------------------------------------

    def consolidate_session(self, episodes: List[Any], session_date: Optional[str],
                            session_id: Optional[str] = None,
                            subjects: Optional[Iterable[str]] = None) -> List[Note]:
        """One session in, new notes out. Costs at most max_calls_per_session."""
        caller = self._caller()
        if caller is None or not episodes:
            return []

        transcript = "\n".join(m.raw_text for m in episodes)
        prompt = EXTRACT_PROMPT.format(
            date=session_date or "unknown",
            transcript=transcript,
            existing_block=self._known_facts_block(subjects or []),
        )
        raw = caller.get_completion(prompt, temperature=0.0, max_tokens=768,
                                    role="consolidate")
        parsed = self._json(raw or "")
        if not parsed or not isinstance(parsed.get("notes"), list):
            logger.debug("consolidation produced no usable JSON for session %s", session_id)
            return []

        ep_ids = [m.id for m in episodes]
        ev_ids: List[str] = []
        for m in episodes:
            ev_ids.extend(m.evidence_ids or [])

        fresh: List[Note] = []
        for item in parsed["notes"][: self.max_notes_per_session]:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement") or "").strip()
            subject = str(item.get("subject") or "").strip()
            if not statement or not subject:
                continue
            note = Note(
                id=str(uuid.uuid4()),
                subject=subject,
                attribute=str(item.get("attribute") or "fact").strip().lower(),
                statement=statement,
                event_time=(str(item["event_time"]).strip()
                            if item.get("event_time") not in (None, "", "null") else None),
                recorded_time=session_date,
                valid_from=str(item.get("event_time") or session_date or "") or None,
                provenance=list(ep_ids),
                evidence_ids=list(dict.fromkeys(ev_ids)),
                session=session_id,
            )
            if self.embedder is not None:
                note.embedding = self.embedder.embed(
                    f"{note.subject} {note.attribute}: {note.statement}")
            fresh.append(note)

        # Notes must be in the store before adjudication so that candidates
        # include earlier notes from this same session.
        self.reconcile_batch(fresh, caller, self.max_calls_per_session - 1)
        for note in fresh:
            self.store.add(note)
        return fresh

    # -- reconciliation --------------------------------------------------

    def _similarity(self, a: Note, b: Note) -> float:
        if a.embedding is None or b.embedding is None:
            return 0.0
        return float(np.dot(a.embedding, b.embedding))

    def _candidates(self, note: Note) -> List[Note]:
        cands: Dict[str, Note] = {}
        for other in self.store.by_key(note.key):
            if other.status != "active" or other.id == note.id:
                continue
            if self._similarity(note, other) >= self.same_key_threshold:
                cands[other.id] = other
        for other in self.store.all(include_superseded=False):
            if other.id in cands or other.id == note.id:
                continue
            if other.subject.lower() != note.subject.lower():
                continue
            if self._similarity(note, other) >= self.reconcile_threshold:
                cands[other.id] = other
        for other in self._group_peers(note):
            if other.id in cands or other.id == note.id:
                continue
            if self._similarity(note, other) >= self.reconcile_threshold:
                cands[other.id] = other

        ordered = sorted(cands.values(),
                         key=lambda n: n.valid_from or n.recorded_time or "")
        return ordered[-self.max_reconcile_pairs:]

    # -- group scoping ---------------------------------------------------

    def attach_groups(self, entity_names: Iterable[str]) -> Dict[str, int]:
        self._entity_patterns = [
            (name, re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE))
            for name in entity_names if len(name) >= 3
        ]
        self._note_groups = {}
        sizes: Dict[str, int] = {}
        for note in self.store.all(include_superseded=False):
            for name in self._entities_in(note):
                self._note_groups.setdefault(name, []).append(note.id)
                sizes[name] = sizes.get(name, 0) + 1
        return sizes

    def _entities_in(self, note: Note) -> List[str]:
        if not getattr(self, "_entity_patterns", None):
            return []
        text = f"{note.subject} {note.statement}"
        return [name for name, pat in self._entity_patterns if pat.search(text)]

    def _group_peers(self, note: Note) -> List[Note]:
        """Active notes sharing an entity group with this one, any session."""
        if not getattr(self, "_note_groups", None):
            return []
        out: Dict[str, Note] = {}
        for name in self._entities_in(note):
            for nid in self._note_groups.get(name, []):
                other = self.store.get(nid)
                if other is not None and other.status == "active":
                    out[nid] = other
        return list(out.values())

    def _apply(self, old: Note, new: Note, relation: str, reason: str) -> None:
        if relation == "same":
            self.ops["noop"] += 1
        elif relation == "update":
            self.store.supersede(old.id, new.id,
                                 new.valid_from or new.recorded_time)
            self.ops["update"] += 1
        elif relation == "conflict":
            self.store.record_conflict(old, new, reason)
            self.ops["conflict"] += 1
        else:
            self.ops["add"] += 1

    def _fallback_relation(self, old: Note, new: Note) -> str:
        """
        Used when no adjudication call is available.
        """
        if new.attribute in STATEFUL_ATTRIBUTES and old.key == new.key:
            if self._similarity(old, new) >= self.same_key_threshold:
                return "update"
        return "independent"

    def reconcile_queue(self, caller: Any = None, calls_left: int = 1) -> Dict[str, int]:
        queued = self.store.take_review_queue()
        if not queued:
            return {"pairs": 0}

        pairs: List[Tuple[Note, Note]] = []
        for a_id, b_id in queued:
            a, b = self.store.get(a_id), self.store.get(b_id)
            if a is None or b is None or a.status != "active" or b.status != "active":
                continue
            # Present them in time order so "earlier/later" in the prompt is true.
            if (a.valid_from or a.recorded_time or "") > (b.valid_from or b.recorded_time or ""):
                a, b = b, a
            pairs.append((a, b))

        if not pairs:
            return {"pairs": 0}

        before = dict(self.ops)
        self._adjudicate(pairs, caller if caller is not None else self._caller(),
                         calls_left)
        return {"pairs": len(pairs),
                **{k: self.ops[k] - before.get(k, 0) for k in self.ops}}

    def reconcile_batch(self, fresh: List[Note], caller: Any,
                        calls_left: int) -> None:
        pairs: List[Tuple[Note, Note]] = []
        for note in fresh:
            for old in self._candidates(note):
                pairs.append((old, note))

        if not pairs:
            self.ops["add"] += len(fresh)
            return
        self._adjudicate(pairs, caller, calls_left)

    def _adjudicate(self, pairs: List[Tuple[Note, Note]], caller: Any,
                    calls_left: int) -> None:
        if calls_left <= 0 or not self.use_llm_reconcile or caller is None:
            for old, new in pairs:
                self._apply(old, new, self._fallback_relation(old, new), "no-llm fallback")
            return

        pairs = pairs[: self.max_pairs_per_call]
        rendered = "\n".join(
            f"{i}. EARLIER ({old.event_time or old.recorded_time or '?'}): "
            f"{old.statement}\n"
            f"   LATER   ({new.event_time or new.recorded_time or '?'}): "
            f"{new.statement}"
            for i, (old, new) in enumerate(pairs, start=1)
        )
        raw = caller.get_completion(
            RECONCILE_PROMPT.format(pairs=rendered),
            temperature=0.0, max_tokens=512, role="consolidate",
        )
        parsed = self._json(raw or "") or {}
        relations = parsed.get("relations")
        if not isinstance(relations, list):
            for old, new in pairs:
                self._apply(old, new, self._fallback_relation(old, new), "unparsed")
            return

        by_index: Dict[int, Dict[str, Any]] = {}
        for item in relations:
            if isinstance(item, dict):
                try:
                    by_index[int(item.get("pair", -1))] = item
                except (TypeError, ValueError):
                    continue
        for i, (old, new) in enumerate(pairs, start=1):
            item = by_index.get(i)
            if item is None:
                self._apply(old, new, self._fallback_relation(old, new), "missing")
                continue
            rel = str(item.get("relation", "independent")).strip().lower()
            if rel not in ("same", "update", "conflict", "independent"):
                rel = self._fallback_relation(old, new)
            self._apply(old, new, rel, str(item.get("reason", "")))
