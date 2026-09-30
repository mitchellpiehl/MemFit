"""
memory_layer.py — MemFit storage layer
"""

from __future__ import annotations

import pickle
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np

from consolidation import NoteStore
from indexes import BM25Index

EMBED_FAILURES = 0
_ENCODER_CACHE: Dict[Any, Any] = {}


class EpisodeEmbedder:
    """Lazy singleton wrapper around a sentence-transformers model."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            import os
            device = os.environ.get("MEMFIT_ENCODER_DEVICE") or None
            key = (self.model_name, device)
            model = _ENCODER_CACHE.get(key)
            if model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self.model_name, device=device)
                _ENCODER_CACHE[key] = model
            self._model = model

    def embed(self, text: str) -> Optional[np.ndarray]:
        global EMBED_FAILURES
        try:
            self._load()
            vec = self._model.encode(
                text, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
        except Exception:
            EMBED_FAILURES += 1
            return None
        if not np.all(np.isfinite(vec)):
            EMBED_FAILURES += 1
            return None
        return vec

    def batch_embed(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        global EMBED_FAILURES
        if not texts:
            return []
        try:
            self._load()
            vecs = self._model.encode(
                texts, normalize_embeddings=True, batch_size=64,
                show_progress_bar=False,
            )
        except Exception:
            EMBED_FAILURES += len(texts)
            return [None] * len(texts)
        out: List[Optional[np.ndarray]] = []
        for v in vecs:
            v = v.astype(np.float32)
            if np.all(np.isfinite(v)):
                out.append(v)
            else:
                EMBED_FAILURES += 1
                out.append(None)
        return out

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


_default_embedder: Optional[EpisodeEmbedder] = None


def get_default_embedder() -> EpisodeEmbedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = EpisodeEmbedder()
    return _default_embedder

class ImageProcessor:
    """Two-tier captioning via an MLLM; falls back to the filename."""

    def __init__(self, mllm: Any = None):
        self._mllm = mllm

    def _resolve(self) -> Any:
        if self._mllm is None:
            return None
        if hasattr(self._mllm, "get_image_completion"):
            return self._mllm
        if hasattr(self._mllm, "llm") and hasattr(self._mllm.llm, "get_image_completion"):
            return self._mllm.llm
        return None

    def caption_concise(self, image_path: str) -> str:
        caller = self._resolve()
        if caller is not None:
            try:
                r = caller.get_image_completion(
                    "Describe this image in one brief sentence (max 20 words). "
                    "Focus on the main subject and key details.",
                    image_path, max_tokens=60, role="caption",
                )
                if r and r.strip():
                    return r.strip()
            except Exception:
                pass
        import os

        return f"Image: {os.path.basename(image_path)}"

    def caption_detailed(self, image_path: str) -> str:
        caller = self._resolve()
        if caller is not None:
            try:
                r = caller.get_image_completion(
                    "Describe this image in detail. Include objects, people, colors, "
                    "visible text, setting, activities and notable features.",
                    image_path, max_tokens=300, role="caption",
                )
                if r and r.strip():
                    return r.strip()
            except Exception:
                pass
        return ""

@dataclass
class MemoryConfig:
    """Storage-side configuration. Retrieval knobs live in RLMConfig."""

    # Tree shape
    max_leaf_objects: int = 6
    max_region_children: int = 4
    max_anchor_terms: int = 8

    # Routing weights for Eq. 1 (anchor-term and prefix overlap).
    routing_weight_anchor: float = 0.4
    routing_weight_lexical: float = 0.25

    # Episodic segmentation
    episodic_max_tokens: int = 800
    episodic_min_tokens: int = 500
    temporal_gap_seconds: float = 300.0
    topic_shift_min_overlap: float = 0.15

    # Embeddings
    enable_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"

    # Multimodal
    enable_multimodal: bool = False
    eager_detailed_caption: bool = True

    random_seed: int = 42

_STOP_CAPS = {
    "The", "This", "That", "There", "These", "Those", "It", "He", "She", "They",
    "We", "You", "I", "A", "An", "And", "But", "Or", "If", "When", "What",
    "Where", "How", "Why", "Who", "Speaker", "Date", "Image", "Yes", "No",
    "Hey", "Hi", "Hello", "Oh", "Wow", "Yeah", "Thanks", "Sorry", "Well",
    "So", "My", "Your", "His", "Her", "Their", "Our", "Its", "Not", "Just",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}

_SPEAKER_RE = re.compile(r"Speaker\s+([A-Z][A-Za-z\-']+)\s+says")
_CAP_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")


@dataclass
class EntityStat:
    count: int = 0        # times seen capitalised
    lower_count: int = 0  # times the same word appears lowercase
    first_seq: int = -1
    last_seq: int = -1


class EntityStats:

    #: a name must be capitalised at least this often ...
    MIN_COUNT = 2
    #: ... and capitalised in at least this fraction of its occurrences.
    CASE_RATIO = 0.75

    def __init__(self) -> None:
        self._stats: Dict[str, EntityStat] = defaultdict(EntityStat)
        self._speakers: Set[str] = set()

    def record(self, text: str, global_seq: int) -> None:
        for m in _SPEAKER_RE.finditer(text):
            self._speakers.add(m.group(1))
        for sentence in re.split(r"(?<=[.!?:])\s+|\n", text):
            words = sentence.split()
            for w in words[1:]:
                name = w.strip(".,!?;:\"'()[]")
                if len(name) < 3 or not _CAP_RE.fullmatch(name):
                    continue
                if name in _STOP_CAPS:
                    continue
                st = self._stats[name]
                st.count += 1
                if st.first_seq < 0:
                    st.first_seq = global_seq
                st.last_seq = global_seq

        # Track lowercase occurrences of the same words to spot common nouns.
        for word in re.findall(r"\b([a-z]{3,})\b", text):
            cap = word.capitalize()
            if cap in self._stats:
                self._stats[cap].lower_count += 1

    def _is_proper(self, name: str, st: EntityStat) -> bool:
        if st.count < self.MIN_COUNT:
            return False
        total = st.count + st.lower_count
        return (st.count / total) >= self.CASE_RATIO if total else False

    def top_entities(self, n: int = 20, min_count: Optional[int] = None) -> List[str]:
        floor = self.MIN_COUNT if min_count is None else min_count
        candidates = [
            (k, v) for k, v in self._stats.items()
            if v.count >= floor and self._is_proper(k, v)
        ]
        candidates.sort(key=lambda kv: kv[1].count, reverse=True)

        out = sorted(self._speakers)          # speakers are always relevant
        for name, _ in candidates:
            if name not in out:
                out.append(name)
            if len(out) >= n:
                break
        return out[:n]

    @property
    def speakers(self) -> Set[str]:
        return set(self._speakers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stats": {k: vars(v) for k, v in self._stats.items()},
            "speakers": sorted(self._speakers),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityStats":
        obj = cls()
        for k, v in (data.get("stats") or {}).items():
            obj._stats[k] = EntityStat(**v)
        obj._speakers = set(data.get("speakers") or [])
        return obj

class ImageCaptionIndex:
    """Flat index over image captions, searched alongside the tree."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def add(self, episode_id: str, caption: str,
            image_path: Optional[str] = None,
            detailed_caption: Optional[str] = None) -> None:
        if not caption and not detailed_caption:
            return
        self._entries.append({
            "episode_id": episode_id,
            "caption": caption or "",
            "detailed_caption": detailed_caption or "",
            "image_path": image_path,
        })

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._entries:
            return []
        q = set(re.findall(r"\w+", query.lower()))
        if not q:
            return []
        scored = []
        for e in self._entries:
            combined = f"{e['caption']} {e.get('detailed_caption', '')}".lower()
            score = _fuzzy_token_overlap(q, set(re.findall(r"\w+", combined)))
            if score > 0.0:
                scored.append({**e, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @property
    def size(self) -> int:
        return len(self._entries)

@dataclass
class MemoryObject:
    """One episode: a bounded span of dialogue with its metadata."""

    id: str
    modality: Literal["text", "image", "video", "audio", "table", "other"]
    raw_text: str
    normalized_text: str
    metadata: Dict[str, Any]
    timestamp: Optional[str] = None
    source: Optional[str] = None

    token_count: int = 0
    episode_utterances: List[str] = field(default_factory=list)
    episode_speakers: List[str] = field(default_factory=list)

    global_seq: int = -1
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    evidence_ids: List[str] = field(default_factory=list)

    # Multimodal
    image_path: Optional[str] = None
    image_caption_concise: str = ""
    image_caption_detailed: str = ""

    _token_cache: Optional[Set[str]] = field(default=None, repr=False, compare=False)
    _stem_cache: Optional[Set[str]] = field(default=None, repr=False, compare=False)

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    @property
    def tokens(self) -> Set[str]:
        if self._token_cache is None:
            self._token_cache = set(re.findall(r"\w+", self.raw_text.lower()))
        return self._token_cache

    @property
    def stem_tokens(self) -> Set[str]:
        if self._stem_cache is None:
            self._stem_cache = _stem_set(self.tokens)
        return self._stem_cache

    def invalidate_lexical_cache(self) -> None:
        self._token_cache = None
        self._stem_cache = None

    @classmethod
    def create(cls, raw_text: str, modality: str = "text", **kw) -> "MemoryObject":
        meta = kw.get("metadata", {}) or {}
        ev = kw.get("evidence_ids")
        if ev is None:
            ev = [meta["dia_id"]] if meta.get("dia_id") else []
        return cls(
            id=kw.get("id", str(uuid.uuid4())),
            modality=modality,
            raw_text=raw_text,
            normalized_text=cls._norm(raw_text),
            metadata=meta,
            timestamp=kw.get("timestamp") or datetime.utcnow().isoformat(),
            source=kw.get("source"),
            token_count=len(raw_text.split()),
            episode_utterances=kw.get("episode_utterances", [raw_text]),
            episode_speakers=kw.get("episode_speakers",
                                    [meta.get("speaker", "Unknown")]),
            global_seq=kw.get("global_seq", -1),
            evidence_ids=list(ev),
            image_path=kw.get("image_path"),
            image_caption_concise=kw.get("image_caption_concise", ""),
            image_caption_detailed=kw.get("image_caption_detailed", ""),
        )

    @classmethod
    def from_utterances(cls, utterances: List[Dict[str, Any]],
                        modality: str = "text", **kw) -> "MemoryObject":
        texts = [u.get("text", "") for u in utterances]
        speakers = [u.get("speaker", "Unknown") for u in utterances]
        timestamps = [u.get("timestamp") for u in utterances if u.get("timestamp")]
        evidence = [u["dia_id"] for u in utterances if u.get("dia_id")]

        raw_text = " ".join(texts)
        metadata = kw.get("metadata", {}) or {}
        if speakers:
            metadata["speakers"] = list(dict.fromkeys(speakers))

        image_paths = [u.get("image_path") for u in utterances if u.get("image_path")]
        dataset_captions = [u.get("dataset_caption") for u in utterances
                            if u.get("dataset_caption")]

        return cls(
            id=kw.get("id", str(uuid.uuid4())),
            modality=modality,
            raw_text=raw_text,
            normalized_text=cls._norm(raw_text),
            metadata=metadata,
            timestamp=min(timestamps) if timestamps else datetime.utcnow().isoformat(),
            source=kw.get("source"),
            token_count=len(raw_text.split()),
            episode_utterances=texts,
            episode_speakers=speakers,
            global_seq=kw.get("global_seq", -1),
            evidence_ids=evidence,
            image_path=image_paths[0] if image_paths else kw.get("image_path"),
            image_caption_concise=kw.get("image_caption_concise", ""),
            image_caption_detailed=(dataset_captions[0] if dataset_captions
                                    else kw.get("image_caption_detailed", "")),
        )

def _jaccard_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _parse_timestamp(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, TypeError):
        return None


def segment_utterances_into_episodes(
    utterances: List[Dict[str, Any]], config: MemoryConfig
) -> List[List[Dict[str, Any]]]:
    """Split utterances at a token ceiling, temporal gap, topic shift or image."""
    if not utterances:
        return []

    episodes: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    tokens = 0
    window = ""

    for utt in utterances:
        text = utt.get("text", "")
        n = len(text.split())
        ts = _parse_timestamp(utt.get("timestamp"))
        has_image = bool(utt.get("image_path"))

        split = False
        if has_image and current:
            split = True
        if not split and tokens + n > config.episodic_max_tokens and tokens >= config.episodic_min_tokens:
            split = True
        if not split and current and ts is not None:
            prev = _parse_timestamp(current[-1].get("timestamp"))
            if prev is not None and (ts - prev) > config.temporal_gap_seconds:
                split = True
        if not split and tokens >= config.episodic_min_tokens:
            if _jaccard_overlap(window, text) < config.topic_shift_min_overlap:
                split = True

        if split and current:
            episodes.append(current)
            current, tokens, window = [], 0, ""

        current.append(utt)
        tokens += n
        window += " " + text

        if has_image:
            episodes.append(current)
            current, tokens, window = [], 0, ""

    if current:
        episodes.append(current)
    return episodes

_SUFFIXES = ["ing", "tion", "ations", "ation", "ness", "ment", "able", "ible",
             "ly", "ed", "er", "est", "es", "s"]


def _simple_stem(word: str) -> str:
    word = word.lower()
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _stem_set(tokens: Set[str]) -> Set[str]:
    return {_simple_stem(t) for t in tokens}


def _within_edit1(a: str, b: str) -> bool:
    la, lb = len(a), len(b)
    if la > lb:
        a, b, la, lb = b, a, lb, la
    if lb - la > 1:
        return False
    if la == lb:
        diff = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                diff += 1
                if diff > 1:
                    return False
        return True
    # lb == la + 1: b must equal a with exactly one character inserted.
    i = j = 0
    skipped = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


def _fuzzy_token_overlap(query_tokens: Set[str], doc_tokens: Set[str],
                         doc_stems: Optional[Set[str]] = None) -> float:
    if not query_tokens:
        return 0.0
    stemmed_doc = doc_stems if doc_stems is not None else _stem_set(doc_tokens)
    short_doc = [d for d in stemmed_doc if len(d) <= 6]
    matched = 0
    for qt in query_tokens:
        sq = _simple_stem(qt)
        if sq in stemmed_doc:
            matched += 1
            continue
        if len(sq) <= 6:
            for dt in short_doc:
                if _within_edit1(sq, dt):
                    matched += 1
                    break
    return matched / len(query_tokens)


class LeafNode:
    """Holds up to max_leaf_objects episodes."""

    def __init__(self, leaf_id: Optional[str] = None, parent_id: Optional[str] = None):
        self.leaf_id: str = leaf_id or str(uuid.uuid4())
        self.parent_id: Optional[str] = parent_id
        self.memory_objects: List[MemoryObject] = []
        self.raw_text: str = ""
        self.speakers: List[str] = []
        self.time_range: Optional[Dict[str, Optional[str]]] = None
        self.seq_range: Tuple[int, int] = (-1, -1)

    def append_memory(self, memory: MemoryObject) -> None:
        self.memory_objects.append(memory)
        self._refresh()

    def rebuild_from_memories(self, memories: List[MemoryObject]) -> None:
        self.memory_objects = list(memories)
        self._refresh()

    def _refresh(self) -> None:
        self.speakers = [m.metadata.get("speaker", "Unknown") for m in self.memory_objects]
        self.raw_text = " ".join(m.raw_text for m in self.memory_objects)
        ts = [m.timestamp for m in self.memory_objects if m.timestamp]
        if ts:
            s = sorted(ts)
            self.time_range = {"start": s[0], "end": s[-1]}
        seqs = [m.global_seq for m in self.memory_objects if m.global_seq >= 0]
        if seqs:
            self.seq_range = (min(seqs), max(seqs))


class RegionNode:
    """Internal node summarised by anchor terms and a first-sentence prefix."""

    def __init__(self, node_id: Optional[str] = None, parent_id: Optional[str] = None):
        self.node_id: str = node_id or str(uuid.uuid4())
        self.parent_id: Optional[str] = parent_id
        self.children: List[str] = []
        self.is_leaf_parent: bool = False

        self.summary_text: str = ""      # first-sentence prefix s_r
        self.anchor_terms: List[str] = []
        self.key_entities: List[str] = []

        self.time_range: Optional[Dict[str, Optional[str]]] = None
        self.seq_range: Tuple[int, int] = (-1, -1)
        self.image_count: int = 0


def compute_anchor_score(query_tokens: Set[str], anchor_terms: List[str]) -> float:
    if not anchor_terms:
        return 0.0
    return _fuzzy_token_overlap(query_tokens, {t.lower() for t in anchor_terms})


def compute_lexical_overlap(query_tokens: Set[str], text: str) -> float:
    if not text:
        return 0.0
    return _fuzzy_token_overlap(query_tokens, set(re.findall(r"\w+", text.lower())))


def evaluate_routing_score(text: str, child: Union[RegionNode, LeafNode],
                           config: MemoryConfig) -> float:
    """Eq. 1: greedy top-down routing score (anchor terms + prefix overlap)."""
    q = set(re.findall(r"\w+", text.lower()))
    if isinstance(child, RegionNode):
        return (config.routing_weight_anchor * compute_anchor_score(q, child.anchor_terms)
                + config.routing_weight_lexical * compute_lexical_overlap(q, child.summary_text))
    return compute_lexical_overlap(q, child.raw_text)

class RecursiveMemoryLayer:
    """LLM-free hierarchical storage."""

    def __init__(self, config: MemoryConfig, llm_controller: Any = None):
        self.config = config
        self.llm = llm_controller

        self.nodes: Dict[str, Union[RegionNode, LeafNode]] = {}
        self.root_id: str = str(uuid.uuid4())
        root = RegionNode(node_id=self.root_id)
        root.is_leaf_parent = True
        self.nodes[self.root_id] = root

        self.entity_stats = EntityStats()
        self._global_seq_counter: int = 0

        self._embedder: Optional[EpisodeEmbedder] = (
            EpisodeEmbedder(config.embedding_model) if config.enable_embeddings else None
        )

        self._image_processor: Optional[ImageProcessor] = None
        self._image_caption_index: Optional[ImageCaptionIndex] = None
        if config.enable_multimodal:
            self._image_processor = ImageProcessor(mllm=llm_controller)
            self._image_caption_index = ImageCaptionIndex()

        # Episode id -> MemoryObject, so retrieval never has to walk the tree.
        self._episodes_by_id: Dict[str, MemoryObject] = {}

        # Incremental BM25 over episode text. Built on the write path because
        # adding a document is O(its tokens) and never triggers a rebuild, so
        # this preserves the LLM-free, near-instant insert the paper claims.
        self.bm25 = BM25Index()

        # Phase 3: the evolving semantic layer. Episodes are immutable; notes
        # are distilled once per session and may be superseded over time.
        self.notes = NoteStore()

    # -- insertion -------------------------------------------------------

    def add_memory(self, raw_text: str, metadata: Optional[Dict[str, Any]] = None,
                   modality: str = "text", source: Optional[str] = None,
                   timestamp: Optional[str] = None,
                   image_path: Optional[str] = None,
                   evidence_ids: Optional[List[str]] = None) -> MemoryObject:
        mem = MemoryObject.create(
            raw_text=raw_text, metadata=metadata or {}, modality=modality,
            source=source, timestamp=timestamp, image_path=image_path,
            evidence_ids=evidence_ids,
        )
        self.insert(mem)
        return mem

    def add_memories_bulk(self, items: List[Dict[str, Any]],
                          modality: str = "text") -> List[MemoryObject]:
        if not items:
            return []

        memories = [
            MemoryObject.create(
                raw_text=it["raw_text"],
                modality=modality,
                metadata=it.get("metadata", {}),
                timestamp=it.get("timestamp"),
                source=it.get("source"),
                image_path=it.get("image_path"),
                evidence_ids=it.get("evidence_ids"),
            )
            for it in items
        ]

        # In multimodal mode the caption is fused into raw_text during insert(),
        # so pre-embedding here would miss it; let insert() embed instead.
        if self._embedder is not None and not self.config.enable_multimodal:
            vecs = self._embedder.batch_embed([m.raw_text for m in memories])
            for m, v in zip(memories, vecs):
                m.embedding = v

        for m in memories:
            self.insert(m)
        return memories

    def add_episode_from_utterances(self, utterances: List[Dict[str, Any]],
                                    modality: str = "text",
                                    source: Optional[str] = None) -> None:
        groups = segment_utterances_into_episodes(utterances, self.config)
        memories = [MemoryObject.from_utterances(g, modality, source=source)
                    for g in groups]
        if self._embedder is not None and not self.config.enable_multimodal:
            vecs = self._embedder.batch_embed([m.raw_text for m in memories])
            for m, v in zip(memories, vecs):
                if m.embedding is None:
                    m.embedding = v
        for m in memories:
            self.insert(m)

    def insert(self, memory: MemoryObject) -> None:
        memory.global_seq = self._global_seq_counter
        self._global_seq_counter += 1

        if self.config.enable_multimodal and memory.image_path and self._image_processor:
            self._process_image(memory)

        if self._embedder is not None and memory.embedding is None:
            memory.embedding = self._embedder.embed(memory.raw_text)

        self.entity_stats.record(memory.raw_text, memory.global_seq)
        self._episodes_by_id[memory.id] = memory
        # After _process_image, so fused captions are searchable lexically.
        self.bm25.add(memory.id, memory.raw_text)

        target_id = self._find_candidate_leaf(memory.raw_text)
        node = self.nodes.get(target_id)
        if isinstance(node, LeafNode):
            leaf = node
        else:
            leaf = LeafNode(parent_id=target_id)
            self.nodes[leaf.leaf_id] = leaf
            parent = self.nodes.get(target_id)
            if isinstance(parent, RegionNode):
                parent.children.append(leaf.leaf_id)
                self._refresh_child_flags(parent.node_id)

        leaf.append_memory(memory)
        self._refresh_ancestors(leaf.leaf_id)
        self._check_and_split(leaf.leaf_id)

    def _process_image(self, memory: MemoryObject) -> None:
        ip = self._image_processor
        if memory.image_caption_detailed and not memory.image_caption_concise:
            words = memory.image_caption_detailed.split()
            memory.image_caption_concise = " ".join(words[:20]) + ("..." if len(words) > 20 else "")
        elif not memory.image_caption_concise:
            memory.image_caption_concise = ip.caption_concise(memory.image_path)

        if self.config.eager_detailed_caption and not memory.image_caption_detailed:
            memory.image_caption_detailed = ip.caption_detailed(memory.image_path)

        if "[Image caption:" not in memory.raw_text:
            cap = memory.image_caption_detailed or memory.image_caption_concise
            if cap:
                memory.raw_text += f" [Image: {cap}]"
        memory.normalized_text = MemoryObject._norm(memory.raw_text)
        memory.invalidate_lexical_cache()

        if self._image_caption_index is not None:
            self._image_caption_index.add(
                episode_id=memory.id,
                caption=memory.image_caption_concise,
                image_path=memory.image_path,
                detailed_caption=memory.image_caption_detailed,
            )

    # -- tree maintenance ------------------------------------------------

    def _find_candidate_leaf(self, text: str) -> str:
        current = self.root_id
        while True:
            node = self.nodes.get(current)
            if node is None:
                return self.root_id
            if isinstance(node, LeafNode):
                return node.leaf_id
            if not node.children:
                return current
            best_score, best_child = -1.0, node.children[0]
            for cid in node.children:
                child = self.nodes.get(cid)
                if child is None:
                    continue
                score = evaluate_routing_score(text, child, self.config)
                if score > best_score:
                    best_score, best_child = score, cid
            current = best_child

    def _refresh_child_flags(self, region_id: str) -> None:
        region = self.nodes.get(region_id)
        if not isinstance(region, RegionNode):
            return
        kinds = [isinstance(self.nodes.get(c), LeafNode) for c in region.children
                 if c in self.nodes]
        region.is_leaf_parent = bool(kinds) and all(kinds)

    def _refresh_ancestors(self, node_id: str) -> None:
        """Recompute routing metadata up the path to the root."""
        current: Optional[str] = self.nodes.get(node_id).parent_id if node_id in self.nodes else None
        while current:
            self._recompute_region(current)
            node = self.nodes.get(current)
            current = node.parent_id if node else None

    def _recompute_region(self, node_id: str) -> None:
        """Anchor terms A_r and first-sentence prefix s_r from descendants."""
        node = self.nodes.get(node_id)
        if not isinstance(node, RegionNode):
            return

        texts: List[str] = []
        starts: List[str] = []
        ends: List[str] = []
        seqs: List[int] = []
        image_count = 0
        first_text = ""

        for cid in node.children:
            child = self.nodes.get(cid)
            if child is None:
                continue
            if isinstance(child, LeafNode):
                texts.append(child.raw_text)
                if not first_text and child.raw_text:
                    first_text = child.raw_text
                image_count += sum(1 for m in child.memory_objects if m.image_path)
            else:
                texts.append(child.summary_text)
                if not first_text and child.summary_text:
                    first_text = child.summary_text
                image_count += child.image_count
            if child.seq_range[0] >= 0:
                seqs.extend(child.seq_range)
            if child.time_range:
                if child.time_range.get("start"):
                    starts.append(child.time_range["start"])
                if child.time_range.get("end"):
                    ends.append(child.time_range["end"])

        aggregated = "\n".join(t for t in texts if t)

        # Anchor terms: most frequent content words of length >= 5.
        words = re.findall(r"\b[a-zA-Z]{5,}\b", aggregated)
        node.anchor_terms = [w for w, _ in Counter(w.lower() for w in words)
                             .most_common(self.config.max_anchor_terms)]

        # First-sentence prefix.
        sentences = re.split(r"(?<=[.!?])\s+", first_text.strip())
        node.summary_text = sentences[0][:200] if sentences and sentences[0] else first_text[:200]

        node.key_entities = [e for e in dict.fromkeys(_CAP_RE.findall(aggregated))
                             if e not in _STOP_CAPS][:5]
        if starts and ends:
            node.time_range = {"start": min(starts), "end": max(ends)}
        if seqs:
            node.seq_range = (min(seqs), max(seqs))
        node.image_count = image_count

    def _check_and_split(self, node_id: str) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        if isinstance(node, LeafNode):
            if len(node.memory_objects) > self.config.max_leaf_objects:
                parent = self._split_leaf(node)
                if parent:
                    self._check_and_split(parent)
        elif len(node.children) > self.config.max_region_children:
            parent = self._split_region(node)
            if parent:
                self._check_and_split(parent)

    def _split_leaf(self, leaf: LeafNode) -> Optional[str]:
        if not leaf.parent_id:
            return None
        parent = self.nodes.get(leaf.parent_id)
        if not isinstance(parent, RegionNode):
            return None
        memories = sorted(leaf.memory_objects, key=lambda m: m.global_seq)
        mid = len(memories) // 2
        leaf.rebuild_from_memories(memories[:mid])
        sibling = LeafNode(parent_id=parent.node_id)
        sibling.rebuild_from_memories(memories[mid:])
        self.nodes[sibling.leaf_id] = sibling
        parent.children.append(sibling.leaf_id)
        self._refresh_child_flags(parent.node_id)
        self._recompute_region(parent.node_id)
        return parent.node_id

    def _split_region(self, region: RegionNode) -> Optional[str]:
        if not region.parent_id:
            new_root = RegionNode()
            self.nodes[new_root.node_id] = new_root
            new_root.children.append(region.node_id)
            region.parent_id = new_root.node_id
            self.root_id = new_root.node_id
            self._refresh_child_flags(new_root.node_id)

        parent = self.nodes.get(region.parent_id)
        if not isinstance(parent, RegionNode):
            return None
        mid = len(region.children) // 2
        sibling = RegionNode(parent_id=parent.node_id)
        sibling.children = region.children[mid:]
        region.children = region.children[:mid]
        for cid in sibling.children:
            child = self.nodes.get(cid)
            if child:
                child.parent_id = sibling.node_id
        self.nodes[sibling.node_id] = sibling
        parent.children.append(sibling.node_id)
        for nid in (region.node_id, sibling.node_id, parent.node_id):
            self._refresh_child_flags(nid)
            self._recompute_region(nid)
        return parent.node_id

    # -- accessors -------------------------------------------------------

    def all_episodes(self) -> List[MemoryObject]:
        return list(self._episodes_by_id.values())

    def get_episode(self, episode_id: str) -> Optional[MemoryObject]:
        return self._episodes_by_id.get(episode_id)

    def get_image_caption_index(self) -> Optional[ImageCaptionIndex]:
        return self._image_caption_index

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "config": self.config,
                "nodes": self.nodes,
                "root_id": self.root_id,
                "global_seq_counter": self._global_seq_counter,
                "entity_stats": self.entity_stats.to_dict(),
                "episodes_by_id": self._episodes_by_id,
                "bm25": self.bm25,
                "notes": self.notes,
            }, f)

    @classmethod
    def load(cls, path: str, llm_controller: Any = None) -> "RecursiveMemoryLayer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(config=data["config"], llm_controller=llm_controller)
        obj.nodes = data["nodes"]
        obj.root_id = data["root_id"]
        obj._global_seq_counter = data.get("global_seq_counter", 0)
        obj.entity_stats = EntityStats.from_dict(data.get("entity_stats", {}))
        obj._episodes_by_id = data.get("episodes_by_id", {})
        notes = data.get("notes")
        if notes is not None:
            obj.notes = notes
        bm25 = data.get("bm25")
        if bm25 is not None:
            obj.bm25 = bm25
        else:                       # cache written before BM25 existed
            for m in obj._episodes_by_id.values():
                obj.bm25.add(m.id, m.raw_text)
        return obj

    # -- inspection ------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        regions = sum(1 for n in self.nodes.values() if isinstance(n, RegionNode))
        leaves = sum(1 for n in self.nodes.values() if isinstance(n, LeafNode))
        stats = {
            "total_nodes": len(self.nodes),
            "total_regions": regions,
            "total_leaves": leaves,
            "total_memories": len(self._episodes_by_id),
            "known_entities": len(self.entity_stats.top_entities(n=10_000, min_count=1)),
            "tree_depth": self._depth(self.root_id),
            "multimodal_enabled": self.config.enable_multimodal,
            **{f"note_{k}": v for k, v in self.notes.stats().items()},
        }
        if self.config.enable_multimodal:
            stats["image_episodes"] = sum(
                1 for m in self._episodes_by_id.values() if m.image_path
            )
            stats["image_caption_index_size"] = (
                self._image_caption_index.size if self._image_caption_index else 0
            )
        return stats

    def _depth(self, node_id: str) -> int:
        node = self.nodes.get(node_id)
        if node is None or isinstance(node, LeafNode) or not node.children:
            return 1
        return 1 + max(self._depth(c) for c in node.children)

    def print_tree(self, node_id: Optional[str] = None, depth: int = 0) -> None:
        node_id = node_id or self.root_id
        node = self.nodes.get(node_id)
        if node is None:
            return
        pad = "  " * depth
        if isinstance(node, RegionNode):
            print(f"{pad}[Region] {node.node_id[:8]} children={len(node.children)} "
                  f"seq={node.seq_range} anchors={node.anchor_terms[:4]}")
            for cid in node.children:
                self.print_tree(cid, depth + 1)
        else:
            print(f"{pad}[Leaf] {node.leaf_id[:8]} memories={len(node.memory_objects)} "
                  f"seq={node.seq_range}")
