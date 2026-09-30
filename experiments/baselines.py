"""
baselines.py — matched-budget retrieval baselines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from indexes import BM25Index
from memory_layer import MemoryObject, RecursiveMemoryLayer
from rlm_controller import RLMConfig, Reader


class _EpisodeBM25:
    """Adapter: shared BM25Index keyed by episode id -> positional indices."""

    def __init__(self, episodes: List[MemoryObject]):
        self._episodes = episodes
        self._pos = {m.id: i for i, m in enumerate(episodes)}
        self._index = BM25Index()
        self._index.add_many((m.id, m.raw_text) for m in episodes)

    def search(self, query: str, top_k: int) -> List[int]:
        return [self._pos[doc_id]
                for doc_id, _ in self._index.search(query, top_k)
                if doc_id in self._pos]


class BaselineRetriever:
    """
    mode:
      "bm25"        lexical only
      "dense"       embedding only (same encoder as MemFit)
      "hybrid"      union of top-k from both (MemFit's retriever, no planner)
      "fullcontext" every episode, in order, no retrieval
    """

    def __init__(self, layer: RecursiveMemoryLayer, mode: str,
                 rlm_config: Optional[RLMConfig] = None, top_k: int = 10,
                 measure_depth: int = 50):
        self.layer = layer
        self.mode = mode
        self.top_k = top_k
        self.measure_depth = measure_depth
        self.config = rlm_config or RLMConfig()
        self.reader = Reader(self.config)
        self._episodes = layer.all_episodes()
        self._episodes.sort(key=lambda m: m.global_seq)
        self._bm25 = _EpisodeBM25(self._episodes) if mode in ("bm25", "hybrid") else None
        self._embedder = getattr(layer, "_embedder", None)

    def _dense(self, query: str, top_k: int) -> List[int]:
        if self._embedder is None:
            return []
        qv = self._embedder.embed(query)
        if qv is None:
            return []
        scores = []
        for i, m in enumerate(self._episodes):
            scores.append(float(np.dot(qv, m.embedding)) if m.embedding is not None else -1.0)
        order = np.argsort(-np.asarray(scores))
        return [int(i) for i in order[:top_k]]

    def _select(self, query: str, k: int) -> List[int]:
        if self.mode == "fullcontext":
            return list(range(len(self._episodes)))
        if self.mode == "bm25":
            return self._bm25.search(query, k)
        if self.mode == "dense":
            return self._dense(query, k)
        if self.mode == "hybrid":
            # Union of top-k from each view, interleaved so both are represented.
            lex = self._bm25.search(query, k)
            sem = self._dense(query, k)
            out, seen = [], set()
            for i in range(max(len(lex), len(sem))):
                for src in (sem, lex):
                    if i < len(src) and src[i] not in seen:
                        seen.add(src[i])
                        out.append(src[i])
                        if len(out) >= k:
                            return out
            return out
        raise ValueError(f"unknown baseline mode: {self.mode}")

    def retrieve(self, query: str, question_image_path: Optional[str] = None,
                 **_ignored) -> Dict[str, Any]:
        ranked = self._select(query, max(self.top_k, self.measure_depth))
        idxs = ranked if self.mode == "fullcontext" else ranked[: self.top_k]
        episodes = [self._episodes[i] for i in idxs]
        episodes.sort(key=lambda m: m.global_seq)

        def _evidence(indices: List[int]) -> List[str]:
            out: List[str] = []
            for i in indices:
                for ev in self._episodes[i].evidence_ids:
                    if ev not in out:
                        out.append(ev)
            return out

        evidence = _evidence(idxs)
        candidates = _evidence(ranked)

        result = self.reader.read(query, episodes, self.layer.llm,
                                  question_image_path=question_image_path)
        return {
            "query": query,
            "answer": result["answer"],
            "retrieved_episode_ids": [self._episodes[i].id for i in idxs],
            "retrieved_evidence_ids": evidence,
            "candidate_evidence_ids": candidates,
            "retrieved_scores": [],
            "plan_steps": 0,
            "plan_from_llm": False,
            "trajectory": [{"action": "baseline_retrieve", "mode": self.mode,
                            "n_excerpts": len(episodes)},
                           {"action": "read", "n_excerpts": len(episodes),
                            "prompt_chars": result["prompt_chars"]}],
        }
