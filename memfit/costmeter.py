#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List

_lock = threading.Lock()
_acc: Dict[str, float] = defaultdict(float)
_installed = False
_count_tokens = True


def _add(key: str, value: float) -> None:
    with _lock:
        _acc[key] += value


def _embed_tokens(embedder: Any, texts: List[str]) -> int:
    model = getattr(embedder, "_model", None)
    tok = getattr(model, "tokenizer", None)
    if tok is None or not texts:
        return 0
    try:
        limit = int(getattr(model, "max_seq_length", 256) or 256)
        enc = tok(list(texts), add_special_tokens=True, truncation=True,
                  max_length=limit)
        return int(sum(len(ids) for ids in enc["input_ids"]))
    except Exception:
        return 0


def _pair_tokens(reranker: Any, query: str, passages: List[str]) -> int:
    model = getattr(reranker, "_model", None)
    tok = getattr(model, "tokenizer", None)
    if tok is None or not passages:
        return 0
    try:
        limit = int(getattr(reranker, "max_length", 384) or 384)
        enc = tok([query] * len(passages), list(passages), truncation=True,
                  max_length=limit)
        return int(sum(len(ids) for ids in enc["input_ids"]))
    except Exception:
        return 0


def install(count_tokens: bool = True) -> None:
    """Wrap the embedder and the reranker. Idempotent."""
    global _installed, _count_tokens
    if _installed:
        return
    _count_tokens = count_tokens

    import indexes
    import memory_layer

    E = memory_layer.EpisodeEmbedder
    embed0, batch0, load0 = E.embed, E.batch_embed, E._load

    # Loading the encoder is a once-per-process start-up cost, not part of any
    # conversation's build or any question's retrieval, so it gets its own
    # line and is subtracted from the embed call that happened to trigger it.
    def _load(self):
        if getattr(self, "_model", None) is not None:
            return load0(self)
        t = time.perf_counter()
        try:
            return load0(self)
        finally:
            _add("encoder_load_s", time.perf_counter() - t)

    def _loaded_during(before: float) -> float:
        with _lock:
            return _acc.get("encoder_load_s", 0.0) - before

    def _load_mark() -> float:
        with _lock:
            return _acc.get("encoder_load_s", 0.0)

    # embed and batch_embed are independent code paths (batch_embed does not
    # call embed), so wrapping both cannot double-count.
    def embed(self, text):
        l0 = _load_mark()
        t = time.perf_counter()
        try:
            return embed0(self, text)
        finally:
            _add("embed_s", time.perf_counter() - t - _loaded_during(l0))
            _add("embed_texts", 1)
            if _count_tokens:
                _add("embed_tokens", _embed_tokens(self, [text]))

    def batch_embed(self, texts):
        l0 = _load_mark()
        t = time.perf_counter()
        try:
            return batch0(self, texts)
        finally:
            _add("embed_s", time.perf_counter() - t - _loaded_during(l0))
            _add("embed_texts", len(texts or []))
            if _count_tokens:
                _add("embed_tokens", _embed_tokens(self, list(texts or [])))

    E._load, E.embed, E.batch_embed = _load, embed, batch_embed

    R = indexes.CrossEncoderReranker
    score0 = R.score

    def score(self, query, passages):
        t = time.perf_counter()
        try:
            return score0(self, query, passages)
        finally:
            _add("rerank_s", time.perf_counter() - t)
            _add("rerank_calls", 1)
            _add("rerank_pairs", len(passages or []))
            if _count_tokens:
                _add("rerank_tokens", _pair_tokens(self, query, list(passages or [])))

    R.score = score
    _installed = True


def installed() -> bool:
    return _installed


def begin() -> Dict[str, float]:
    """A snapshot to measure from."""
    with _lock:
        return dict(_acc)


def since(mark: Dict[str, float]) -> Dict[str, float]:
    """Per-stage totals accumulated after `mark` was taken."""
    with _lock:
        out = {k: v - mark.get(k, 0.0) for k, v in _acc.items()}
    return {k: (int(round(v)) if not k.endswith("_s") else round(v, 6))
            for k, v in out.items() if v}


def percentile(values: List[float], q: float) -> float:
    """Nearest-rank percentile; no numpy so the harness import stays light."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, int(round(q / 100.0 * (len(xs) - 1)))))
    return float(xs[k])


def question_cost(stages: Dict[str, float], latency_s: float,
                  usage: Dict[str, Any]) -> Dict[str, Any]:
    reader_s = float(usage.get("llm_latency_s") or 0.0)
    embed_s = float(stages.get("embed_s", 0.0))
    rerank_s = float(stages.get("rerank_s", 0.0))
    load_s = float(stages.get("encoder_load_s", 0.0))   # once per process, if at all
    other = latency_s - reader_s - embed_s - rerank_s - load_s
    return {
        "total_s": round(latency_s - load_s, 6),
        "encoder_load_s": round(load_s, 6),
        "reader_s": round(reader_s, 6),
        "embed_s": round(embed_s, 6),
        "rerank_s": round(rerank_s, 6),
        "retrieval_other_s": round(max(0.0, other), 6),
        "retrieval_other_raw_s": round(other, 6),
        "retrieval_s": round(latency_s - reader_s - load_s, 6),
        "embed_tokens": int(stages.get("embed_tokens", 0)),
        "embed_texts": int(stages.get("embed_texts", 0)),
        "rerank_pairs": int(stages.get("rerank_pairs", 0)),
        "rerank_tokens": int(stages.get("rerank_tokens", 0)),
    }


def summarise_questions(costs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Totals and latency percentiles over a run's questions."""
    if not costs:
        return {}
    keys = ("total_s", "reader_s", "embed_s", "rerank_s", "retrieval_other_s",
            "retrieval_s")
    out: Dict[str, Any] = {"n": len(costs)}
    for k in keys:
        vals = [c.get(k, 0.0) for c in costs]
        out[f"{k}_sum"] = round(sum(vals), 3)
        out[f"{k}_mean"] = round(sum(vals) / len(vals), 4)
        out[f"{k}_p50"] = round(percentile(vals, 50), 4)
        out[f"{k}_p95"] = round(percentile(vals, 95), 4)
    for k in ("embed_tokens", "embed_texts", "rerank_pairs", "rerank_tokens"):
        out[f"{k}_sum"] = int(sum(c.get(k, 0) for c in costs))
    return out
