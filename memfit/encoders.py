from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("encoders")

DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EncoderFailure(RuntimeError):
    """An encoder failed to load or failed mid-run. The run is not valid."""


def configure_reranker(model_name: Optional[str]) -> None:
    import indexes
    name = model_name or DEFAULT_RERANKER
    device = os.environ.get("MEMFIT_ENCODER_DEVICE") or None
    indexes._default_reranker = indexes.CrossEncoderReranker(name)
    indexes._default_reranker._device = device


def startup_check(layer: Any, want_embedder: str, rerank: bool) -> Dict[str, Any]:
    got = getattr(getattr(layer, "config", None), "embedding_model", None)
    if got != want_embedder:
        raise EncoderFailure(
            f"layer was built with embedder {got!r} but {want_embedder!r} was "
            f"requested — a stale cache. Use a cache_dir specific to this embedder.")

    emb = getattr(layer, "_embedder", None)
    if emb is None:
        raise EncoderFailure("layer has no embedder; dense retrieval would be off")
    probe = emb.embed("probe sentence for the encoder check")
    if probe is None:
        raise EncoderFailure(f"embedder {want_embedder!r} failed to embed")

    stored = next((m.embedding for m in layer.all_episodes()
                   if getattr(m, "embedding", None) is not None), None)
    if stored is None:
        raise EncoderFailure("layer holds no episode embeddings")
    if np.asarray(stored).shape != np.asarray(probe).shape:
        raise EncoderFailure(
            f"dimension mismatch: stored {np.asarray(stored).shape} vs query "
            f"{np.asarray(probe).shape} — layer and embedder disagree")

    info = {"embedder": want_embedder, "embedding_dim": int(np.asarray(probe).shape[-1])}
    if rerank:
        from indexes import get_default_reranker
        rr = get_default_reranker()
        s = rr.score("what did caroline research?",
                     ["Caroline researched adoption agencies.", "The weather was mild."])
        if s is None or len(s) != 2:
            raise EncoderFailure(f"reranker {rr.model_name!r} failed to score")
        if not s[0] > s[1]:
            raise EncoderFailure(
                f"reranker {rr.model_name!r} ranked an irrelevant passage above a "
                f"relevant one on the probe ({s}) — it is not working")
        info["reranker"] = rr.model_name
    else:
        info["reranker"] = None
    return info


def final_check(rerank: bool) -> Dict[str, Any]:
    import memory_layer
    if memory_layer.EMBED_FAILURES:
        raise EncoderFailure(
            f"{memory_layer.EMBED_FAILURES} texts failed to embed during the run; "
            f"those queries or episodes had no dense retrieval. Invalid run.")
    if not rerank:
        return {"reranker_failed": False, "embed_failures": 0}
    from indexes import get_default_reranker
    rr = get_default_reranker()
    if getattr(rr, "_failed", False):
        raise EncoderFailure(
            f"reranker {rr.model_name!r} failed mid-run and disabled itself; "
            f"some or all queries were answered without reranking. Invalid run.")
    return {"reranker_failed": False, "embed_failures": 0}
