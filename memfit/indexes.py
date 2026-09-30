"""
indexes.py — retrieval indexes shared by the memory layer and the baselines.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._doc_len: Dict[str, int] = {}
        self._df: Counter = Counter()
        self._total_len: int = 0

    # -- construction ----------------------------------------------------

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._doc_len:      # idempotent re-add
            return
        tokens = tokenize(text)
        if not tokens:
            self._doc_len[doc_id] = 0
            return
        tf = Counter(tokens)
        for term, count in tf.items():
            self._postings[term].append((doc_id, count))
            self._df[term] += 1
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)

    def add_many(self, docs: Iterable[Tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add(doc_id, text)

    # -- properties ------------------------------------------------------

    @property
    def n_docs(self) -> int:
        return len(self._doc_len)

    @property
    def avgdl(self) -> float:
        return (self._total_len / self.n_docs) if self.n_docs else 1.0

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        # BM25+ style idf, always positive so common terms never subtract.
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    # -- query -----------------------------------------------------------

    def score_all(self, query: str) -> Dict[str, float]:
        """BM25 score for every document that shares a term with the query."""
        terms = tokenize(query)
        if not terms or not self.n_docs:
            return {}
        avgdl = self.avgdl
        scores: Dict[str, float] = defaultdict(float)
        # Counter over query terms so a repeated query term counts once per
        # occurrence, matching the standard formulation.
        for term, q_count in Counter(terms).items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            if idf <= 0.0:
                continue
            for doc_id, tf in postings:
                dl = self._doc_len.get(doc_id, 0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                if denom > 0:
                    scores[doc_id] += q_count * idf * (tf * (self.k1 + 1.0)) / denom
        return dict(scores)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        scores = self.score_all(query)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

def minmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def rrf(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal rank fusion, kept so the fusion choice can be ablated."""
    out: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            out[doc_id] += 1.0 / (k + rank)
    return dict(out)

class CrossEncoderReranker:
    """
    Reranks a candidate pool against the ORIGINAL question.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 384):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._failed = False

    @property
    def available(self) -> bool:
        if self._failed:
            return False
        if self._model is not None:
            return True
        try:
            import os
            from sentence_transformers import CrossEncoder

            device = (getattr(self, "_device", None)
                      or os.environ.get("MEMFIT_ENCODER_DEVICE") or None)
            self._model = CrossEncoder(self.model_name, max_length=self.max_length,
                                       device=device)
            return True
        except Exception as e:
            import logging
            logging.getLogger("indexes").error(
                "reranker %s failed to load: %s", self.model_name, e)
            self._failed = True
            return False

    def score(self, query: str, passages: List[str]) -> Optional[List[float]]:
        if not passages or not self.available:
            return None
        try:
            pairs = [(query, p) for p in passages]
            scores = [float(s) for s in self._model.predict(pairs, show_progress_bar=False)]
        except Exception as e:
            # Recorded, and fatal at the end of the run (encoders.final_check):
            # once this is set every later query goes unreranked.
            import logging
            logging.getLogger("indexes").error("reranker failed mid-run: %s", e)
            self._failed = True
            return None
        # A NaN score raises nothing and sorts arbitrarily, so a model that has
        # gone numerically bad would silently scramble the ranking. Seen for
        # real: ms-marco-MiniLM-L-6-v2 returns all-NaN when forced onto one
        # CPU build. Treated exactly like a crash — fatal at final_check.
        if any(s != s for s in scores):
            import logging
            logging.getLogger("indexes").error(
                "reranker %s returned NaN scores — treating as a failure",
                self.model_name)
            self._failed = True
            return None
        return scores


_default_reranker: Optional[CrossEncoderReranker] = None


def get_default_reranker() -> CrossEncoderReranker:
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = CrossEncoderReranker()
    return _default_reranker
