"""
metrics.py — evaluation metrics for MemFit.
"""

from __future__ import annotations

import math
import re
import statistics
import string
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: Any) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: Any) -> List[str]:
    return normalize_answer(s).split()

def token_f1(prediction: Any, reference: Any) -> float:
    """SQuAD-style token F1 using multiset (Counter) overlap."""
    pred = _tokens(prediction)
    gold = _tokens(reference)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: Any, reference: Any) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def bleu1(prediction: Any, reference: Any) -> float:
    pred = _tokens(prediction)
    gold = _tokens(reference)
    if not pred or not gold:
        return 0.0
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if overlap == 0:
        # method1 smoothing: replace a zero numerator with a small epsilon
        precision = 0.1 / len(pred)
    else:
        precision = overlap / len(pred)
    # brevity penalty
    if len(pred) >= len(gold):
        bp = 1.0
    else:
        bp = math.exp(1 - len(gold) / len(pred))
    return bp * precision


def answer_metrics(prediction: Any, reference: Any) -> Dict[str, float]:
    """All answer-quality metrics for one (prediction, reference) pair."""
    return {
        "f1": token_f1(prediction, reference),
        "bleu1": bleu1(prediction, reference),
        "exact_match": exact_match(prediction, reference),
    }

def retrieval_metrics(
    retrieved_ids: Sequence[str],
    gold_ids: Iterable[str],
    ks: Sequence[int] = (5, 10, 20, 30),
) -> Dict[str, float]:
    gold: Set[str] = {g for g in gold_ids if g}
    out: Dict[str, float] = {"n_gold": float(len(gold))}
    if not gold:
        return out

    ranked = list(retrieved_ids)
    for k in ks:
        topk = set(ranked[:k])
        found = len(gold & topk)
        out[f"recall@{k}"] = found / len(gold)
        out[f"hit@{k}"] = 1.0 if found > 0 else 0.0
        out[f"all@{k}"] = 1.0 if found == len(gold) else 0.0

    mrr = 0.0
    for rank, rid in enumerate(ranked, start=1):
        if rid in gold:
            mrr = 1.0 / rank
            break
    out["mrr"] = mrr
    return out

def _summarise(values: List[float], n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Mean with a bootstrap 95% CI (reviewers asked for error bars)."""
    if not values:
        return {"mean": 0.0, "n": 0, "ci_low": 0.0, "ci_high": 0.0}
    mean = statistics.mean(values)
    n = len(values)
    if n < 2:
        return {"mean": mean, "n": n, "ci_low": mean, "ci_high": mean}

    import random

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return {
        "mean": mean,
        "n": n,
        "std": statistics.stdev(values),
        "ci_low": lo,
        "ci_high": hi,
    }


def aggregate(
    records: List[Dict[str, Any]],
    metric_keys: Optional[Sequence[str]] = None,
    category_key: str = "category",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not records:
        return {}

    if metric_keys is None:
        keys: Set[str] = set()
        for r in records:
            for k, v in r.items():
                if k != category_key and isinstance(v, (int, float)):
                    keys.add(k)
        metric_keys = sorted(keys)

    overall: Dict[str, List[float]] = defaultdict(list)
    per_cat: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        cat = r.get(category_key)
        for k in metric_keys:
            v = r.get(k)
            if isinstance(v, (int, float)):
                overall[k].append(float(v))
                per_cat[cat][k].append(float(v))

    result: Dict[str, Dict[str, Dict[str, float]]] = {
        "overall": {k: _summarise(v) for k, v in overall.items()}
    }
    for cat in sorted(per_cat.keys(), key=lambda c: str(c)):
        result[f"category_{cat}"] = {k: _summarise(v) for k, v in per_cat[cat].items()}
    return result

LOCOMO_CATEGORIES: Dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def category_name(cat: Any) -> str:
    try:
        return LOCOMO_CATEGORIES.get(int(cat), f"cat_{cat}")
    except (TypeError, ValueError):
        return f"cat_{cat}"
