#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("grouping")

#: Fraction of variance the projection must retain. A property of the
#: estimator: too little loses the topic structure, too much re-introduces the
#: noise dimensions that make a mixture unidentifiable at these sample sizes.
VARIANCE_TARGET = 0.95

#: Hard floor and ceiling on the BIC search. Not a tuned granularity — the
#: ceiling only stops the search running away on large corpora, and BIC picks
#: the value inside it. The floor is 1 deliberately: k = 1 is the hypothesis
#: "this corpus has no topic structure", and it must be allowed to win, or the
#: method reports groups that are not there. It does win on LoCoMo.
MIN_COMPONENTS = 1
MAX_COMPONENTS = 60

#: Below this many episodes a mixture cannot be estimated at all; the corpus is
#: one group and that is the honest answer.
MIN_EPISODES_FOR_MIXTURE = 30


@dataclass
class Group:
    """One group. `kind` is how it was keyed, not what it contains."""
    id: str
    kind: str                       # "topic" | "entity"
    members: List[str] = field(default_factory=list)
    label: str = ""                 # entity name, or "" for topic groups
    centroid: Optional[np.ndarray] = None
    summary: str = ""               # L2, filled lazily elsewhere
    dirty: bool = True
    new_since_summary: int = 0

    def __len__(self) -> int:
        return len(self.members)


# ==============================================================================
# Projection and mixture selection
# ==============================================================================

def _project(mat: np.ndarray) -> np.ndarray:
    """
    PCA to the number of components retaining VARIANCE_TARGET of variance.

    Implemented directly with an SVD rather than pulled from sklearn so the
    write path keeps its current dependency set.
    """
    n, d = mat.shape
    centred = mat - mat.mean(axis=0, keepdims=True)
    # Economy SVD: singular values squared are proportional to explained variance.
    try:
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return centred

    var = s ** 2
    total = float(var.sum())
    if total <= 0:
        return centred
    keep = int(np.searchsorted(np.cumsum(var) / total, VARIANCE_TARGET) + 1)
    # A mixture needs more samples than parameters; with diagonal covariances a
    # component costs ~2*dim, so cap dimensionality by the sample size.
    keep = max(2, min(keep, d, max(2, n // 10)))
    basis = vt[:keep]
    return centred @ basis.T


def _project_with_basis(mat: np.ndarray):
    """As `_project`, but also returns (mean, basis) so other vectors — e.g.
    LLM-proposed topic labels — can be mapped into the same space."""
    n, d = mat.shape
    mu = mat.mean(axis=0, keepdims=True)
    centred = mat - mu
    try:
        _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return centred, mu, np.eye(d, dtype=mat.dtype)
    var = sv ** 2
    total = float(var.sum())
    if total <= 0:
        return centred, mu, np.eye(d, dtype=mat.dtype)
    keep = int(np.searchsorted(np.cumsum(var) / total, VARIANCE_TARGET) + 1)
    keep = max(2, min(keep, d, max(2, n // 10)))
    basis = vt[:keep]
    return centred @ basis.T, mu, basis


def _fit_diagonal_gmm(X: np.ndarray, k: int, seed: int = 0,
                      iters: int = 100, tol: float = 1e-4,
                      init_means: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, float]:
    """
    EM for a diagonal-covariance Gaussian mixture.

    Returns (responsibilities, log-likelihood). Diagonal covariance because at
    a few hundred episodes a full covariance per component is not estimable,
    and because the projected axes are already decorrelated by the PCA.
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)

    if init_means is not None:
        # Means supplied by the caller (LLM-proposed topic labels). EM then
        # refines them against the actual episodes and yields real posteriors,
        # so assignment uses exactly the same machinery as the unsupervised
        # path and the two are directly comparable.
        means = np.asarray(init_means, dtype=X.dtype).copy()
        k = means.shape[0]
        return _em_loop(X, means, rng, iters, tol)

    # k-means++ style seeding: spread the initial means out, which matters more
    # for stability here than any amount of EM tuning.
    means = np.empty((k, d), dtype=X.dtype)
    means[0] = X[rng.integers(n)]
    closest = ((X - means[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        probs = closest / closest.sum() if closest.sum() > 0 else None
        idx = rng.choice(n, p=probs) if probs is not None else rng.integers(n)
        means[j] = X[idx]
        closest = np.minimum(closest, ((X - means[j]) ** 2).sum(axis=1))

    return _em_loop(X, means, rng, iters, tol)


def _em_loop(X: np.ndarray, means: np.ndarray, rng, iters: int, tol: float
             ) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    k = means.shape[0]
    var = np.tile(X.var(axis=0) + 1e-6, (k, 1))
    weights = np.full(k, 1.0 / k)
    floor = float(X.var(axis=0).mean()) * 1e-3 + 1e-9
    prev_ll = -np.inf

    for _ in range(iters):
        # E step in log space.
        log_det = np.log(var).sum(axis=1)                      # (k,)
        diff = X[:, None, :] - means[None, :, :]               # (n,k,d)
        maha = ((diff ** 2) / var[None, :, :]).sum(axis=2)     # (n,k)
        log_prob = -0.5 * (d * math.log(2 * math.pi) + log_det[None, :] + maha)
        log_prob += np.log(weights)[None, :]

        mx = log_prob.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(log_prob - mx).sum(axis=1))
        ll = float(lse.sum())
        resp = np.exp(log_prob - lse[:, None])

        # M step.
        nk = resp.sum(axis=0) + 1e-10
        weights = nk / n
        means = (resp.T @ X) / nk[:, None]
        for j in range(k):
            d2 = X - means[j]
            var[j] = (resp[:, j][:, None] * d2 ** 2).sum(axis=0) / nk[j]
        np.maximum(var, floor, out=var)

        if ll - prev_ll < tol * abs(ll if ll else 1.0):
            prev_ll = ll
            break
        prev_ll = ll

    return resp, prev_ll


#: Seeds the BIC search averages over. EM is randomised, and a single-seed
#: search picks wildly different counts run to run — one MemGallery scenario
#: gave k = 7, 6 and 1 on three seeds. Averaging removes the dominant source of
#: partition instability. This is an estimator setting like an iteration limit,
#: not a per-corpus constant: the same value is defensible on unseen data.
SELECTION_SEEDS = (0, 1, 2)


def select_components(X: np.ndarray, seed: int = 0
                      ) -> Tuple[int, np.ndarray, Dict[int, float]]:
    n, d = X.shape
    hi = int(min(MAX_COMPONENTS, max(MIN_COMPONENTS, n // 10)))
    best_k, best_mean_bic = MIN_COMPONENTS, math.inf
    scores: Dict[int, float] = {}
    fits: Dict[int, Tuple[np.ndarray, float]] = {}

    for k in range(MIN_COMPONENTS, hi + 1):
        bics: List[float] = []
        best_fit: Optional[Tuple[np.ndarray, float]] = None
        # weights (k-1) + means (k*d) + diagonal variances (k*d)
        params = (k - 1) + 2 * k * d

        for s in (SELECTION_SEEDS if k > 1 else (seed,)):
            try:
                if k == 1:
                    # Closed form: one diagonal Gaussian at the sample moments.
                    var = X.var(axis=0) + 1e-9
                    ll = float((-0.5 * (d * math.log(2 * math.pi)
                                        + np.log(var).sum()
                                        + (((X - X.mean(axis=0)) ** 2)
                                           / var).sum(axis=1))).sum())
                    resp = np.ones((n, 1))
                else:
                    resp, ll = _fit_diagonal_gmm(X, k, seed=s)
            except Exception as e:                 # pragma: no cover
                logger.debug("GMM k=%d seed=%d failed: %s", k, s, e)
                continue
            bics.append(-2.0 * ll + params * math.log(n))
            if best_fit is None or ll > best_fit[1]:
                best_fit = (resp, ll)

        if not bics or best_fit is None:
            continue
        mean_bic = float(np.mean(bics))
        scores[k] = mean_bic
        fits[k] = best_fit
        if mean_bic < best_mean_bic:
            best_k, best_mean_bic = k, mean_bic

    if best_k not in fits:                         # pragma: no cover
        return 1, np.ones((n, 1)), scores
    return best_k, fits[best_k][0], scores

def build_topic_groups(ids: Sequence[str], mat: np.ndarray, seed: int = 0
                       ) -> Tuple[List[Group], Dict[str, Any]]:
    n = len(ids)
    if n < MIN_EPISODES_FOR_MIXTURE:
        g = Group(id="topic-0", kind="topic", members=list(ids),
                  centroid=mat.mean(axis=0) if n else None)
        return [g], {"n_episodes": n, "k": 1, "reason": "too few episodes"}

    X = _project(mat)
    k, resp, bic = select_components(X, seed=seed)

    # Perplexity of each episode's posterior = its own effective number of
    # groups. Peaked -> 1, evenly split between two -> 2, and so on.
    ent = -(resp * np.log(resp + 1e-12)).sum(axis=1)
    perplexity = np.exp(ent)
    n_join = np.clip(np.rint(perplexity).astype(int), 1, k)

    groups = [Group(id=f"topic-{j}", kind="topic") for j in range(k)]
    order = np.argsort(-resp, axis=1)
    for i, eid in enumerate(ids):
        for j in order[i, :n_join[i]]:
            groups[int(j)].members.append(eid)

    groups = [g for g in groups if g.members]
    index = {eid: i for i, eid in enumerate(ids)}
    for g in groups:
        rows = mat[[index[m] for m in g.members]]
        c = rows.mean(axis=0)
        nrm = float(np.linalg.norm(c))
        g.centroid = c / nrm if nrm > 0 else c

    sizes = [len(g) for g in groups]
    diag = {
        "n_episodes": n,
        "k_selected": k,
        "k_searched": [min(bic), max(bic)] if bic else [],
        "projected_dims": int(X.shape[1]),
        "n_groups": len(groups),
        "mean_size": round(float(np.mean(sizes)), 2) if sizes else 0.0,
        "median_size": int(np.median(sizes)) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "memberships_per_episode": round(sum(sizes) / max(1, n), 3),
        "mean_perplexity": round(float(perplexity.mean()), 3),
        "frac_shared": round(float((n_join > 1).mean()), 3),
    }
    return groups, diag


_ATTRIBUTION_RE = re.compile(
    r"^\s*(?:DATE:[^|\n]*\|\s*)?(?:[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)?\s+says:\s*)",
    re.MULTILINE)


def strip_attribution(text: str) -> str:
    """
    Remove the "DATE: … | Speaker says:" scaffolding before entity matching.
    """
    return _ATTRIBUTION_RE.sub("", text or "")


def build_entity_groups(episodes: Sequence[Any], entity_names: Sequence[str],
                        min_members: int = 2,
                        aliases: Optional[Dict[str, str]] = None) -> List[Group]:
    """
    Exact, by mention in the episode's *content*. One group per entity appearing
    in at least `min_members` episodes — not a tuning knob, since a group of one
    has nothing to reconcile and nothing to summarise.

    """
    aliases = aliases or {}
    by_entity: Dict[str, List[str]] = {}
    patterns = [(name, re.compile(r"\b" + re.escape(name) + r"\b"))
                for name in entity_names]
    for ep in episodes:
        text = strip_attribution(ep.raw_text or "")
        seen: set = set()
        for name, pat in patterns:
            if pat.search(text):
                canon = aliases.get(name, name)
                if canon in seen:
                    continue          # one membership per entity, not per form
                seen.add(canon)
                by_entity.setdefault(canon, []).append(ep.id)

    return [Group(id=f"entity-{name}", kind="entity", label=name, members=ids)
            for name, ids in sorted(by_entity.items())
            if len(ids) >= min_members]


ALIAS_PROMPT = """In one conversation, these pairs of names look similar. For \
each, decide whether the two refer to the SAME person or thing, or to \
DIFFERENT ones.

{pairs}

A pair is the same only when one is plainly a nickname, shortening, or spelling \
variant of the other AND nothing suggests they are distinct. Two different \
people with similar names are DIFFERENT. A word that merely resembles a name is \
DIFFERENT.

Reply with JSON only, one entry per pair, in order:
{{"pairs": [{{"n": 1, "same": true, "canonical": "Melanie"}}]}}"""


TOPIC_PROMPT = """Below are excerpts sampled from one long conversation.

{excerpts}

List the distinct recurring subjects this conversation covers — the things a \
person would say it is "about". Aim for the natural number: use as many or as \
few as the material actually supports.

Each subject should be a short noun phrase (2-6 words) that is specific enough \
to distinguish it from the others.

Reply with JSON only:
{{"topics": ["training for a marathon", "her mother's illness"]}}"""


def _json_block(text: str) -> Optional[Dict[str, Any]]:
    """First JSON object in a completion, tolerating fences and preamble."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    return None

    # Unbalanced: the completion hit its token limit mid-structure. Salvage by
    # trimming to the last complete element and closing what is still open —
    # a truncated list of spans is still most of the answer.
    frag = s[start:]
    for cut in range(len(frag) - 1, 0, -1):
        if frag[cut] in "]}," :
            continue
        head = frag[:cut + 1]
        for closing in ("}", "]}", "}]}", "]}]}"):
            try:
                return json.loads(head + closing)
            except json.JSONDecodeError:
                continue
        break
    return None


def resolve_aliases(entity_counts: Dict[str, int], llm: Any,
                    max_names: int = 60) -> Dict[str, str]:
    if not entity_counts or llm is None:
        return {}

    names = sorted(entity_counts.items(), key=lambda kv: -kv[1])[:max_names]
    known = {n for n, _ in names}
    candidates = _alias_candidates([n for n, _ in names])
    if not candidates:
        return {}

    rendered = "\n".join(
        f"{i}. {a} ({entity_counts.get(a, 0)} mentions) / {b} "
        f"({entity_counts.get(b, 0)} mentions)"
        for i, (a, b) in enumerate(candidates, start=1))

    try:
        raw = llm.get_completion(ALIAS_PROMPT.format(pairs=rendered),
                                 temperature=0.0, max_tokens=512, role="grouping")
    except Exception as e:
        logger.warning("alias call failed: %s", e)
        return {}

    parsed = _json_block(raw or "") or {}
    mapping: Dict[str, str] = {}
    for entry in parsed.get("pairs") or []:
        if not isinstance(entry, dict) or not entry.get("same"):
            continue
        try:
            pair = candidates[int(entry.get("n", 0)) - 1]
        except (TypeError, ValueError, IndexError):
            continue
        canon = str(entry.get("canonical", "")).strip()
        # The canonical must be one of the two names actually offered: anything
        # else would create a group matching no episode.
        if canon not in pair:
            canon = max(pair, key=len)
        other = pair[0] if pair[1] == canon else pair[1]
        if other in known and other != canon:
            mapping[other] = canon

    # Collapse chains (a -> b -> c becomes a -> c) so membership is one hop.
    for _ in range(3):
        changed = False
        for k, v in list(mapping.items()):
            if v in mapping and mapping[v] != k:
                mapping[k] = mapping[v]
                changed = True
        if not changed:
            break
    return mapping


def _alias_candidates(names: Sequence[str], max_pairs: int = 40
                      ) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            lo, hi = (a, b) if len(a) <= len(b) else (b, a)
            if len(lo) < 3:
                continue
            same = hi.lower().startswith(lo.lower()) or _edit_within_one(
                lo.lower(), hi.lower())
            if same and lo.lower() != hi.lower():
                out.append((lo, hi))
    return out[:max_pairs]


def _edit_within_one(a: str, b: str) -> bool:
    """True iff Levenshtein(a, b) <= 1, without building a DP matrix."""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return True
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    seen = False
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            if seen:
                return False
            seen = True
            if len(a) == len(b):
                i += 1
            j += 1
            continue
        i += 1
        j += 1
    return True


def propose_topics(episodes: Sequence[Any], llm: Any, n_sample: int = 40,
                   seed: int = 0) -> List[str]:
    if llm is None or not episodes:
        return []

    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(episodes), size=min(n_sample, len(episodes)),
                            replace=False))
    lines = []
    for i in idx:
        text = strip_attribution(episodes[i].raw_text or "").strip()
        lines.append("- " + " ".join(text.split())[:200])

    try:
        raw = llm.get_completion(TOPIC_PROMPT.format(excerpts="\n".join(lines)),
                                 temperature=0.0, max_tokens=512, role="grouping")
    except Exception as e:
        logger.warning("topic call failed: %s", e)
        return []

    parsed = _json_block(raw or "") or {}
    out = []
    for t in parsed.get("topics") or []:
        t = str(t).strip()
        if 0 < len(t) <= 120:
            out.append(t)
    return out


def assign_to_topics(ids: Sequence[str], mat: np.ndarray, topics: Sequence[str],
                     embedder: Any) -> Tuple[List[Group], Dict[str, Any]]:
    if not topics or embedder is None:
        return [], {}

    vecs = embedder.batch_embed(list(topics))
    if any(v is None for v in vecs):
        return [], {}
    tmat = np.asarray(np.vstack(vecs), dtype="float32")
    nrm = np.linalg.norm(tmat, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    tmat = tmat / nrm

    # Map episodes and topic labels into one space, then let EM refine the
    # labels into actual components. Scoring cosine-to-label directly needs a
    # temperature to turn similarities into a posterior, and that temperature
    # silently controls how many groups each episode joins — a knob. Seeding EM
    # instead yields genuine posteriors, so the perplexity rule means the same
    # thing here as in the unsupervised path.
    X, mu, basis = _project_with_basis(mat)
    T = (tmat - mu) @ basis.T
    resp, _ = _fit_diagonal_gmm(X, len(topics), init_means=T)
    post = resp

    ent = -(post * np.log(post + 1e-12)).sum(axis=1)
    n_join = np.clip(np.rint(np.exp(ent)).astype(int), 1, len(topics))
    order = np.argsort(-post, axis=1)

    groups = [Group(id=f"topic-llm-{j}", kind="topic", label=t)
              for j, t in enumerate(topics)]
    for i, eid in enumerate(ids):
        for j in order[i, :n_join[i]]:
            groups[int(j)].members.append(eid)
    groups = [g for g in groups if g.members]

    sizes = [len(g) for g in groups]
    return groups, {
        "n_groups": len(groups),
        "k_selected": len(topics),
        "mean_size": round(float(np.mean(sizes)), 2) if sizes else 0.0,
        "memberships_per_episode": round(sum(sizes) / max(1, len(ids)), 3),
        "frac_shared": round(float((n_join > 1).mean()), 3),
    }


# ==============================================================================
# Index
# ==============================================================================

class GroupIndex:
    """Groups over one memory layer, plus the reverse map."""

    def __init__(self) -> None:
        self.groups: Dict[str, Group] = {}
        self._by_episode: Dict[str, List[str]] = {}
        self.diagnostics: Dict[str, Any] = {}

    @classmethod
    def build(cls, layer: Any, seed: int = 0, llm: Any = None,
              embedder: Any = None) -> "GroupIndex":
        self = cls()
        episodes = [e for e in layer.all_episodes()
                    if getattr(e, "embedding", None) is not None]
        if not episodes:
            return self

        ids = [e.id for e in episodes]
        mat = np.vstack([np.asarray(e.embedding, dtype="float32") for e in episodes])
        nrm = np.linalg.norm(mat, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        mat = mat / nrm

        names = layer.entity_stats.top_entities(n=200, min_count=2)

        topics, diag = [], {}
        if llm is not None:
            embedder = embedder or getattr(layer, "_embedder", None)
            labels = propose_topics(episodes, llm)
            if labels and embedder is not None:
                topics, diag = assign_to_topics(ids, mat, labels, embedder)
                diag["source"] = "llm"
        if not topics:
            topics, diag = build_topic_groups(ids, mat, seed=seed)
            diag["source"] = "mixture"

        aliases = {}
        if llm is not None and names:
            counts = {}
            for ep in episodes:
                text = strip_attribution(ep.raw_text or "")
                for nm in names:
                    if re.search(r"\b" + re.escape(nm) + r"\b", text):
                        counts[nm] = counts.get(nm, 0) + 1
            aliases = resolve_aliases(counts, llm)
        entities = build_entity_groups(episodes, names, aliases=aliases)
        self.aliases = aliases

        for g in topics + entities:
            self.groups[g.id] = g
            for m in g.members:
                self._by_episode.setdefault(m, []).append(g.id)

        diag["n_entity_groups"] = len(entities)
        diag["mean_entity_size"] = (
            round(sum(len(g) for g in entities) / len(entities), 2) if entities else 0.0)
        self.diagnostics = diag
        return self

    def groups_for(self, episode_id: str) -> List[Group]:
        return [self.groups[g] for g in self._by_episode.get(episode_id, [])]

    def of_kind(self, kind: str) -> List[Group]:
        return [g for g in self.groups.values() if g.kind == kind]

    def mark_dirty(self, episode_id: str) -> None:
        for g in self.groups_for(episode_id):
            g.dirty = True
            g.new_since_summary += 1

def cohesion_report(groups: Sequence[Group], ids: Sequence[str],
                    mat: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """
    Mean within-group similarity against a same-size random baseline.

    """
    index = {e: i for i, e in enumerate(ids)}
    rng = np.random.default_rng(seed)
    coh: List[float] = []
    rnd: List[float] = []

    for g in groups:
        rows = [index[m] for m in g.members if m in index]
        if len(rows) < 2:
            continue
        # Sample at random, not the first 50. Members arrive in corpus order, so
        # taking a prefix of a large group scores temporally adjacent episodes,
        # which are similar for reasons that have nothing to do with the group.
        if len(rows) > 50:
            rows = list(rng.choice(rows, size=50, replace=False))
        sub = mat[rows]
        sim = sub @ sub.T
        m = len(sub)
        coh.append(float((sim.sum() - np.trace(sim)) / (m * m - m)))

        pick = rng.choice(len(ids), size=min(50, len(rows)), replace=False)
        sub2 = mat[pick]
        sim2 = sub2 @ sub2.T
        m2 = len(sub2)
        if m2 > 1:
            rnd.append(float((sim2.sum() - np.trace(sim2)) / (m2 * m2 - m2)))

    return {
        "cohesion": round(float(np.mean(coh)), 4) if coh else None,
        "random": round(float(np.mean(rnd)), 4) if rnd else None,
        "gap": round(float(np.mean(coh) - np.mean(rnd)), 4) if coh and rnd else None,
        "n_scored": len(coh),
    }
