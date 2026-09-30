#!/usr/bin/env python3

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grouping import (Group, _project, _project_with_basis, _fit_diagonal_gmm,
                      select_components, build_topic_groups, build_entity_groups,
                      resolve_aliases, propose_topics, strip_attribution,
                      _json_block)

logger = logging.getLogger("grouping_methods")


def _perplexity_assign(post: np.ndarray, ids: Sequence[str], prefix: str,
                       labels: Optional[Sequence[str]] = None) -> List[Group]:
    """Shared soft assignment: each item joins exp(H) of its own posterior."""
    k = post.shape[1]
    ent = -(post * np.log(post + 1e-12)).sum(axis=1)
    n_join = np.clip(np.rint(np.exp(ent)).astype(int), 1, k)
    order = np.argsort(-post, axis=1)
    groups = [Group(id=f"{prefix}-{j}", kind="topic",
                    label=(labels[j] if labels and j < len(labels) else ""))
              for j in range(k)]
    for i, eid in enumerate(ids):
        for j in order[i, :n_join[i]]:
            groups[int(j)].members.append(eid)
    return [g for g in groups if g.members]


def _hard_groups(labels_arr: np.ndarray, ids: Sequence[str], prefix: str
                 ) -> List[Group]:
    """Groups from a hard label vector; -1 means noise and joins nothing."""
    out: Dict[int, Group] = {}
    for i, lab in enumerate(labels_arr):
        lab = int(lab)
        if lab < 0:
            continue
        out.setdefault(lab, Group(id=f"{prefix}-{lab}", kind="topic"))
        out[lab].members.append(ids[i])
    return list(out.values())


# ==============================================================================
# Controls
# ==============================================================================

def group_random(ids, mat, target_k: int = 8, seed: int = 0, **kw) -> List[Group]:
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, max(1, target_k), size=len(ids))
    return _hard_groups(lab, ids, "random")


def group_window(ids, mat, target_k: int = 8, **kw) -> List[Group]:
    """
    Consecutive turns in equal blocks — the locality baseline.

    A conversation stays on a subject for a while, so contiguity alone is
    informative. Any clustering that cannot beat this is not earning its cost.
    """
    k = max(1, target_k)
    size = max(1, math.ceil(len(ids) / k))
    groups = []
    for j, start in enumerate(range(0, len(ids), size)):
        groups.append(Group(id=f"window-{j}", kind="topic",
                            members=list(ids[start:start + size])))
    return groups


# ==============================================================================
# Geometric
# ==============================================================================

def group_mixture(ids, mat, seed: int = 0, **kw) -> List[Group]:
    groups, _ = build_topic_groups(ids, mat, seed=seed)
    return groups


def group_agglomerative(ids, mat, **kw) -> List[Group]:
    from scipy.cluster.hierarchy import linkage, fcluster
    X = _project(mat)
    n = len(ids)
    if n < 4:
        return [Group(id="agglom-0", kind="topic", members=list(ids))]
    Z = linkage(X, method="ward")
    heights = Z[:, 2]
    # consider only the top merges; a gap deep in the tree is noise
    top = heights[-min(40, len(heights)):]
    gaps = np.diff(top)
    if len(gaps) == 0:
        return [Group(id="agglom-0", kind="topic", members=list(ids))]
    cut_at = int(np.argmax(gaps))
    threshold = (top[cut_at] + top[cut_at + 1]) / 2.0
    lab = fcluster(Z, t=threshold, criterion="distance")
    return _hard_groups(lab - 1, ids, "agglom")


def group_hdbscan(ids, mat, **kw) -> List[Group]:
    from sklearn.cluster import HDBSCAN
    X = _project(mat)
    n = len(ids)
    mcs = max(5, n // 50)          # derived from corpus size, not tuned
    lab = HDBSCAN(min_cluster_size=mcs).fit_predict(X)
    return _hard_groups(lab, ids, "hdbscan")


def group_spectral(ids, mat, seed: int = 0, target_k: int = 8, **kw) -> List[Group]:
    from sklearn.cluster import SpectralClustering
    n = len(ids)
    k = max(2, min(target_k, n // 10))
    aff = (mat @ mat.T + 1.0) / 2.0
    np.fill_diagonal(aff, 1.0)
    lab = SpectralClustering(n_clusters=k, affinity="precomputed",
                             random_state=seed,
                             assign_labels="kmeans").fit_predict(aff)
    return _hard_groups(lab, ids, "spectral")


def group_louvain(ids, mat, seed: int = 0, **kw) -> List[Group]:
    import networkx as nx
    n = len(ids)
    k = max(2, int(math.log2(max(2, n))))
    sims = mat @ mat.T
    np.fill_diagonal(sims, -np.inf)
    nbrs = np.argsort(-sims, axis=1)[:, :k]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in nbrs[i]:
            j = int(j)
            # mutual kNN: keep the edge only if the relation is reciprocated,
            # which prunes hub edges that merge unrelated regions
            if i in nbrs[j]:
                w = float(sims[i, j])
                if w > 0:
                    G.add_edge(i, j, weight=w)

    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    groups = []
    for j, c in enumerate(comms):
        members = [ids[i] for i in sorted(c)]
        if members:
            groups.append(Group(id=f"louvain-{j}", kind="topic", members=members))
    return groups

def group_segment(ids, mat, **kw) -> List[Group]:
    """
    TextTiling: score each gap by how much the conversation shifts across it,
    then cut at significant valleys.
    """
    n = len(ids)
    w = max(3, int(round(math.sqrt(n) / 2)))
    if n < 4 * w:
        return [Group(id="segment-0", kind="topic", members=list(ids))]

    # block similarity across every internal gap
    gaps = []
    for i in range(w, n - w + 1):
        left = mat[i - w:i].mean(axis=0)
        right = mat[i:i + w].mean(axis=0)
        ln, rn = np.linalg.norm(left), np.linalg.norm(right)
        gaps.append(float(left @ right / (ln * rn)) if ln and rn else 0.0)
    gaps = np.asarray(gaps)

    # depth score: how far this valley sits below the peaks either side
    depth = np.zeros_like(gaps)
    for i in range(len(gaps)):
        l = i
        while l > 0 and gaps[l - 1] >= gaps[l]:
            l -= 1
        r = i
        while r < len(gaps) - 1 and gaps[r + 1] >= gaps[r]:
            r += 1
        depth[i] = (gaps[l] - gaps[i]) + (gaps[r] - gaps[i])

    # TextTiling's published cutoff
    cutoff = depth.mean() + depth.std() / 2.0
    cuts = [i + w for i in range(len(depth))
            if depth[i] > cutoff and
            (i == 0 or depth[i] >= depth[i - 1]) and
            (i == len(depth) - 1 or depth[i] >= depth[i + 1])]

    bounds = [0] + sorted(set(cuts)) + [n]
    groups = []
    for j in range(len(bounds) - 1):
        members = list(ids[bounds[j]:bounds[j + 1]])
        if members:
            groups.append(Group(id=f"segment-{j}", kind="topic", members=members))
    return groups

def group_llm_topics(ids, mat, episodes=None, llm=None, embedder=None,
                     n_sample: int = 40, seed: int = 0, **kw) -> List[Group]:
    labels = propose_topics(episodes, llm, n_sample=n_sample, seed=seed)
    return _topics_to_groups(ids, mat, labels, embedder, "llmtopic")


def group_llm_topics_x3(ids, mat, episodes=None, llm=None, embedder=None,
                        **kw) -> List[Group]:
    pooled: List[str] = []
    for s in (0, 1, 2):
        pooled.extend(propose_topics(episodes, llm, n_sample=40, seed=s))
    if not pooled or embedder is None:
        return []

    vecs = embedder.batch_embed(pooled)
    keep, kept_vecs = [], []
    for label, v in zip(pooled, vecs):
        if v is None:
            continue
        v = np.asarray(v, dtype="float32")
        v = v / (np.linalg.norm(v) or 1.0)
        if any(float(v @ kv) > 0.85 for kv in kept_vecs):
            continue          # near-duplicate of a label already kept
        keep.append(label)
        kept_vecs.append(v)
    return _topics_to_groups(ids, mat, keep, embedder, "llmtopic3")


OUTLINE_PROMPT = """Below are numbered excerpts from one conversation, in order.

{excerpts}

The conversation moves between subjects. Split it into consecutive spans, each \
covering one subject, and give each span a short label. Merge neighbouring \
excerpts that share a subject — far fewer spans than excerpts.

Reply with JSON only, spans in order, each [start_number, "label"]:
{{"spans": [[1, "planning the trip"], [14, "her new job"]]}}"""


def group_llm_outline(ids, mat, episodes=None, llm=None, embedder=None,
                      **kw) -> List[Group]:
    """
    Ask the model where the conversation changes subject, then cut there.

    The LLM equivalent of `segment`: it combines the sequential structure that
    clustering discards with the semantic judgement that geometry lacks. Only a
    sample is shown, and the boundaries it returns are mapped back onto the full
    stream, so the call stays bounded regardless of conversation length.
    """
    if llm is None or not episodes:
        return []
    n = len(ids)
    step = max(1, n // 40)
    idx = list(range(0, n, step))[:40]
    lines = []
    for num, i in enumerate(idx, start=1):
        text = strip_attribution(episodes[i].raw_text or "").strip()
        lines.append(f"{num}. " + " ".join(text.split())[:160])

    try:
        raw = llm.get_completion(OUTLINE_PROMPT.format(excerpts="\n".join(lines)),
                                 temperature=0.0, max_tokens=1200, role="grouping")
    except Exception as e:
        logger.warning("outline call failed: %s", e)
        return []

    parsed = _json_block(raw or "") or {}
    spans = []
    for sp in parsed.get("spans") or []:
        # accept both [n, "label"] and {"start": n, "label": ...}
        if isinstance(sp, (list, tuple)) and len(sp) >= 2:
            start_raw, label = sp[0], sp[1]
        elif isinstance(sp, dict):
            start_raw, label = sp.get("start", 0), sp.get("label", "")
        else:
            continue
        try:
            start = int(start_raw)
        except (TypeError, ValueError):
            continue
        if 1 <= start <= len(idx):
            spans.append((idx[start - 1], str(label).strip()))
    if not spans:
        return []

    spans.sort()
    if spans[0][0] != 0:
        spans.insert(0, (0, spans[0][1]))
    groups = []
    for j, (start, label) in enumerate(spans):
        end = spans[j + 1][0] if j + 1 < len(spans) else n
        members = list(ids[start:end])
        if members:
            groups.append(Group(id=f"outline-{j}", kind="topic",
                                members=members, label=label))
    return groups


def _topics_to_groups(ids, mat, labels, embedder, prefix) -> List[Group]:
    """Embed labels, seed EM with them, assign by perplexity."""
    if not labels or embedder is None:
        return []
    vecs = embedder.batch_embed(list(labels))
    rows = [(l, np.asarray(v, dtype="float32"))
            for l, v in zip(labels, vecs) if v is not None]
    if not rows:
        return []
    labels = [l for l, _ in rows]
    tmat = np.vstack([v for _, v in rows])
    tmat = tmat / np.maximum(np.linalg.norm(tmat, axis=1, keepdims=True), 1e-9)

    X, mu, basis = _project_with_basis(mat)
    T = (tmat - mu) @ basis.T
    resp, _ = _fit_diagonal_gmm(X, len(labels), init_means=T)
    return _perplexity_assign(resp, ids, prefix, labels)


def group_entity(ids, mat, episodes=None, layer=None, llm=None,
                 use_aliases: bool = True, **kw) -> List[Group]:
    names = layer.entity_stats.top_entities(n=200, min_count=2) if layer else []
    aliases = {}
    if use_aliases and llm is not None and names:
        counts: Dict[str, int] = {}
        pats = [(nm, re.compile(r"\b" + re.escape(nm) + r"\b")) for nm in names]
        for ep in episodes:
            t = strip_attribution(ep.raw_text or "")
            for nm, p in pats:
                if p.search(t):
                    counts[nm] = counts.get(nm, 0) + 1
        aliases = resolve_aliases(counts, llm)
    return build_entity_groups(episodes, names, aliases=aliases)


SUMMARY_PROMPT = """Summarise what this part of a conversation is about, in two \
sentences. State the subject and what is currently true of it.

{excerpts}

SUMMARY:"""


def group_hierarchical(ids, mat, episodes=None, llm=None, embedder=None,
                       seed: int = 0, max_summaries: int = 12, **kw
                       ) -> List[Group]:
    """
    RAPTOR's move: cluster, summarise each cluster, cluster the summaries.

    The second level is what answers questions no single episode covers. It
    costs one call per first-level group, which is why the count is capped —
    the point is to test whether a second level adds anything, not to pay for
    a deep tree.
    """
    base = group_mixture(ids, mat, seed=seed)
    if len(base) < 3 or llm is None or embedder is None:
        return base

    by_id = {e.id: e for e in (episodes or [])}
    summaries, keep = [], []
    for g in sorted(base, key=len, reverse=True)[:max_summaries]:
        texts = []
        for m in g.members[:12]:
            ep = by_id.get(m)
            if ep is not None:
                texts.append(" ".join(strip_attribution(ep.raw_text or "").split())[:200])
        if not texts:
            continue
        try:
            s = llm.get_completion(
                SUMMARY_PROMPT.format(excerpts="\n".join("- " + t for t in texts)),
                temperature=0.0, max_tokens=160, role="grouping")
        except Exception as e:
            logger.warning("summary call failed: %s", e)
            continue
        if s and s.strip():
            summaries.append(" ".join(s.split()))
            keep.append(g)

    if len(summaries) < 3:
        return base

    vecs = embedder.batch_embed(summaries)
    rows = [(i, np.asarray(v, "float32")) for i, v in enumerate(vecs) if v is not None]
    if len(rows) < 3:
        return base
    smat = np.vstack([v for _, v in rows])
    smat = smat / np.maximum(np.linalg.norm(smat, axis=1, keepdims=True), 1e-9)

    # cluster the summaries; each level-2 group is the union of its children
    sk, sresp, _ = select_components(_project(smat))
    parent = _perplexity_assign(sresp, [str(i) for i, _ in rows], "L2")
    out = list(base)
    for p in parent:
        members: List[str] = []
        for child_idx in p.members:
            g = keep[int(child_idx)]
            members.extend(g.members)
        if members:
            out.append(Group(id=f"hier-{p.id}", kind="topic",
                             members=sorted(set(members)),
                             label="; ".join(summaries[int(i)] for i in p.members[:2])[:160]))
    return out


# ==============================================================================
# Registry
# ==============================================================================

METHODS = {
    "random": (group_random, False),
    "window": (group_window, False),
    "mixture": (group_mixture, False),
    "agglom": (group_agglomerative, False),
    "hdbscan": (group_hdbscan, False),
    "spectral": (group_spectral, False),
    "louvain": (group_louvain, False),
    "segment": (group_segment, False),
    "llm_topics": (group_llm_topics, True),
    "llm_topics_x3": (group_llm_topics_x3, True),
    "llm_outline": (group_llm_outline, True),
    "entity": (group_entity, True),
    "hierarchical": (group_hierarchical, True),
}
