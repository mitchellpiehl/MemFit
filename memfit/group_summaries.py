#!/usr/bin/env python3
"""
L2 group summaries, and the only question that decides whether they belong:
does putting them in the retrieval pool make answers better?

Grouping is now measured intrinsically (co-evidence lift) and `segment` wins.
That is necessary, not sufficient — a grouping is only worth having if something
downstream improves. This builds one summary per group and files it in the
existing NoteStore, so the summaries compete for pack slots against raw episodes
under the same cross-encoder, with no new retrieval path and no query-time cost.

Reusing NoteStore is deliberate: it is already indexed, already competes in the
pool, already carries provenance, and is already exercised by the harness's
`--notes` flag. A parallel store would be new untested machinery for no gain.

    python group_summaries.py --benchmark memgallery --method segment
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from consolidation import Note
from grouping import strip_attribution
from grouping_methods import METHODS

logger = logging.getLogger("group_summaries")

SUMMARY_PROMPT = """Below are consecutive excerpts from one conversation.

{excerpts}

Write two sentences stating what this stretch of the conversation is about and \
what is currently true of it. Name the people, places and things explicitly — \
this will be read on its own, without the excerpts.

SUMMARY:"""

#: A group smaller than this has nothing a summary would add over its own turns.
MIN_GROUP = 3
#: Excerpts shown per summary. Bounded so cost does not scale with group size.
MAX_EXCERPTS = 14


def summarise_groups(layer: Any, llm: Any, method: str = "segment",
                     max_groups: int = 60) -> Dict[str, Any]:
    """
    Build groups with `method`, summarise each, and add them to layer.notes.

    Cost is one LLM call per group, entirely off the query path — the same
    budget class as session consolidation, which is ~1.9 calls per session.
    """
    from llm_controller import LEDGER

    n_ledger0 = len(LEDGER.records)
    t_start = time.time()
    eps = [e for e in layer.all_episodes()
           if getattr(e, "embedding", None) is not None]
    eps.sort(key=lambda e: e.global_seq)
    if len(eps) < MIN_GROUP:
        return {"groups": 0, "summaries": 0}

    ids = [e.id for e in eps]
    mat = np.vstack([np.asarray(e.embedding, dtype="float32") for e in eps])
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)

    fn, _ = METHODS[method]
    t_grp = time.time()
    groups = fn(ids, mat, episodes=eps, layer=layer, llm=llm,
                embedder=getattr(layer, "_embedder", None), target_k=8)
    group_s = time.time() - t_grp
    n_segments = len(groups)
    groups = [g for g in groups if len(g) >= MIN_GROUP]
    n_eligible = len(groups)
    groups.sort(key=len, reverse=True)
    groups = groups[:max_groups]

    by_id = {e.id: e for e in eps}
    embedder = getattr(layer, "_embedder", None)
    # Load the encoder before the clock starts: a process's first embed pays
    # for loading the model, which is start-up cost, not summarising. (It was
    # showing up as 19.9 s of "embedding" for 29 two-sentence summaries.)
    encoder_load_s = 0.0
    if embedder is not None and getattr(embedder, "_model", None) is None:
        t_l = time.time()
        try:
            embedder._load()
        except Exception:
            pass
        encoder_load_s = time.time() - t_l
    made = 0
    embed_s = 0.0
    t0 = time.time()

    for gi, g in enumerate(groups):
        members = [by_id[m] for m in g.members if m in by_id]
        if len(members) < MIN_GROUP:
            continue
        members.sort(key=lambda e: e.global_seq)
        # Evenly spaced excerpts, so a long group is represented across its
        # whole span rather than by its opening.
        step = max(1, len(members) // MAX_EXCERPTS)
        shown = members[::step][:MAX_EXCERPTS]
        text = "\n".join(
            "- " + " ".join(strip_attribution(m.raw_text or "").split())[:220]
            for m in shown)

        try:
            raw = llm.get_completion(SUMMARY_PROMPT.format(excerpts=text),
                                     temperature=0.0, max_tokens=200,
                                     role="summarise")
        except Exception as e:
            logger.warning("summary call failed: %s", e)
            continue
        summary = " ".join((raw or "").split())
        if len(summary) < 20:
            continue

        stamps = [m.metadata.get("session_datetime") or m.timestamp
                  for m in members if (m.metadata.get("session_datetime") or m.timestamp)]
        note = Note(
            id=f"gs-{uuid.uuid4().hex[:10]}",
            subject=f"group {gi}",
            attribute="summary",
            statement=summary,
            event_time=stamps[0] if stamps else None,
            recorded_time=stamps[-1] if stamps else None,
            provenance=[m.id for m in members],
            # Evidence ids are deliberately NOT inherited: a summary that
            # inherited its members' gold ids would inflate retrieval metrics
            # without retrieving anything. This bit me once already.
            evidence_ids=[],
            session=None,
        )
        if embedder is not None:
            t_e = time.time()
            try:
                note.embedding = embedder.embed(summary)
            except Exception:
                pass
            embed_s += time.time() - t_e
        layer.notes.add(note)
        made += 1

    recs = LEDGER.records[n_ledger0:]
    loop_s = time.time() - t0
    return {"method": method, "segments": n_segments, "groups": len(groups),
            "summaries": made,
            # segments too small to summarise (deliberate), and segments large
            # enough but dropped by max_groups (a cap that binds only at long
            # scale: zero on LoCoMo and MemGallery, most of them on BEAM-10M)
            "small_segments": n_segments - n_eligible,
            "capped_segments": max(0, n_eligible - max_groups),
            "seconds": round(loop_s, 1),
            "group_s": round(group_s, 4),
            "summarise_s": round(loop_s - embed_s, 4),
            "embed_s": round(embed_s, 4),
            "encoder_load_s": round(encoder_load_s, 4),
            "wall_s": round(time.time() - t_start, 4),
            "llm": {"calls": len(recs),
                    "prompt_tokens": int(sum(r.prompt_tokens for r in recs)),
                    "completion_tokens": int(sum(r.completion_tokens for r in recs)),
                    "llm_latency_s": round(float(sum(r.latency_s for r in recs)), 4),
                    "failures": int(sum(0 if r.ok else 1 for r in recs))},
            "mean_group_size": round(float(np.mean([len(g) for g in groups])), 1)
            if groups else 0.0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="memgallery",
                   choices=["memgallery", "locomo"])
    p.add_argument("--method", default="segment", choices=sorted(METHODS))
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--ollama_host", default="http://localhost:11434")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cache_dir", default=None,
                   help="layers to summarise; an alternative embedder keeps its "
                        "own cache, and its summaries must be built there")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--out", default=None,
                   help="where to write the per-conversation cost record "
                        "(default: results/group_summaries_<bench>_<method>[_<cache>].json)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    for n in ("sentence_transformers", "transformers", "httpx", "urllib3"):
        logging.getLogger(n).setLevel(logging.ERROR)

    from llm_controller import LLMController, LEDGER
    from memory_layer import RecursiveMemoryLayer

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
    llm = LLMController(backend="ollama", model=args.model,
                        ollama_host=args.ollama_host).llm
    default_cache = "cache_memgallery" if args.benchmark == "memgallery" else "cache_locomo"
    cache = os.path.join(here, args.cache_dir or default_cache)
    if not os.path.isdir(cache):
        raise SystemExit(f"cache dir {cache} does not exist — build the base layers "
                         f"first by running the evaluation once with this cache_dir")
    prefix = "mg_" if args.benchmark == "memgallery" else "locomo_sample_"
    files = [f for f in sorted(os.listdir(cache))
             if f.startswith(prefix) and f.endswith(".pkl")
             and "_notes" not in f and "_gs" not in f]
    if args.limit:
        files = files[:args.limit]
    if args.num_shards > 1:
        files = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
        logger.info("shard %d/%d: %d conversations", args.shard,
                    args.num_shards, len(files))

    LEDGER.reset()
    info = []
    for f in files:
        layer = RecursiveMemoryLayer.load(os.path.join(cache, f), llm)
        d = summarise_groups(layer, llm, method=args.method)
        # A layer saved with zero summaries is indistinguishable from the base
        # layer, so the evaluation would silently measure base twice and report
        # it as the summaries arm. Refuse to write it. (Seen for real: a wedged
        # backend returned empty for all 11 calls and this wrote a no-op layer.)
        if d.get("summaries", 0) == 0:
            logger.error("%s produced 0 summaries from %d groups — NOT saving. "
                         "The backend is probably returning empty responses.",
                         f, d.get("groups", 0))
            d["failed"] = True
            info.append(d)
            continue
        out = os.path.join(cache, f.replace(".pkl", f"_gs-{args.method}.pkl"))
        layer.save(out)
        d["dataset"] = f[:-4]
        info.append(d)
        logger.info("%-44s %d groups -> %d summaries (%.0fs)",
                    f[:-4][:44], d.get("groups", 0), d.get("summaries", 0),
                    d.get("seconds", 0))

    tot = LEDGER.totals()
    wall_all = sum(d.get("wall_s", 0.0) for d in info)
    n_failed = sum(1 for d in info if d.get("failed"))
    print(f"\n{sum(d.get('summaries', 0) for d in info)} summaries over "
          f"{len(info)} conversations ({n_failed} produced none)")
    print(f"LLM calls {tot['llm_calls']} "
          f"({tot['llm_calls']/max(1,len(info)):.1f} per conversation), "
          f"{tot['total_tokens']} tokens — all off the query path")
    tag = "" if args.num_shards <= 1 else f".s{args.shard}"
    ctag = "" if not args.cache_dir or args.cache_dir == default_cache \
        else "_" + os.path.basename(os.path.normpath(args.cache_dir))
    out = args.out or os.path.join(
        here, "results", f"group_summaries_{args.benchmark}_{args.method}{ctag}{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"benchmark": args.benchmark, "method": args.method,
               "model": args.model, "cache_dir": cache,
               "ollama_host": args.ollama_host, "shard": args.shard,
               "num_shards": args.num_shards, "wall_s": round(wall_all, 3),
               "llm_totals": tot, "per_conversation": info},
              open(out, "w"), indent=1)
    print(f"wrote {out}")

    if n_failed:
        print(f"FAILED: {n_failed} of {len(info)} conversations produced no "
              f"summaries. Not exiting 0 — a downstream run would otherwise "
              f"measure the base layer and call it the summaries arm.")
        sys.exit(1)


if __name__ == "__main__":
    main()
