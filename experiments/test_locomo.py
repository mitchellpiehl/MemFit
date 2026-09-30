#!/usr/bin/env python3
"""
test_locomo.py — LoCoMo evaluation harness.

Category 5 (adversarial) is excluded, consistent with prior work.
"""

from __future__ import annotations

# Repository root (holds data/, cache_*/ and results/); the system code is in memfit/.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "memfit"))

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from baselines import BaselineRetriever
from consolidation import Consolidator
import costmeter
from llm_controller import LEDGER, LLMController
from load_dataset import load_locomo_dataset
from memory_layer import MemoryConfig, RecursiveMemoryLayer
from metrics import aggregate, answer_metrics, category_name, retrieval_metrics
from rlm_controller import RLMConfig, RLMController

logger = logging.getLogger("locomo")

ALLOWED_CATEGORIES = (1, 2, 3, 4)   # 5 = adversarial, excluded as in prior work
RECALL_KS = (5, 10, 20, 30)

def build_turn_text(session_datetime: Optional[str], speaker: str, text: str) -> str:
    if session_datetime:
        return f"DATE: {session_datetime} | Speaker {speaker} says: {text}"
    return f"Speaker {speaker} says: {text}"


def _iso_date(session_datetime: Optional[str]) -> Optional[str]:
    """
    LoCoMo session stamps look like '1:56 pm on 8 May, 2023'.
    Normalise to ISO so date arithmetic in the reader works.
    """
    if not session_datetime:
        return None
    m = None
    import re

    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", session_datetime)
    if not m:
        return session_datetime
    day, month_name, year = m.group(1), m.group(2), m.group(3)
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(f"{day} {month_name} {year}", fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return session_datetime


def build_memory(sample, mem_config: MemoryConfig, llm) -> RecursiveMemoryLayer:
    """One MemoryObject per dialogue turn, tagged with its dia_id."""
    layer = RecursiveMemoryLayer(mem_config, llm)
    items: List[Dict[str, Any]] = []
    for _, session in sample.conversation.sessions.items():
        iso = _iso_date(session.date_time)
        for turn in session.turns:
            items.append({
                "raw_text": build_turn_text(session.date_time, turn.speaker, turn.text),
                "metadata": {"speaker": turn.speaker, "dia_id": turn.dia_id,
                             "session_datetime": session.date_time,
                             "session_id": str(getattr(session, "session_id", ""))},
                "timestamp": iso,
                "source": "locomo",
                "evidence_ids": [turn.dia_id],
            })
    layer.add_memories_bulk(items)
    return layer


def consolidate(layer: RecursiveMemoryLayer, llm, max_calls_per_session: int = 2,
                use_llm_reconcile: bool = True,
                group_scoped: bool = True) -> Dict[str, Any]:
    """
    Phase 3: distil each session into notes, once, off the write path.
    """
    consolidator = Consolidator(
        store=layer.notes, llm=llm, embedder=getattr(layer, "_embedder", None),
        max_calls_per_session=max_calls_per_session,
        use_llm_reconcile=use_llm_reconcile,
    )
    by_session: Dict[str, List[Any]] = defaultdict(list)
    for m in sorted(layer.all_episodes(), key=lambda x: x.global_seq):
        by_session[m.metadata.get("session_datetime") or "unknown"].append(m)

    t0 = time.time()
    sessions = sorted(by_session.items())
    for i, (session_date, episodes) in enumerate(sessions):
        subjects = set()
        for m in episodes:
            subjects.update(m.episode_speakers or [])
        consolidator.consolidate_session(
            episodes, session_date,
            session_id=episodes[0].metadata.get("session_id") if episodes else None,
            subjects=subjects,
        )
        if group_scoped:
            consolidator.attach_groups(
                layer.entity_stats.top_entities(n=200, min_count=2))

    out = {"sessions": len(by_session), "seconds": time.time() - t0,
           "ops": dict(consolidator.ops), **layer.notes.stats()}
    if group_scoped:
        out["entity_groups"] = len(getattr(consolidator, "_note_groups", {}) or {})
    return out

def _ledger_since(n0: int) -> Dict[str, Any]:
    """LLM usage recorded after record index n0, split by role."""
    recs = LEDGER.records[n0:]
    by_role: Dict[str, Dict[str, float]] = {}
    for r in recs:
        d = by_role.setdefault(r.role, {"calls": 0, "prompt_tokens": 0,
                                        "completion_tokens": 0, "llm_latency_s": 0.0})
        d["calls"] += 1
        d["prompt_tokens"] += r.prompt_tokens
        d["completion_tokens"] += r.completion_tokens
        d["llm_latency_s"] += r.latency_s
    return {"calls": len(recs), "by_role": by_role}


def evaluate(
    dataset_path: str,
    model: str,
    backend: str,
    system: str,
    output_path: Optional[str],
    num_samples: Optional[int],
    max_questions: Optional[int],
    top_k: int,
    ollama_host: str,
    api_key: Optional[str],
    reader_prompt: str,
    reader_question: str,
    pool_strategy: str,
    enable_planner: bool,
    cache_dir: Optional[str],
    run_name: str,
    question_seed: int = 0,
    num_ctx: Optional[int] = None,
    lexical_scorer: str = "bm25",
    fusion: str = "convex",
    alpha: float = 0.7,
    scan_top_k: int = 30,
    rerank: bool = True,
    consolidate_memory: bool = False,
    consolidate_calls: int = 2,
    no_llm_reconcile: bool = False,
    planner_style: str = "conservative",
    use_notes: bool = True,
    note_mode: str = "compete",
    group_scoped: bool = True,
    summaries: Optional[str] = None,
    rerank_normalise: bool = True,
    expand_summaries: int = 0,
    prf_docs: int = 0,
    shard: int = 0,
    num_shards: int = 1,
    embedder: str = "all-MiniLM-L6-v2",
    reranker: Optional[str] = None,
    rerank_pool: int = 40,
    context_window: int = 0,
    build_only: bool = False,
    measure_cost: bool = False,
) -> Dict[str, Any]:
    if measure_cost:
        costmeter.install()
    resolved_ctx = num_ctx if num_ctx else (65536 if system == "fullcontext" else None)
    llm_controller = LLMController(
        backend=backend, model=model, api_key=api_key, ollama_host=ollama_host,
        num_ctx=resolved_ctx,
    )
    llm = llm_controller.llm

    from encoders import configure_reranker, startup_check, final_check
    configure_reranker(reranker)
    encoder_info: Dict[str, Any] = {}
    mem_config = MemoryConfig(enable_multimodal=False, enable_embeddings=True,
                              embedding_model=embedder)
    rlm_config = RLMConfig(
        top_k_episodes=top_k,
        reader_prompt=reader_prompt,
        reader_question=reader_question,
        pool_strategy=pool_strategy,
        enable_planner=enable_planner,
        lexical_scorer=lexical_scorer,
        fusion=fusion,
        alpha=alpha,
        scan_top_k=scan_top_k,
        rerank=rerank,
        planner_style=planner_style,
        use_notes=use_notes,
        note_mode=note_mode,
        rerank_normalise=rerank_normalise,
        expand_summaries=expand_summaries,
        prf_docs=prf_docs,
        rerank_pool=rerank_pool,
        context_window=context_window,
    )

    expected_calls = 2 if (system == "memfit" and enable_planner) else 1

    samples = load_locomo_dataset(dataset_path)
    if num_samples:
        samples = samples[:num_samples]
    indexed = list(enumerate(samples))
    if num_shards > 1:
        indexed = [(i, x) for i, x in indexed if i % num_shards == shard]
        logger.info("shard %d/%d: %d samples (original indices %s)",
                    shard, num_shards, len(indexed), [i for i, _ in indexed])

    LEDGER.reset()
    records: List[Dict[str, Any]] = []
    per_question: List[Dict[str, Any]] = []
    build_times: List[float] = []
    consolidation_info: List[Dict[str, Any]] = []
    budget_violations = 0
    build_report: List[Dict[str, Any]] = []
    question_costs: List[Dict[str, Any]] = []

    for pos, (si, sample) in enumerate(indexed):
        logger.info("Sample %d (index %d) of %d — building memory",
                    pos + 1, si, len(indexed))

        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            if summaries:
                suffix = f"_gs-{summaries}"
            else:
                suffix = ("_notes" + ("" if group_scoped else "_nogrp")
                          ) if consolidate_memory else ""
            cache_file = os.path.join(cache_dir, f"locomo_sample_{si}{suffix}.pkl")

        n_ledger0 = len(LEDGER.records)
        mark_b = costmeter.begin() if measure_cost else None
        save_s = 0.0          # writing the cache is persistence, not construction
        t0 = time.time()
        if summaries and not (cache_file and os.path.exists(cache_file)):
            raise SystemExit(
                f"sample {si}: --summaries {summaries} requested but {cache_file} "
                f"does not exist. Run group_summaries.py --cache_dir {cache_dir} first.")
        if cache_file and os.path.exists(cache_file):
            layer = RecursiveMemoryLayer.load(cache_file, llm)
            layer.llm = llm
            build_s = 0.0
        else:
            base_cache = (os.path.join(cache_dir, f"locomo_sample_{si}.pkl")
                          if cache_dir else None)
            if base_cache and os.path.exists(base_cache):
                layer = RecursiveMemoryLayer.load(base_cache, llm)
                layer.llm = llm
            else:
                layer = build_memory(sample, mem_config, llm)
                if base_cache:
                    t_save = time.time()
                    layer.save(base_cache)
                    save_s = time.time() - t_save
            build_s = time.time() - t0 - save_s
            build_times.append(build_s)
            if consolidate_memory:
                LEDGER.begin_scope()
                cinfo = consolidate(layer, llm,
                                    max_calls_per_session=consolidate_calls,
                                    use_llm_reconcile=not no_llm_reconcile,
                                    group_scoped=group_scoped)
                cinfo["llm"] = LEDGER.scope_summary()
                consolidation_info.append(cinfo)
                logger.info("  consolidated: %s", cinfo)
            if cache_file:
                layer.save(cache_file)

        stats = layer.get_stats()
        logger.info("  built in %.3fs — %s", build_s, stats)

        build_report.append({
            "sample": si, "built": build_s > 0, "build_s": round(build_s, 4),
            "save_s": round(save_s, 4),
            "n_episodes": len(layer.all_episodes()),
            "stages": costmeter.since(mark_b) if mark_b is not None else {},
            "llm": _ledger_since(n_ledger0),
        })
        if build_only:
            logger.info("  build only: %.2fs, %d episodes", build_s,
                        build_report[-1]["n_episodes"])
            continue

        if system == "memfit":
            if summaries:
                n_sum = sum(1 for x in layer.notes.all()
                            if getattr(x, "attribute", "") == "summary")
                if n_sum == 0:
                    raise SystemExit(
                        f"sample {si}: --summaries {summaries} was requested "
                        f"but the layer holds no summary notes. Run "
                        f"group_summaries.py first; measuring this would "
                        f"silently reproduce base.")
            encoder_info = startup_check(layer, embedder, rerank=rlm_config.rerank)
            retriever: Any = RLMController(layer, rlm_config)
        else:
            retriever = BaselineRetriever(layer, system, rlm_config, top_k=top_k)

        questions = [qa for qa in sample.qa if int(qa.category or 0) in ALLOWED_CATEGORIES]
        if max_questions and len(questions) > max_questions:
            import random

            rng = random.Random(question_seed + si)
            idx = sorted(rng.sample(range(len(questions)), max_questions))
            questions = [questions[i] for i in idx]

        for qi, qa in enumerate(questions):
            LEDGER.begin_scope()
            mark_q = costmeter.begin() if measure_cost else None
            t_q = time.time()
            try:
                out = retriever.retrieve(qa.question)
            except Exception:
                logger.exception("retrieval failed for: %s", qa.question)
                out = {"answer": "", "retrieved_evidence_ids": [], "trajectory": []}
            latency = time.time() - t_q
            usage = LEDGER.scope_summary()
            qcost = None
            if mark_q is not None:
                qcost = costmeter.question_cost(costmeter.since(mark_q), latency, usage)
                question_costs.append(qcost)

            if usage["llm_calls"] != expected_calls:
                budget_violations += 1

            prediction = out.get("answer", "") or ""
            gold = qa.final_answer or ""

            am = answer_metrics(prediction, gold)

            candidates = (out.get("candidate_evidence_ids")
                          or out.get("retrieved_evidence_ids", []))
            rm = retrieval_metrics(candidates, qa.evidence or [], ks=RECALL_KS)

            gold_set = {g for g in (qa.evidence or []) if g}
            pack = set(out.get("retrieved_evidence_ids", []))
            if gold_set:
                rm["pack_hit"] = 1.0 if (gold_set & pack) else 0.0
                rm["pack_all"] = 1.0 if gold_set <= pack else 0.0

            rec: Dict[str, Any] = {
                "category": int(qa.category),
                "category_name": category_name(qa.category),
                **am,
                **{k: v for k, v in rm.items() if k != "n_gold"},
                "llm_calls": float(usage["llm_calls"]),
                "prompt_tokens": float(usage["prompt_tokens"]),
                "completion_tokens": float(usage["completion_tokens"]),
                "total_tokens": float(usage["total_tokens"]),
                "latency_s": latency,
            }
            records.append(rec)

            per_question.append({
                "sample": si,
                "question": qa.question,
                "gold": gold,
                "prediction": prediction,
                "category": int(qa.category),
                "category_name": category_name(qa.category),
                "gold_evidence": qa.evidence,
                "retrieved_evidence": out.get("retrieved_evidence_ids", [])[:30],
                "metrics": rec,
                "usage": usage,
                "cost": qcost,
                "trajectory": out.get("trajectory", []),
                "reasoning": (out.get("reasoning") or "")[:400],
            })

            if (qi + 1) % 10 == 0 or qi == len(questions) - 1:
                done = len(records)
                run_f1 = sum(r["f1"] for r in records) / done
                run_hit = sum(r.get("hit@10", 0.0) for r in records) / done
                logger.info("  [%d/%d] running F1=%.4f hit@10=%.3f calls=%d",
                            qi + 1, len(questions), run_f1, run_hit,
                            usage["llm_calls"])

    if build_only:
        result = {"run_name": run_name, "benchmark": "locomo", "model": model,
                  "build_only": True, "embedder": embedder,
                  "n_samples": len(indexed), "build_report": build_report,
                  "build_s_total": round(sum(b["build_s"] for b in build_report), 3),
                  "llm_totals": LEDGER.totals()}
        if output_path:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("wrote %s", output_path)
        return result

    agg = aggregate(records, category_key="category_name")
    totals = LEDGER.totals()

    result = {
        "run_name": run_name,
        "system": system,
        "model": model,
        "backend": backend,
        "config": {
            "top_k_episodes": top_k,
            "lexical_scorer": lexical_scorer,
            "fusion": fusion,
            "alpha": alpha,
            "scan_top_k": scan_top_k,
            "rerank": rerank,
            "planner_style": planner_style,
            "consolidated": consolidate_memory,
            "use_notes": use_notes,
            "note_mode": note_mode,
            "reader_prompt": reader_prompt,
            "reader_question": reader_question,
            "pool_strategy": pool_strategy,
            "enable_planner": enable_planner,
            "categories": list(ALLOWED_CATEGORIES),
            "metric": "squad_token_f1",
            "embedder": embedder,
            "reranker": reranker or "default",
            "rerank_pool": rerank_pool,
            "context_window": context_window,
            "expand_summaries": expand_summaries,
            "prf_docs": prf_docs,
            "summaries": summaries,
            "encoder_check": encoder_info,
            "reranker_health": final_check(rerank),
        },
        "question_seed": question_seed,
        "n_questions": len(records),
        "n_samples": len(indexed),
        "expected_llm_calls_per_question": expected_calls,
        "llm_call_budget_violations": budget_violations,
        "consolidation": consolidation_info,
        "mean_build_seconds_per_sample": (
            sum(build_times) / len(build_times) if build_times else None
        ),
        "llm_totals": totals,
        "build_report": build_report,
        "cost_summary": (costmeter.summarise_questions(question_costs)
                         if measure_cost else None),
        "aggregate": agg,
        "per_question": per_question,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("wrote %s", output_path)

    print_summary(result)
    return result


def print_summary(result: Dict[str, Any]) -> None:
    agg = result["aggregate"]
    n = result["n_questions"]
    print("\n" + "=" * 78)
    print(f"{result['system']}  |  {result['model']}  |  {n} questions  "
          f"|  {result['run_name']}")
    print("=" * 78)

    order = ["multi_hop", "temporal", "open_domain", "single_hop"]
    print(f"{'category':<14} {'n':>4}  {'F1':>16}  {'BLEU-1':>7}  {'EM':>6}  "
          f"{'packhit':>7}  {'rec@10':>7}")
    print("-" * 78)
    for cat in order:
        key = f"category_{cat}"
        if key not in agg:
            continue
        c = agg[key]
        f1 = c["f1"]
        print(f"{cat:<14} {int(f1['n']):>4}  "
              f"{f1['mean'] * 100:>6.2f} [{f1['ci_low'] * 100:5.2f},{f1['ci_high'] * 100:5.2f}]  "
              f"{c['bleu1']['mean'] * 100:>7.2f}  {c['exact_match']['mean'] * 100:>6.2f}  "
              f"{c.get('pack_hit', {}).get('mean', 0) * 100:>7.2f}  "
              f"{c.get('recall@10', {}).get('mean', 0) * 100:>7.2f}")
    print("-" * 78)
    o = agg.get("overall", {})
    if o:
        f1 = o["f1"]
        print(f"{'OVERALL':<14} {int(f1['n']):>4}  "
              f"{f1['mean'] * 100:>6.2f} [{f1['ci_low'] * 100:5.2f},{f1['ci_high'] * 100:5.2f}]  "
              f"{o['bleu1']['mean'] * 100:>7.2f}  {o['exact_match']['mean'] * 100:>6.2f}  "
              f"{o.get('pack_hit', {}).get('mean', 0) * 100:>7.2f}  "
              f"{o.get('recall@10', {}).get('mean', 0) * 100:>7.2f}")

    print("\nRetrieval (gold evidence turns, over the ranked candidate list):")
    for k in RECALL_KS:
        hit = o.get(f"hit@{k}", {}).get("mean", 0) * 100
        rec = o.get(f"recall@{k}", {}).get("mean", 0) * 100
        alk = o.get(f"all@{k}", {}).get("mean", 0) * 100
        print(f"  @{k:<3} hit {hit:6.2f}   recall {rec:6.2f}   all-gold {alk:6.2f}")
    print(f"  MRR   {o.get('mrr', {}).get('mean', 0):.4f}")
    print(f"  in the pack the reader saw: hit {o.get('pack_hit', {}).get('mean', 0) * 100:6.2f}"
          f"   all-gold {o.get('pack_all', {}).get('mean', 0) * 100:6.2f}")
    empty = sum(1 for r in result["per_question"] if not (r["prediction"] or "").strip())
    refused_with_gold = sum(
        1 for r in result["per_question"]
        if not (r["prediction"] or "").strip() and r["metrics"].get("pack_hit", 0) == 1.0
    )
    print(f"  refusals: {empty}/{n}  (of which {refused_with_gold} had gold evidence "
          f"in the pack)")

    t = result["llm_totals"]
    print(f"\nCost: {t['llm_calls']} LLM calls ({t['calls_by_role']}), "
          f"{t['total_tokens']:,} tokens "
          f"({t['prompt_tokens']:,} in / {t['completion_tokens']:,} out)")
    if n:
        print(f"      per question: {t['llm_calls'] / n:.2f} calls, "
              f"{t['total_tokens'] / n:,.0f} tokens, "
              f"{o.get('latency_s', {}).get('mean', 0):.2f}s")
    if result["mean_build_seconds_per_sample"] is not None:
        print(f"      memory build: {result['mean_build_seconds_per_sample']:.3f}s "
              f"per conversation (LLM-free)")
    print(f"      call-budget violations: {result['llm_call_budget_violations']}"
          f" (expected {result['expected_llm_calls_per_question']}/question)")
    if t["failures"]:
        print(f"      WARNING: {t['failures']} failed/empty LLM responses")
    if t.get("truncated"):
        print(f"      WARNING: {t['truncated']} prompts exceeded the context "
              f"window and were silently truncated")
    print("=" * 78 + "\n")

def main() -> None:
    p = argparse.ArgumentParser(description="LoCoMo evaluation for MemFit")
    p.add_argument("--dataset", default="data/locomo10.json")
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    p.add_argument("--ollama_host", default="http://localhost:11434")
    p.add_argument("--api_key", default=None)
    p.add_argument("--system", default="memfit",
                   choices=["memfit", "bm25", "dense", "hybrid", "fullcontext"])
    p.add_argument("--output", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=None,
                   help="number of conversations (default: all 10)")
    p.add_argument("--max_questions", type=int, default=None,
                   help="cap questions per conversation")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--reader_prompt", default="uniform", choices=["uniform", "legacy", "reason"])
    p.add_argument("--context_window", type=int, default=0,
                   help="also show the reader the turns within N of each retrieved turn")
    p.add_argument("--reader_question", default="original", choices=["original", "plan"])
    p.add_argument("--pool_strategy", default="roundrobin", choices=["roundrobin", "score"])
    p.add_argument("--planner", action="store_true",
                   help="enable the LLM planner (off by default: measured harmful)")
    p.add_argument("--cache_dir", default="cache_locomo")
    p.add_argument("--run_name", default="run")
    p.add_argument("--question_seed", type=int, default=0)
    p.add_argument("--num_ctx", type=int, default=None)
    p.add_argument("--lexical_scorer", default="bm25", choices=["bm25", "fuzzy"])
    p.add_argument("--fusion", default="convex", choices=["convex", "rrf", "max"])
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--scan_top_k", type=int, default=30)
    p.add_argument("--no_rerank", action="store_true")
    p.add_argument("--no_rerank_normalise", action="store_true",
                   help="compare cross-encoder scores across sources on their "
                        "raw scales (pre-round-8 behaviour)")
    p.add_argument("--embedder", default="all-MiniLM-L6-v2",
                   help="sentence-transformers model for episodes AND queries; "
                        "needs a cache_dir of its own")
    p.add_argument("--reranker", default=None,
                   help="cross-encoder model (default ms-marco-MiniLM-L-6-v2)")
    p.add_argument("--rerank_pool", type=int, default=40)
    p.add_argument("--build_only", action="store_true",
                   help="build (or load) each memory, record its cost, answer nothing")
    p.add_argument("--measure_cost", action="store_true",
                   help="time the encoders per question and per build (costmeter)")
    p.add_argument("--expand_summaries", type=int, default=0,
                   help="pull N matching summaries' member turns into the "
                        "candidate pool (summaries as an index)")
    p.add_argument("--prf_docs", type=int, default=0,
                   help="pseudo-relevance feedback: re-query using the top "
                        "N episodes' rarest terms")
    p.add_argument("--summaries", default=None,
                   help="use layers prebuilt by group_summaries.py "
                        "with this grouping method (e.g. segment)")
    p.add_argument("--consolidate", action="store_true",
                   help="Phase 3: distil notes per session before answering")
    p.add_argument("--no_group_scope", action="store_true",
                   help="reconcile against same-subject notes only (pre-group behaviour)")
    p.add_argument("--consolidate_calls", type=int, default=2,
                   help="max LLM calls per session during consolidation")
    p.add_argument("--no_llm_reconcile", action="store_true",
                   help="reconcile by recency only, without the adjudication call")
    p.add_argument("--note_mode", default="compete", choices=["compete", "additive", "route"],
                   help="whether notes take pack slots from episodes or are appended")
    p.add_argument("--notes", action="store_true",
                   help="retrieve consolidated notes (off by default: measured harmful on LoCoMo)")
    p.add_argument("--planner_style", default="conservative", choices=["conservative", "multiquery", "adaptive"])
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("load_dataset").setLevel(logging.WARNING)

    here = _ROOT
    dataset = args.dataset if os.path.isabs(args.dataset) else os.path.join(here, args.dataset)
    output = args.output or os.path.join(
        here, "results", f"locomo_{args.system}_{args.run_name}.json"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    evaluate(
        dataset_path=dataset,
        model=args.model,
        backend=args.backend,
        system=args.system,
        output_path=output,
        num_samples=args.num_samples,
        shard=args.shard, num_shards=args.num_shards,
        max_questions=args.max_questions,
        top_k=args.top_k,
        ollama_host=args.ollama_host,
        api_key=args.api_key,
        reader_prompt=args.reader_prompt,
        reader_question=args.reader_question,
        pool_strategy=args.pool_strategy,
        enable_planner=args.planner,
        cache_dir=os.path.join(here, args.cache_dir) if args.cache_dir else None,
        run_name=args.run_name,
        question_seed=args.question_seed,
        num_ctx=args.num_ctx,
        lexical_scorer=args.lexical_scorer,
        fusion=args.fusion,
        alpha=args.alpha,
        scan_top_k=args.scan_top_k,
        rerank=not args.no_rerank,
        consolidate_memory=args.consolidate,
        consolidate_calls=args.consolidate_calls,
        no_llm_reconcile=args.no_llm_reconcile,
        use_notes=args.notes,
        note_mode=args.note_mode,
        summaries=args.summaries,
        expand_summaries=args.expand_summaries,
        embedder=args.embedder, reranker=args.reranker,
        rerank_pool=args.rerank_pool,
        context_window=args.context_window,
        prf_docs=args.prf_docs,
        rerank_normalise=not args.no_rerank_normalise,
        group_scoped=not args.no_group_scope,
        planner_style=args.planner_style,
        build_only=args.build_only, measure_cost=args.measure_cost,
    )


if __name__ == "__main__":
    main()
