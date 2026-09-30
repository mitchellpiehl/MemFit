#!/usr/bin/env python3
from __future__ import annotations

# Repository root (holds data/, cache_*/ and results/); the system code is in memfit/.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "memfit"))

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from judge import Judge, JudgeUnavailable
from llm_controller import LLMController

logger = logging.getLogger("rejudge_v2")
DEFAULT_PORTS = [11434, 11435, 11436, 11437, 11438, 11439, 11440, 11441]
#: Per-shard partial outputs (<run>.s3of8.json). Their answers are all in the
#: merged file; judging them too would only write partial .j2 files that look
#: like extra, incomplete arms.
SHARD = re.compile(r"\.s\d+of\d+\.json$")

Item = Tuple[str, str, str]          # question, gold, prediction


def _worker(port: int, host: str, model: str, cache_dir: str,
            items: List[Tuple[str, Item]]) -> Dict[str, Optional[float]]:
    """One port, one Judge, one private cache file."""
    llm = LLMController(backend="ollama", model=model,
                        ollama_host=f"http://{host}:{port}").llm
    judge = Judge(llm, cache_path=os.path.join(cache_dir, f"judge_v2.p{port}.json"),
                  model_tag=model)
    for other in glob.glob(os.path.join(cache_dir, "judge_v2.p*.json")):
        judge.absorb(other)
    out: Dict[str, Optional[float]] = {}
    t0 = time.time()
    for n, (key, (q, g, p)) in enumerate(items, 1):
        try:
            out[key] = judge.score(q, g, p)
        except JudgeUnavailable as e:
            logger.warning(":%d unjudged: %s", port, e)
            out[key] = None
        if n % 200 == 0:
            logger.info(":%d judged %d/%d (%.1f min)", port, n, len(items),
                        (time.time() - t0) / 60)
    judge.save()
    return out


def needs_judging(path: str) -> bool:
    """False when an up-to-date, fully judged .j2 already exists."""
    out = _out_path(path)
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
        return True
    try:
        return json.load(open(out)).get("unjudged", 1) != 0
    except Exception:
        return True


def _out_path(path: str) -> str:
    stem, ext = os.path.splitext(path)
    return f"{stem}.j2{ext}"


def judge_files(files: List[str], ports: List[int], host: str, model: str,
                cache_dir: str) -> List[Dict[str, Any]]:
    datas = {f: json.load(open(f)) for f in files}
    probe = Judge(None, model_tag=model)             # for keys and the union cache
    for other in glob.glob(os.path.join(cache_dir, "judge_v2.p*.json")):
        probe.absorb(other)

    todo: Dict[str, Item] = {}
    for d in datas.values():
        for q in d["per_question"]:
            item = (q["question"], str(q.get("gold", "")), str(q.get("prediction", "")))
            if not item[1].strip() or not item[2].strip():
                continue                               # scored without a call
            key = Judge._key(*item, model)
            if key not in probe._cache:
                todo[key] = item
    print(f"  {sum(len(d['per_question']) for d in datas.values())} answers in "
          f"{len(files)} files; {len(todo)} distinct answers need a judge call")
    sys.stdout.flush()

    verdicts: Dict[str, Optional[float]] = {k: float(v["score"])
                                            for k, v in probe._cache.items()}
    if todo:
        buckets: List[List[Tuple[str, Item]]] = [[] for _ in ports]
        for n, kv in enumerate(sorted(todo.items())):
            buckets[n % len(ports)].append(kv)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(ports)) as ex:
            futs = [ex.submit(_worker, port, host, model, cache_dir, b)
                    for port, b in zip(ports, buckets) if b]
            for f in futs:
                verdicts.update(f.result())
        print(f"  judged in {(time.time() - t0) / 60:.1f} min")

    rows = []
    for path, d in datas.items():
        n_none = 0
        for q in d["per_question"]:
            q.setdefault("metrics", {})
            q["metrics"].pop("judge", None)
            gold, pred = str(q.get("gold", "")), str(q.get("prediction", ""))
            if not gold.strip() or not pred.strip():
                v: Optional[float] = 0.0
            else:
                v = verdicts.get(Judge._key(q["question"], gold, pred, model))
            if v is None:
                n_none += 1
                continue
            q["metrics"]["judge"] = v
        judged = [q["metrics"]["judge"] for q in d["per_question"] if "judge" in q["metrics"]]
        d["judge_version"] = "v2"
        d["judge_model"] = model
        d["unjudged"] = n_none
        out = _out_path(path)
        tmp = out + ".tmp"
        json.dump(d, open(tmp, "w"), indent=1)
        os.replace(tmp, out)
        rows.append({"file": os.path.basename(path), "n": len(d["per_question"]),
                     "judged": len(judged), "unjudged": n_none,
                     "judge": 100.0 * sum(judged) / len(judged) if judged else float("nan")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--ports", default=",".join(map(str, DEFAULT_PORTS)))
    ap.add_argument("--cache_dir", default="cache_judge_v2")
    ap.add_argument("--force", action="store_true",
                    help="re-write .j2 files even when up to date")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    for n in ("httpx", "urllib3"):
        logging.getLogger(n).setLevel(logging.WARNING)
    here = _ROOT
    cache_dir = os.path.join(here, args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    ports = [int(p) for p in args.ports.split(",") if p.strip()]

    files = []
    for f in args.files:
        if not os.path.exists(f):
            print(f"{os.path.basename(f):52s} MISSING")
        elif f.endswith(".j2.json") or ".INCOMPLETE." in f or SHARD.search(f):
            continue
        elif args.force or needs_judging(f):
            files.append(f)
        else:
            print(f"{os.path.basename(f):52s} up to date")
    if not files:
        return
    rows = judge_files(files, ports, args.host, args.model, cache_dir)
    print(f"{'file':52s} {'n':>5s} {'judged':>7s} {'unjudged':>9s} {'judge':>7s}")
    for r in rows:
        print(f"{r['file'][:52]:52s} {r['n']:5d} {r['judged']:7d} {r['unjudged']:9d} "
              f"{r['judge']:7.2f}")
    if any(r["unjudged"] for r in rows):
        print("NOTE: unjudged answers are excluded from the mean, never scored wrong.")


if __name__ == "__main__":
    main()
