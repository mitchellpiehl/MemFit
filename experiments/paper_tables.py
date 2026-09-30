#!/usr/bin/env python3
"""
paper_tables.py — per-category numbers for the paper's tables.
"""

from __future__ import annotations

# Repository root (holds data/, cache_*/ and results/); the system code is in memfit/.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "memfit"))

import argparse
import glob
import json
import os
import re
from typing import Dict, List

LC_ORDER = ["multi_hop", "temporal", "open_domain", "single_hop"]
LC_HEAD = ["Multi-Hop", "Temporal", "OpenDomain", "Single-Hop"]
MG_ORDER = ["AR", "CD", "FR", "KR", "MR", "TR", "TTL", "VR", "VS"]


def load(path: str) -> dict:
    j2 = path[:-5] + ".j2.json"
    return json.load(open(j2 if os.path.exists(j2) else path))


def cell(qs: List[dict], key: str) -> float:
    v = [q["metrics"][key] for q in qs
         if (q.get("metrics") or {}).get(key) is not None]
    return 100 * sum(v) / len(v) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bench", default="locomo", choices=["locomo", "memgallery"])
    ap.add_argument("--arms", default="", help="comma list; default all arms of the run")
    ap.add_argument("--results_dir", default=os.path.join(
        _ROOT, "results"))
    args = ap.parse_args()

    pre = "locomo_memfit_" if args.bench == "locomo" else "memgallery_"
    files = [f for f in sorted(glob.glob(os.path.join(args.results_dir,
                                                      f"{pre}{args.run}_*.json")))
             if not re.search(r"\.s\d+of\d+\.json$", f) and not f.endswith(".j2.json")]
    want = [a for a in args.arms.split(",") if a]
    order = LC_ORDER if args.bench == "locomo" else MG_ORDER
    head = LC_HEAD if args.bench == "locomo" else MG_ORDER
    key = "category_name" if args.bench == "locomo" else "category"

    print(f"| arm | reader | " + " | ".join(f"{h} F1/BLEU" if args.bench == "locomo"
                                            else f"{h}" for h in head)
          + " | macro F1 | macro BLEU | micro F1 | judge |")
    print("|---" * (len(head) + 6) + "|")
    rows = {}
    for f in files:
        name = os.path.basename(f)[len(pre):-5]
        arm = name[len(args.run) + 4:]
        if want and not any(arm.startswith(a) for a in want):
            continue
        d = load(f)
        pq = d.get("per_question") or []
        if not pq:
            continue
        by: Dict[str, List[dict]] = {}
        for q in pq:
            by.setdefault(str(q.get(key) or "?").upper() if args.bench == "memgallery"
                          else str(q.get(key)), []).append(q)
        f1s = [cell(by.get(c, []), "f1") for c in order]
        bls = [cell(by.get(c, []), "bleu1") for c in order]
        macro_f1 = sum(x for x in f1s if x == x) / max(1, sum(1 for x in f1s if x == x))
        macro_bl = sum(x for x in bls if x == x) / max(1, sum(1 for x in bls if x == x))
        cells = ([f"{a:.2f}/{b:.2f}" for a, b in zip(f1s, bls)] if args.bench == "locomo"
                 else [f"{a:.2f}" for a in f1s])
        print(f"| {name} | {d.get('model','?')} | " + " | ".join(cells)
              + f" | {macro_f1:.2f} | {macro_bl:.2f} | {cell(pq,'f1'):.2f} "
                f"| {cell(pq,'judge'):.2f} |")
        rows[name] = (f1s, bls, macro_f1, macro_bl)

    if args.bench == "locomo":
        print("\n% LaTeX rows for Table 1 (F1 & BLEU per category, then the "
              "four-category mean):")
        for name, (f1s, bls, mf, mb) in rows.items():
            vals = " & ".join(f"{a:.2f} & {b:.2f}" for a, b in zip(f1s, bls))
            print(f"% {name}\n& \\textbf{{\\name}} & {vals} & {mf:.2f} & {mb:.2f} \\\\")
    else:
        print("\n% LaTeX rows for Table 2 (F1 per category, then Total):")
        for name, (f1s, _, _, _) in rows.items():
            vals = " & ".join(f"{a:.2f}" for a in f1s)
            micro = "see micro F1 column"
            print(f"% {name}\n& \\textbf{{\\name}} & {vals} & {micro} \\\\")


if __name__ == "__main__":
    main()
