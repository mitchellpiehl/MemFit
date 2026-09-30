# MemFit

Code for **MemFit: EFFECIENT LONG-TERM AGENTIC MEMORY**, with the scripts that reproduce the paper's LoCoMo results (Table 1 and the ablation and sensitivity studies).

```
MemFit/
├── memfit/       
│   ├── memory_layer.py      append-only episode store (BM25 index + embedding matrix)
│   ├── indexes.py           BM25 index
│   ├── encoders.py          sentence encoder and cross-encoder
│   ├── grouping_methods.py  TextTiling segmentation (method "segment")
│   ├── grouping.py          grouping utilities
│   ├── group_summaries.py   segment summaries: one LLM call per segment
│   ├── consolidation.py     summary store (hybrid search over summaries)
│   ├── rlm_controller.py    retrieval: hybrid scan, expansions, rerank, answer call
│   ├── llm_controller.py    Ollama / OpenAI backends and call ledger
│   └── costmeter.py         LLM call and token metering
├── experiments/     LoCoMo evaluation
│   ├── test_locomo.py       evaluation harness
│   ├── load_dataset.py      LoCoMo loader
│   ├── metrics.py           F1, BLEU-1, retrieval metrics
│   ├── baselines.py         plain retrieval baselines sharing the same store
│   ├── judge.py             LLM judge prompt and cache
│   ├── rejudge_v2.py        adds LLM-judge verdicts to a results file
│   └── paper_tables.py      per-category tables (F1 / BLEU-1 / judge)
└── data/           
```

Below are the instructions to recreate the locomo results.
Every command below is run from this directory (`MemFit/`). 

## 1. Setup

```bash
pip install -r requirements.txt
```

Open-weight backbones and the judge are served by [Ollama](https://ollama.com):

```bash
ollama pull qwen3:8b
ollama pull gemma3:27b
ollama pull gpt-oss:20b
```

GPT backbones use the OpenAI API: `export OPENAI_API_KEY=...`.

The encoders (`all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`) are downloaded by `sentence-transformers` on first use. 

## 2. Data

Download `locomo10.json` from the LoCoMo repository (https://github.com/snap-research/locomo, `data/locomo10.json`) and place it at `data/locomo10.json`. The 1,540 non-adversarial questions are evaluated; category 5 (adversarial) is excluded.

## 3. Build the memory

**Episode store** (LLM-free: embeddings and index only):

```bash
python experiments/test_locomo.py --build_only --run_name build
```

**Segment summaries** (one LLM call per segment; 437 calls for the ten conversations):

```bash
python memfit/group_summaries.py --benchmark locomo --model qwen3:8b
```

## 4. Run MemFit

The full configuration used throughout the paper:

```bash
FULL="--summaries segment --notes --note_mode compete --expand_summaries 5 --prf_docs 2 --scan_top_k 60 --rerank_pool 80 --context_window 1"
```

```bash
# open-weight backbones (Ollama)
python experiments/test_locomo.py --model qwen3:8b   $FULL --run_name paper_lc_full_c
python experiments/test_locomo.py --model gemma3:27b $FULL --run_name paper_lc_full_c_gemma3-27b

# proprietary backbones (OpenAI API)
python experiments/test_locomo.py --backend openai --model gpt-4.1-mini $FULL --run_name paper_lc_full_c_gpt-4.1-mini
python experiments/test_locomo.py --backend openai --model gpt-4o       $FULL --run_name paper_lc_full_c_gpt-4o
```

Use `--ollama_host http://localhost:PORT` to point at another Ollama instance. Each run makes exactly one LLM call per question.

## 5. Judge and tabulate

LLM-judge accuracy uses `gpt-oss:20b` through Ollama with the prompt in `experiments/judge.py`. The judge writes a `.j2.json` file next to each result:

```bash
python experiments/rejudge_v2.py --ports 11434 results/locomo_memfit_paper_*.json
```

Per-category F1 and BLEU-1, their four-category average (the "Average" column of Table 1), micro F1, and judge accuracy:

```bash
python experiments/paper_tables.py --run paper --bench locomo
```

## 6. Ablation and sensitivity

Each ablation removes one component from `$FULL`:

| Configuration | Flags |
|---|---|
| Full MemFit | `$FULL` |
| w/o context window | `--summaries segment --notes --note_mode compete --expand_summaries 5 --prf_docs 2 --scan_top_k 60 --rerank_pool 80` |
| w/o pseudo-relevance feedback | `--summaries segment --notes --note_mode compete --expand_summaries 5 --scan_top_k 60 --rerank_pool 80 --context_window 1` |
| w/o summary expansion | `--summaries segment --notes --note_mode compete --prf_docs 2 --scan_top_k 60 --rerank_pool 80 --context_window 1` |
| w/o wide first-stage scan | `--summaries segment --notes --note_mode compete --expand_summaries 5 --prf_docs 2 --context_window 1` |
| w/o summaries | `--prf_docs 2 --scan_top_k 60 --rerank_pool 80 --context_window 1` |
| Plain retrieval | (no flags) |

Example:

```bash
python experiments/test_locomo.py --model gemma3:27b --prf_docs 2 --scan_top_k 60 --rerank_pool 80 --context_window 1 --run_name paper_lc_no_summaries_gemma3-27b
```

Sensitivity changes one setting of `$FULL` at a time: `--top_k 5|20|30`, `--context_window 2`, `--prf_docs 4`, `--expand_summaries 10`, `--scan_top_k 30 --rerank_pool 40` (narrower pools), and `--scan_top_k 100 --rerank_pool 120` (wider scan).

## 7. Expected results

Four-category average F1 / BLEU-1 and LLM-judge accuracy (%) of the full system on LoCoMo:

| Backbone | Avg. F1 | Avg. BLEU-1 | Judge |
|---|---|---|---|
| gpt-4.1-mini | 50.26 | 44.74 | 72.37 |
| gpt-4o | 49.33 | 43.96 | 64.91 |
| qwen3:8b | 44.19 | 39.38 | 61.82 |
| gemma3:27b | 45.42 | 39.92 | 64.59 |

Ablation with `gemma3:27b` (Avg. F1 / judge): w/o context window 43.56 / 62.87, w/o pseudo-relevance feedback 44.07 / 63.68, w/o summary expansion 44.29 / 64.46, w/o wide scan 44.53 / 63.94, w/o summaries 44.59 / 63.94, plain retrieval 42.10 / 59.09.

Answer extraction uses greedy decoding, and retrieval is deterministic. Rerunning `qwen3:8b` from a fresh build reproduced 99.8% of its answers. Larger models served by Ollama can vary slightly between runs, so expect small differences in individual answers.

## 8. Cost

Add `--measure_cost` to a run to record LLM calls, tokens, and time per stage. With `gemma3:27b` on one GPU, building the memory of all ten conversations takes about 16 minutes, 15 seconds of it LLM-free storage and indexing and the rest the summary calls. Answering takes about 1.3 s per question.
