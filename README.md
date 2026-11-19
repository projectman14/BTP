# Context Window Analysis for LLMs — Phase 2: Persistent Memory Layer

B.Tech Project (LNMIIT, 2026–27) — Lakshya Jain, Monish Mathur, Nishit Luhadia. Guide: Dr. Rahul Sharma.

Phase 1 analysed the limits of long context windows and RAG and *proposed* a memory layer.
Phase 2 (this repository) *builds* it and benchmarks it against a plain LLM, full-context prompting,
a sliding window and plain vector RAG, all with the same local model (Qwen2.5-1.5B-Instruct).

## Layout

```
memory_layer/            the package
  config.py              all knobs + ablation presets
  text_utils.py          tokeniser, third-person normalisation
  embeddings.py          sentence-transformer embedder, cross-encoder re-ranker
  vector_store.py        FAISS flat index
  keyword_index.py       BM25 index
  graph_memory.py        entity extraction + memory/entity graph (networkx)
  query_expansion.py     rule-based synonym expansion for BM25
  scoring.py             importance / recency / access scores
  retriever.py           hybrid multi-stage retriever (RRF -> graph -> rerank -> scoring -> conflicts)
  pipeline.py            MemoryLayer: ingest / retrieve / build_context / answer / grounding gate
  llm.py                 local HF LLM wrapper (batched, chunked prefill for long prompts)
benchmarks/
  make_dataset.py        synthetic long-term conversational memory benchmark
  run_eval.py            main comparison of answering strategies (needs the LLM)
  run_ablation.py        retrieval-only ablation ladder, pruning and conflict-threshold sweeps
  run_niah.py            needle-in-a-haystack scaling experiment (1k–16k tokens)
  tune.py                grid search on the dev split
  plots.py               figures + LaTeX tables from results/*.json
data/                    generated benchmark files
results/                 JSON results, tables.tex
report/                  LaTeX source of the Phase-2 report (compiled with Tectonic)
```

## Quick start

```bash
pip install -r requirements.txt
python benchmarks/make_dataset.py --users 20 --out data/longmem_synth.json
python benchmarks/make_dataset.py --users 6 --seed 99 --out data/dev_synth.json
python benchmarks/run_ablation.py --data data/longmem_synth.json --out results/ablation.json
python benchmarks/run_eval.py --data data/longmem_synth.json --model Qwen/Qwen2.5-1.5B-Instruct --out results/main_results.json
python benchmarks/run_niah.py --model Qwen/Qwen2.5-1.5B-Instruct --out results/niah.json
python benchmarks/plots.py
cd report && tectonic main.tex
```

## Using the memory layer

```python
from memory_layer import MemoryLayer, MemoryConfig, LocalLLM

ml = MemoryLayer(MemoryConfig())
ml.ingest_session([{"role": "user", "text": "I live in Jaipur."},
                   {"role": "assistant", "text": "Noted!"}], session=1, timestamp="2026-01-05")
ml.ingest_session([{"role": "user", "text": "I moved to Pune last month."}], session=2)
llm = LocalLLM("Qwen/Qwen2.5-1.5B-Instruct")
print(ml.answer("Where does the user live now?", llm, current_session=3)["answer"])
```
