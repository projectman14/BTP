"""Grid search of retrieval hyper-parameters on the *dev* split (retrieval-only, no LLM).

python benchmarks/tune.py --data data/dev_synth.json
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_layer import MemoryLayer, MemoryConfig
from memory_layer.embeddings import Embedder, Reranker
from benchmarks.common import load_dataset
from benchmarks.run_eval import ingest_user
from benchmarks.run_ablation import evaluate, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dev_synth.json")
    ap.add_argument("--out", default="results/tuning.json")
    args = ap.parse_args()
    users = load_dataset(args.data)["users"]
    embedder = Embedder(MemoryConfig.embedding_model)
    reranker = Reranker(MemoryConfig.reranker_model)

    # ingest once per user, reuse the same stores for every configuration (retrieval is read-only
    # apart from access counts, which we reset)
    layers = []
    for u in users:
        layer = MemoryLayer(MemoryConfig(), embedder=embedder, reranker=reranker)
        m2t = ingest_user(layer, u)
        layers.append((layer, u, m2t))

    grid = dict(w_sparse=[0.0, 0.2, 0.4, 0.7, 1.0], rerank_blend=[0.0, 0.3, 0.5],
                graph_propagation=[0.0, 0.7, 0.85], w_recency=[0.0, 0.15, 0.3])
    results = []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(grid.keys(), combo))
        kw["use_graph"] = kw["graph_propagation"] > 0
        recs = []
        for layer, u, m2t in layers:
            layer.cfg = layer.retriever.cfg = MemoryConfig().with_(**kw)
            for m in layer.store:
                m.access_count = 0
            recs += evaluate(layer, u, m2t)
        a = aggregate(recs)
        row = dict(kw, **{k: a[k] for k in ("recall@1", "recall@3", "recall@5", "mrr", "temporal_ok")},
                   multihop_r5=a["by_type"].get("multihop", {}).get("recall@5", 0))
        results.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    results.sort(key=lambda r: (r["recall@5"], r["mrr"]), reverse=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("\nTOP 8:")
    for r in results[:8]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
