"""Retrieval-only ablation of the memory layer (no LLM needed).

Ladder: vector-only (Phase-1 design) -> +BM25 (RRF) -> +query expansion -> +cross-encoder re-ranking
-> +entity-graph expansion -> +memory scoring & conflict-aware recency (full memory layer).
Also sweeps: top-k, write-time pruning threshold, conflict threshold.

python benchmarks/run_ablation.py --data data/longmem_synth.json --out results/ablation.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_layer import MemoryLayer, MemoryConfig, PRESETS
from memory_layer.embeddings import Embedder, Reranker
from benchmarks.common import load_dataset
from benchmarks.run_eval import ingest_user

LADDER = ["rag_vector", "bm25_only", "hybrid", "hybrid_qe", "hybrid_qe_rerank",
          "hybrid_qe_rerank_graph", "memory_layer"]


def evaluate(layer: MemoryLayer, user: dict, mid_to_turn: dict, ks=(1, 3, 5, 8)) -> list[dict]:
    last_session = user["sessions"][-1]["session"] + 1
    out = []
    for q in user["questions"]:
        if not q["answerable"]:
            continue
        t0 = time.time()
        r = layer.retrieve(q["question"], last_session, top_k=max(ks))
        dt = time.time() - t0
        turns = [mid_to_turn[x.id] for x in r]
        rec = dict(qid=q["qid"], type=q["type"], latency=dt)
        for k in ks:
            rec[f"recall@{k}"] = all(g in turns[:k] for g in q["gold_turns"])
        rr = 0.0
        for i, t in enumerate(turns):
            if t in q["gold_turns"]:
                rr = 1.0 / (i + 1)
                break
        rec["rr"] = rr
        # temporal questions: the correct version must be in the top-5 and, for "current" questions,
        # the outdated version must either be absent, ranked below it, or tagged as an older note
        if q["type"].startswith("temporal"):
            top5 = turns[:5]
            gold_rank = next((i for i, t in enumerate(top5) if t in q["gold_turns"]), None)
            trap_turns = [tid for mid, tid in mid_to_turn.items()
                          if q.get("trap") and q["trap"] in layer.store.get(mid).text
                          and layer.store.get(mid).role == "user"]
            trap_rank = next((i for i, t in enumerate(top5) if t in trap_turns), None)
            trap_tagged = any(x.outdated and mid_to_turn[x.id] in trap_turns for x in r[:5])
            if q["type"] == "temporal_current":
                rec["temporal_ok"] = gold_rank is not None and (trap_rank is None or gold_rank < trap_rank or trap_tagged)
            else:
                rec["temporal_ok"] = gold_rank is not None
        out.append(rec)
    return out


def aggregate(recs: list[dict]) -> dict:
    n = len(recs)
    keys = [k for k in recs[0] if k.startswith("recall@")]
    agg = {k: sum(r[k] for r in recs) / n for k in keys}
    agg["mrr"] = sum(r["rr"] for r in recs) / n
    agg["latency_ms"] = 1000 * sum(r["latency"] for r in recs) / n
    agg["n"] = n
    temporal = [r for r in recs if "temporal_ok" in r]
    if temporal:
        agg["temporal_ok"] = sum(r["temporal_ok"] for r in temporal) / len(temporal)
    agg["by_type"] = {}
    for t in sorted({r["type"] for r in recs}):
        rs = [r for r in recs if r["type"] == t]
        agg["by_type"][t] = {k: sum(r[k] for r in rs) / len(rs) for k in keys}
        agg["by_type"][t]["mrr"] = sum(r["rr"] for r in rs) / len(rs)
        agg["by_type"][t]["n"] = len(rs)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/longmem_synth.json")
    ap.add_argument("--out", default="results/ablation.json")
    ap.add_argument("--users", type=int, default=0)
    args = ap.parse_args()
    data = load_dataset(args.data)
    users = data["users"][: args.users] if args.users else data["users"]
    embedder = Embedder(MemoryConfig.embedding_model)
    reranker = Reranker(MemoryConfig.reranker_model)

    results = {"ladder": {}, "pruning": {}, "conflict_threshold": {}, "meta": {"users": len(users)}}
    t0 = time.time()

    def run(cfg_kw: dict) -> dict:
        recs = []
        stats = []
        for user in users:
            cfg = MemoryConfig().with_(**cfg_kw)
            layer = MemoryLayer(cfg, embedder=embedder, reranker=reranker if cfg.use_rerank else None)
            m2t = ingest_user(layer, user)
            stats.append(layer.stats())
            recs += evaluate(layer, user, m2t)
        agg = aggregate(recs)
        agg["memories_stored"] = sum(s["memories"] for s in stats) / len(stats)
        agg["memories_seen"] = sum(s["seen"] for s in stats) / len(stats)
        return agg

    for name in LADDER:
        results["ladder"][name] = run(PRESETS[name])
        a = results["ladder"][name]
        print(f"{name:26s} R@1={a['recall@1']:.3f} R@3={a['recall@3']:.3f} R@5={a['recall@5']:.3f} "
              f"MRR={a['mrr']:.3f} temporal={a.get('temporal_ok', 0):.2f} {time.time() - t0:.0f}s", flush=True)

    for thr in (0.0, 0.3, 0.45, 0.55, 0.65):
        results["pruning"][str(thr)] = run(dict(min_importance_to_store=thr))
        a = results["pruning"][str(thr)]
        print(f"prune>={thr}: stored={a['memories_stored']:.0f}/{a['memories_seen']:.0f} R@5={a['recall@5']:.3f}", flush=True)

    for thr in (0.3, 0.4, 0.45, 0.5, 0.6, 1.01):
        results["conflict_threshold"][str(thr)] = run(dict(conflict_sim_threshold=thr))
        a = results["conflict_threshold"][str(thr)]
        print(f"conflict thr {thr}: temporal_ok={a.get('temporal_ok', 0):.3f} R@5={a['recall@5']:.3f}", flush=True)

    results["meta"]["elapsed_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
