"""Main comparison: baseline LLM vs context-window approaches vs RAG-style memory vs the proposed
memory layer, all answered by the same local LLM.

python benchmarks/run_eval.py --data data/longmem_synth.json --out results/main_results.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_layer import MemoryLayer, MemoryConfig, PRESETS, LocalLLM
from memory_layer.embeddings import Embedder, Reranker
from memory_layer.pipeline import SYSTEM_PROMPT, is_abstention
from benchmarks.common import (load_dataset, is_correct, is_hallucination, summarise, prf, format_history)

METHODS = ["no_memory", "full_context", "sliding_window", "rag_vector", "memory_layer"]


def build_layer(preset: str, embedder, reranker, **kw) -> MemoryLayer:
    cfg = MemoryConfig().with_(**PRESETS[preset], **kw)
    return MemoryLayer(cfg, embedder=embedder, reranker=reranker if cfg.use_rerank else None)


def ingest_user(layer: MemoryLayer, user: dict) -> dict[int, str]:
    """Feed the sessions in order; returns memory-id -> turn-id map."""
    mid_to_turn = {}
    for s in user["sessions"]:
        mids = layer.ingest_session([{"role": t["role"], "text": t["text"]} for t in s["turns"]],
                                    s["session"], s["date"])
        for t, mid in zip(s["turns"], mids):
            if mid is not None:
                mid_to_turn[mid] = t["id"]
    return mid_to_turn


def retrieval_metrics(retrieved_turns: list[str], gold_turns: list[str]) -> tuple[bool, float]:
    hit = all(g in retrieved_turns for g in gold_turns) if gold_turns else True
    rr = 0.0
    for i, t in enumerate(retrieved_turns):
        if t in gold_turns:
            rr = 1.0 / (i + 1)
            break
    return hit, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/longmem_synth.json")
    ap.add_argument("--out", default="results/main_results.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--users", type=int, default=0, help="limit number of users (0 = all)")
    ap.add_argument("--window_turns", type=int, default=40)
    ap.add_argument("--max_ctx", type=int, default=20000)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    data = load_dataset(args.data)
    users = data["users"][: args.users] if args.users else data["users"]
    methods = args.methods.split(",")

    llm = LocalLLM(args.model, max_context=args.max_ctx)
    embedder = Embedder(MemoryConfig.embedding_model)
    reranker = Reranker(MemoryConfig.reranker_model)
    print(f"LLM on {llm.device}; {len(users)} users; methods={methods}", flush=True)

    records: dict[str, list[dict]] = {m: [] for m in methods}
    records["memory_layer_gated"] = []
    records["rag_vector_gated"] = []
    layer_stats = []
    t_start = time.time()

    for ui, user in enumerate(users):
        qs = user["questions"]
        last_session = user["sessions"][-1]["session"] + 1
        base = [dict(qid=q["qid"], type=q["type"], answerable=q["answerable"], gold=q["answer"],
                     question=q["question"]) for q in qs]

        # ---- prompt-only baselines ---------------------------------------------
        if "no_memory" in methods:
            prompts = [f"Memory notes:\n(no notes)\n\nQuestion: {q['question']}" for q in qs]
            outs = llm.chat_batch(SYSTEM_PROMPT, prompts, batch_size=16)
            for b, q, o in zip(base, qs, outs):
                records["no_memory"].append(_record(b, o["answer"], o, retrieved=None))

        if "full_context" in methods:
            hist = format_history(user["sessions"])
            hist = llm.truncate_to_tokens(hist, args.max_ctx - 400, keep="tail")
            prompts = [f"Memory notes (full conversation history):\n{hist}\n\nQuestion: {q['question']}" for q in qs]
            outs = llm.chat_batch(SYSTEM_PROMPT, prompts, batch_size=2)
            for b, q, o in zip(base, qs, outs):
                records["full_context"].append(_record(b, o["answer"], o, retrieved=None))

        if "sliding_window" in methods:
            hist = format_history(user["sessions"], turns_limit=args.window_turns)
            prompts = [f"Memory notes (recent conversation):\n{hist}\n\nQuestion: {q['question']}" for q in qs]
            outs = llm.chat_batch(SYSTEM_PROMPT, prompts, batch_size=8)
            for b, q, o in zip(base, qs, outs):
                records["sliding_window"].append(_record(b, o["answer"], o, retrieved=None))

        # ---- retrieval-based memories --------------------------------------------
        for preset in ("rag_vector", "memory_layer"):
            if preset not in methods:
                continue
            layer = build_layer(preset, embedder, reranker, top_k=args.top_k)
            mid_to_turn = ingest_user(layer, user)
            if preset == "memory_layer":
                layer_stats.append(layer.stats())
            prompts, ctxs, rets, robjs = [], [], [], []
            for q in qs:
                r = layer.retrieve(q["question"], last_session)
                ctx = layer.build_context(r, llm.count_tokens)
                prompts.append(layer.make_prompt(q["question"], ctx))
                ctxs.append(ctx)
                rets.append([mid_to_turn[x.id] for x in r])
                robjs.append(r)
            outs = llm.chat_batch(SYSTEM_PROMPT, prompts, batch_size=16)
            for b, q, o, ctx, ret, r in zip(base, qs, outs, ctxs, rets, robjs):
                hit, rr = retrieval_metrics(ret, q["gold_turns"])
                rec = _record(b, o["answer"], o, retrieved=ret)
                rec.update(recall_hit=hit, rr=rr, context=ctx)
                records[preset].append(rec)
                # grounding gate applied post-hoc on the same generation
                supported, reason = layer.verify(o["answer"], r, q["question"])
                gated = o["answer"] if supported else "I don't know (not supported by memory notes)."
                g = _record(b, gated, o, retrieved=ret)
                g.update(recall_hit=hit, rr=rr, flagged=not supported, gate_reason=reason,
                         raw_answer=o["answer"])
                records[preset + "_gated"].append(g)

        done = sum(len(v) for v in records.values())
        print(f"user {ui + 1}/{len(users)} done; records={done}; elapsed={time.time() - t_start:.0f}s", flush=True)

    summary = {m: summarise(r) for m, r in records.items() if r}
    # hallucination detection quality of the grounding gate (compare with Phase-1 Table 4.1)
    detection = {}
    for preset in ("rag_vector", "memory_layer"):
        rs = records.get(preset + "_gated", [])
        if rs:
            truth = [is_hallucination(r["raw_answer"], r["gold"], r["answerable"]) for r in rs]
            pred = [r["flagged"] for r in rs]
            detection[preset] = prf(truth, pred)
    out = {"meta": {"model": args.model, "device": llm.device, "users": len(users), "top_k": args.top_k,
                    "window_turns": args.window_turns, "max_ctx": args.max_ctx,
                    "dataset_meta": data["meta"], "elapsed_s": time.time() - t_start},
           "summary": summary, "hallucination_detection": detection,
           "layer_stats": layer_stats, "records": records}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({m: {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}
                      for m, s in summary.items()}, indent=1))
    print("detection:", json.dumps(detection, indent=1))


def _record(base: dict, answer: str, o: dict, retrieved):
    r = dict(base)
    r.update(answer=answer, correct=is_correct(answer, base["gold"], base["answerable"]),
             halluc=is_hallucination(answer, base["gold"], base["answerable"]),
             abstained=is_abstention(answer), prompt_tokens=o["prompt_tokens"], latency=o["latency"],
             retrieved=retrieved)
    return r


if __name__ == "__main__":
    main()
