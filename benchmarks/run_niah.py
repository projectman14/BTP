"""Needle-in-a-haystack style context-length scaling experiment.

A single personal fact (needle) is buried at a chosen depth inside a filler conversation of
L tokens. The same LLM answers either from the full transcript (context-window approach) or
from the top-k notes selected by the memory layer.

python benchmarks/run_niah.py --out results/niah.json
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_layer import MemoryLayer, MemoryConfig, LocalLLM
from memory_layer.embeddings import Embedder, Reranker
from memory_layer.pipeline import SYSTEM_PROMPT
from benchmarks.common import is_correct, is_hallucination
from benchmarks.make_dataset import ATTRS, FILLER_USER, FILLER_ASSISTANT, ACKS, DISTRACTORS, NAMES


def make_history(rng: random.Random, n_pairs: int, needle: str, depth: float, pairs_per_session: int = 12,
                 n_distractors: int = 6, exclude_values: set | None = None):
    """Build sessions of filler; insert the needle at the given depth, plus a few distractor facts."""
    turns = []
    for _ in range(n_pairs):
        turns.append(dict(role="user", text=rng.choice(FILLER_USER)))
        turns.append(dict(role="assistant", text=rng.choice(FILLER_ASSISTANT)))
    pos = int(round(depth * (n_pairs - 1))) * 2
    turns[pos] = dict(role="user", text=needle, needle=True)
    turns[pos + 1] = dict(role="assistant", text=rng.choice(ACKS))
    # distractors somewhere else
    free = [i for i in range(0, len(turns), 2) if i != pos]
    for i in rng.sample(free, min(n_distractors, len(free))):
        a, tmpl = rng.choice(DISTRACTORS)
        v = rng.choice([x for x in ATTRS[a]["values"] if x not in (exclude_values or set())])
        turns[i] = dict(role="user", text=tmpl.format(n=rng.choice(NAMES), v=v))
        turns[i + 1] = dict(role="assistant", text=rng.choice(ACKS))
    sessions = []
    for s in range(0, len(turns), 2 * pairs_per_session):
        sessions.append(dict(session=len(sessions) + 1, date=f"2026-{(len(sessions) % 12) + 1:02d}-07",
                             turns=turns[s:s + 2 * pairs_per_session]))
    return sessions


def render(sessions):
    lines = []
    for s in sessions:
        lines.append(f"[Session {s['session']}, {s['date']}]")
        for t in s["turns"]:
            lines.append(f"{t['role'].capitalize()}: {t['text']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/niah.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--lengths", default="1000,2000,4000,8000,16000")
    ap.add_argument("--depths", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--needles", type=int, default=4, help="needles per (length, depth) cell")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    rng = random.Random(args.seed)

    llm = LocalLLM(args.model, max_context=max(lengths) + 2000)
    embedder = Embedder(MemoryConfig.embedding_model)
    reranker = Reranker(MemoryConfig.reranker_model)
    # tokens per filler pair, measured with the LLM tokenizer
    probe = make_history(random.Random(0), 60, "I live in Jaipur.", 0.5)
    tok_per_pair = llm.count_tokens(render(probe)) / 60
    print(f"device={llm.device}; ~{tok_per_pair:.1f} tokens per dialogue pair", flush=True)

    cells = []
    t0 = time.time()
    for L in lengths:
        n_pairs = max(4, int(L / tok_per_pair))
        for d in depths:
            for i in range(args.needles):
                attr = rng.choice([a for a in ATTRS if "update" not in ATTRS[a]])
                spec = ATTRS[attr]
                v = rng.choice(spec["values"])
                needle = rng.choice(spec["stmts"]).format(v=v)
                needle = needle[0].upper() + needle[1:]
                question = rng.choice(spec["qs"])
                sessions = make_history(rng, n_pairs, needle, d, exclude_values={v})
                # -- full context
                transcript = render(sessions)
                o_full = llm.chat_batch(SYSTEM_PROMPT, [f"Memory notes (full conversation history):\n{transcript}\n\nQuestion: {question}"],
                                        batch_size=1)[0]
                # -- memory layer
                layer = MemoryLayer(MemoryConfig(), embedder=embedder, reranker=reranker)
                t_ing = time.time()
                for s in sessions:
                    layer.ingest_session(s["turns"], s["session"], s["date"])
                t_ing = time.time() - t_ing
                t_ret = time.time()
                r = layer.retrieve(question, len(sessions) + 1)
                t_ret = time.time() - t_ret
                ctx = layer.build_context(r, llm.count_tokens)
                o_mem = llm.chat_batch(SYSTEM_PROMPT, [layer.make_prompt(question, ctx)], batch_size=1)[0]
                needle_found = any(layer.store.get(x.id).raw_text == needle for x in r)
                cells.append(dict(length=L, depth=d, needle_idx=i, attr=attr, answer=v, question=question,
                                  full=dict(answer=o_full["answer"], correct=is_correct(o_full["answer"], v, True),
                                            halluc=is_hallucination(o_full["answer"], v, True),
                                            prompt_tokens=o_full["prompt_tokens"], latency=o_full["latency"]),
                                  memory=dict(answer=o_mem["answer"], correct=is_correct(o_mem["answer"], v, True),
                                              halluc=is_hallucination(o_mem["answer"], v, True),
                                              prompt_tokens=o_mem["prompt_tokens"], latency=o_mem["latency"] + t_ret,
                                              retrieval_latency=t_ret, ingest_latency=t_ing,
                                              needle_retrieved=needle_found, memories=len(layer.store))))
            cs = [c for c in cells if c["length"] == L and c["depth"] == d]
            print(f"L={L:6d} depth={d:.2f}: full acc={sum(c['full']['correct'] for c in cs) / len(cs):.2f} "
                  f"mem acc={sum(c['memory']['correct'] for c in cs) / len(cs):.2f} "
                  f"tokens full={cs[0]['full']['prompt_tokens']} mem={cs[0]['memory']['prompt_tokens']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # aggregate
    summary = {}
    for L in lengths:
        cs = [c for c in cells if c["length"] == L]
        summary[str(L)] = {
            m: dict(accuracy=sum(c[m]["correct"] for c in cs) / len(cs),
                    hallucination_rate=sum(c[m]["halluc"] for c in cs) / len(cs),
                    avg_prompt_tokens=sum(c[m]["prompt_tokens"] for c in cs) / len(cs),
                    avg_latency_s=sum(c[m]["latency"] for c in cs) / len(cs))
            for m in ("full", "memory")}
        summary[str(L)]["memory"]["needle_recall"] = sum(c["memory"]["needle_retrieved"] for c in cs) / len(cs)
    heat = {m: {str(L): {str(d): sum(c[m]["correct"] for c in cells if c["length"] == L and c["depth"] == d) /
                         max(1, sum(1 for c in cells if c["length"] == L and c["depth"] == d))
                         for d in depths} for L in lengths} for m in ("full", "memory")}
    out = dict(meta=dict(model=args.model, device=llm.device, lengths=lengths, depths=depths,
                         needles=args.needles, tok_per_pair=tok_per_pair, elapsed_s=time.time() - t0),
               summary=summary, heatmap=heat, cells=cells)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
