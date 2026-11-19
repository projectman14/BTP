"""Produce the qualitative trace shown in the report (report/example_trace.txt).

python benchmarks/example_trace.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_layer import MemoryLayer, MemoryConfig, LocalLLM

SESSIONS = [
    (1, "2026-01-05", [("user", "I live in Jaipur and I work as a data analyst."),
                       ("assistant", "Nice! Jaipur is a lovely city."),
                       ("user", "My sister Priya visited me last weekend."),
                       ("assistant", "That sounds fun."),
                       ("user", "Can you suggest a good movie for tonight?"),
                       ("assistant", "Sure, how about Interstellar?")]),
    (2, "2026-01-12", [("user", "Priya adopted a cat and named it Milo."),
                       ("assistant", "Milo is a cute name!"),
                       ("user", "My colleague Rohan lives in Delhi and drives a Honda City."),
                       ("assistant", "Okay, noted.")]),
    (3, "2026-01-19", [("user", "I moved to Pune last month, so I'm living in Pune now."),
                       ("assistant", "Congrats on the move!")]),
]
QUESTIONS = ["Where does the user currently live?", "Where did the user live before moving?",
             "What is the name of the cat adopted by the user's sister?", "What car does the user drive?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="report/example_trace.txt")
    args = ap.parse_args()
    ml = MemoryLayer(MemoryConfig())
    for s, date, turns in SESSIONS:
        ml.ingest_session([{"role": r, "text": t} for r, t in turns], s, date)
    llm = LocalLLM(args.model)
    lines = [f"Stored memories: {ml.stats()['memories']}  entities: {ml.stats()['entities']}", ""]
    for m in ml.store:
        if m.role == "user":
            lines.append(f"  mem#{m.id} s{m.session} imp={m.importance:.2f} ents={m.entities}: {m.text}")
    for q in QUESTIONS:
        res = ml.answer(q, llm, current_session=4)
        lines += ["", f"Q: {q}", "  retrieved notes (chronological, injected into the prompt):"]
        lines += ["    " + l for l in res["context"].split("\n")]
        lines.append(f"  LLM raw answer : {res['raw_answer']}")
        lines.append(f"  grounding gate : {'supported' if res['supported'] else 'REJECTED'} ({res['gate_reason']})")
        lines.append(f"  final answer   : {res['answer']}")
        lines.append(f"  prompt tokens  : {res['prompt_tokens']}   latency: {res['latency']:.2f}s")
    text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
