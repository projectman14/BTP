"""Shared evaluation helpers (answer scoring, metrics)."""
import json
import re
import string
from collections import Counter

from memory_layer.pipeline import is_abstention

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def is_correct(pred: str, gold: str, answerable: bool) -> bool:
    if not answerable:
        return is_abstention(pred)
    if is_abstention(pred):
        return False
    p, g = normalize_answer(pred), normalize_answer(gold)
    return g in p or token_f1(pred, gold) >= 0.6


def is_hallucination(pred: str, gold: str, answerable: bool) -> bool:
    """A hallucination is a confident (non-abstaining) answer that is wrong, or any confident
    answer to an unanswerable question."""
    if is_abstention(pred):
        return False
    if not answerable:
        return True
    return not is_correct(pred, gold, True)


def load_dataset(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarise(records: list[dict]) -> dict:
    """records: dicts with keys correct, halluc, abstained, answerable, type, prompt_tokens, latency,
    and optionally recall_hit (bool) and rr (float)."""
    n = len(records)
    ans = [r for r in records if r["answerable"]]
    unans = [r for r in records if not r["answerable"]]
    out = {
        "n": n,
        "accuracy": sum(r["correct"] for r in records) / n,
        "accuracy_answerable": sum(r["correct"] for r in ans) / max(1, len(ans)),
        "abstain_rate_unanswerable": sum(r["abstained"] for r in unans) / max(1, len(unans)),
        "hallucination_rate": sum(r["halluc"] for r in records) / n,
        "hallucination_rate_answerable": sum(r["halluc"] for r in ans) / max(1, len(ans)),
        "avg_prompt_tokens": sum(r["prompt_tokens"] for r in records) / n,
        "avg_latency_s": sum(r["latency"] for r in records) / n,
        "by_type": {},
    }
    if any("recall_hit" in r for r in ans):
        out["recall_at_k"] = sum(r.get("recall_hit", False) for r in ans) / max(1, len(ans))
        out["mrr"] = sum(r.get("rr", 0.0) for r in ans) / max(1, len(ans))
    for t in sorted({r["type"] for r in records}):
        rs = [r for r in records if r["type"] == t]
        out["by_type"][t] = {"n": len(rs), "accuracy": sum(r["correct"] for r in rs) / len(rs),
                             "hallucination_rate": sum(r["halluc"] for r in rs) / len(rs)}
    return out


def prf(truth: list[bool], pred: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(truth, pred) if t and p)
    fp = sum(1 for t, p in zip(truth, pred) if (not t) and p)
    fn = sum(1 for t, p in zip(truth, pred) if t and (not p))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "positives": sum(truth), "n": len(truth)}


def format_history(sessions: list[dict], turns_limit: int | None = None) -> str:
    """Render the raw conversation as a transcript (used by the full-context / sliding baselines)."""
    lines = []
    for s in sessions:
        lines.append(f"[Session {s['session']}, {s['date']}]")
        for t in s["turns"]:
            lines.append(f"{t['role'].capitalize()}: {t['text']}")
    if turns_limit is not None:
        # keep the last `turns_limit` dialogue lines (session headers excluded from the count)
        dialogue = [l for l in lines if not l.startswith("[Session")]
        return "\n".join(dialogue[-turns_limit:])
    return "\n".join(lines)
