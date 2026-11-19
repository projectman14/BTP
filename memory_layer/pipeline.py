"""End-to-end memory layer: ingest conversation turns, retrieve, inject, answer, verify."""
import re
import time
import numpy as np

from .config import MemoryConfig
from .embeddings import Embedder, Reranker
from .vector_store import VectorStore
from .keyword_index import KeywordIndex
from .memory_store import MemoryStore, Memory
from .graph_memory import MemoryGraph, extract_entities
from .scoring import importance_score
from .retriever import HybridRetriever, Retrieved, is_about_user, THIRD_PARTY
from .text_utils import content_tokens, approx_tokens, to_third_person

_ABOUT_USER_Q = re.compile(r"\bthe user('s)?\b", re.I)


def about_user_question(question: str) -> bool:
    """True when the question is about the user themself (not about the user's sister, manager, ...)."""
    return bool(_ABOUT_USER_Q.search(question)) and not THIRD_PARTY.search(question)

SYSTEM_PROMPT = (
    "You are a personal assistant with notes from earlier conversations with the user. "
    "Answer the question using ONLY the memory notes. Notes are listed from oldest to newest; "
    "if two notes conflict, trust the most recent one. Notes about other people (a colleague, "
    "neighbour, friend or relative) are not about the user. "
    "If the notes do not contain the answer, reply exactly: I don't know. "
    "Answer with a short phrase, no explanation."
)

ABSTAIN_RE = re.compile(
    r"(don'?t know|do not know|not know|no information|not mentioned|not provided|not specified|"
    r"not stated|unknown|cannot determine|can'?t determine|no notes|not in the notes|not available|"
    r"i'm not sure|i am not sure|unable to)", re.I)


def is_abstention(answer: str) -> bool:
    return bool(ABSTAIN_RE.search(answer))


class MemoryLayer:
    def __init__(self, cfg: MemoryConfig | None = None, embedder: Embedder | None = None,
                 reranker: Reranker | None = None, device: str | None = None):
        self.cfg = cfg or MemoryConfig()
        self.embedder = embedder or Embedder(self.cfg.embedding_model, device=device)
        self.reranker = reranker if reranker is not None else (
            Reranker(self.cfg.reranker_model, device=device) if self.cfg.use_rerank else None)
        self.store = MemoryStore()
        self.vectors = VectorStore(self.embedder.dim)
        self.keywords = KeywordIndex()
        self.graph = MemoryGraph(self.cfg.graph_max_entity_degree)
        self.retriever = HybridRetriever(self.cfg, self.store, self.vectors, self.keywords,
                                         self.graph, self.embedder, self.reranker)
        self.n_seen = 0          # turns offered to the layer
        self.n_dropped = 0       # turns rejected by write-time pruning

    # -- write path ----------------------------------------------------------
    def ingest(self, text: str, role: str, session: int, turn: int, timestamp: str = "") -> int | None:
        self.n_seen += 1
        if role == "assistant" and not self.cfg.store_assistant_turns:
            self.n_dropped += 1
            return None
        imp = importance_score(text, role)
        if imp < self.cfg.min_importance_to_store:
            self.n_dropped += 1
            return None
        norm = to_third_person(text, role)
        mid = self.store.new_id()
        ents = extract_entities(norm)
        mem = Memory(id=mid, text=norm, raw_text=text, role=role, session=session, turn=turn,
                     timestamp=timestamp, importance=imp, entities=ents)
        self.store.add(mem)
        vec = self.embedder.encode_one(norm)
        self.vectors.add([mid], vec.reshape(1, -1))
        self.retriever.register_embedding(mid, vec)
        self.keywords.add(mid, norm)
        self.graph.add_memory(mid, ents)
        return mid

    def ingest_session(self, turns: list[dict], session: int, timestamp: str = "") -> list[int | None]:
        """turns: [{role, text}] in order. Embeds in one batch for speed."""
        texts = [to_third_person(t["text"], t["role"]) for t in turns]
        self.embedder.encode(texts)   # warm the cache in one batch
        return [self.ingest(t["text"], t["role"], session, i, timestamp) for i, t in enumerate(turns)]

    # -- read path -----------------------------------------------------------
    def retrieve(self, query: str, current_session: int, top_k: int | None = None) -> list[Retrieved]:
        return self.retriever.retrieve(query, current_session, top_k)

    def build_context(self, retrieved: list[Retrieved], token_counter=None) -> str:
        """Chronological, token-budgeted rendering of the selected memories."""
        count = token_counter or approx_tokens
        mems = sorted(((self.store.get(r.id), r) for r in retrieved), key=lambda x: (x[0].session, x[0].turn))
        lines, used = [], 0
        for m, r in mems:
            tag = " (older note)" if r.outdated else ""
            line = f"{m.label()}{tag}: {m.text}"
            n = count(line)
            if used + n > self.cfg.context_token_budget and lines:
                break
            lines.append(line)
            used += n
        return "\n".join(lines) if lines else "(no relevant notes)"

    def make_prompt(self, question: str, context: str) -> str:
        return f"Memory notes:\n{context}\n\nQuestion: {question}"

    # -- grounding gate ------------------------------------------------------
    def verify(self, answer: str, retrieved: list[Retrieved], question: str = "",
               min_overlap: float = 0.5) -> tuple[bool, str]:
        """Evidence-validity check for a generated answer. Returns (supported, reason).

        An answer is rejected when
          (a) no injected note covers most of its content words (not attributable),
          (b) the only notes that cover it are questions/requests, not statements (no fact), or
          (c) the question asks about the user but the supporting note is about a third party
              (subject mismatch, e.g. the neighbour's dog offered as the user's dog).
        Abstentions are always accepted."""
        if is_abstention(answer):
            return True, "abstained"
        a = content_tokens(answer)
        if not a:
            return True, "empty"
        supporting = []
        for r in retrieved:
            m = self.store.get(r.id)
            if len(a & content_tokens(m.text)) / len(a) >= min_overlap:
                supporting.append(m)
        if not supporting:
            return False, "not_attributable"
        statements = [m for m in supporting if not m.text.strip().endswith("?") and not m.role == "assistant"]
        if not statements:
            return False, "no_fact"
        if question and about_user_question(question):
            if not any(is_about_user(m) for m in statements):
                return False, "subject_mismatch"
        return True, "supported"

    def answer(self, question: str, llm, current_session: int, apply_gate: bool = True) -> dict:
        t0 = time.time()
        retrieved = self.retrieve(question, current_session)
        t_ret = time.time() - t0
        context = self.build_context(retrieved, llm.count_tokens)
        res = llm.chat(SYSTEM_PROMPT, self.make_prompt(question, context))
        raw = res["answer"]
        abst = is_abstention(raw)
        supported, reason = self.verify(raw, retrieved, question)
        final = raw
        if apply_gate and not supported:
            final = "I don't know (the answer is not supported by my memory notes)."
        return {"answer": final, "raw_answer": raw, "retrieved": [r.id for r in retrieved],
                "outdated": [r.id for r in retrieved if r.outdated], "context": context,
                "supported": supported, "gate_reason": reason, "abstained": abst,
                "prompt_tokens": res["prompt_tokens"], "latency": res["latency"] + t_ret,
                "retrieval_latency": t_ret}

    # -- maintenance ---------------------------------------------------------
    def prune(self, min_importance: float) -> int:
        """Forget low-importance, never-accessed memories. Returns number removed.
        (The FAISS index keeps the vector but search results are filtered by the store.)"""
        victims = [m.id for m in self.store if m.importance < min_importance and m.access_count == 0]
        for mid in victims:
            self.store.remove(mid)
        if victims:
            self._rebuild_indexes()
        return len(victims)

    def _rebuild_indexes(self):
        self.vectors = VectorStore(self.embedder.dim)
        self.keywords = KeywordIndex()
        self.graph = MemoryGraph(self.cfg.graph_max_entity_degree)
        ids = self.store.all_ids()
        if ids:
            vecs = np.stack([self.retriever._emb_cache[m] for m in ids])
            self.vectors.add(ids, vecs)
        for m in ids:
            mem = self.store.get(m)
            self.keywords.add(m, mem.text)
            self.graph.add_memory(m, mem.entities)
        self.retriever.vectors, self.retriever.keywords, self.retriever.graph = \
            self.vectors, self.keywords, self.graph

    def stats(self) -> dict:
        return {"memories": len(self.store), "seen": self.n_seen, "dropped": self.n_dropped,
                "entities": sum(1 for n in self.graph.g.nodes if n[0] == "e"),
                "graph_edges": self.graph.g.number_of_edges()}
