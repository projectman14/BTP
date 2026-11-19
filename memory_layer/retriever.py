"""Multi-stage hybrid retriever.

  dense (FAISS)  +  sparse (BM25, expanded query)
        |                  |
        +---- RRF fusion ---+
                 |
        entity-graph expansion (1 hop)
                 |
        cross-encoder re-ranking
                 |
        memory scoring: relevance + recency + importance + access + graph coherence
                 |
        conflict-aware recency (newer memory wins over near-duplicate older one)
                 |
              top-k
"""
from dataclasses import dataclass, field
import math
import numpy as np

from .config import MemoryConfig
from .scoring import recency_score, access_score
from .query_expansion import expand_query
from .graph_memory import extract_entities
import re

# a memory is a statement *about the user* if it is a user turn that mentions the user and is not
# about a third party (relational nouns / "X from the user's office")
THIRD_PARTY = re.compile(
    r"\b(colleague|neighbou?r|roommate|room-mate|aunt|uncle|friend|sister|brother|cousin|manager|boss|"
    r"teammate|team-mate|from the user's (office|team|college)|wife|husband|daughter|son|mother|father|"
    r"mom|dad)\b", re.I)
UPDATE_CUE = re.compile(
    r"\b(moved|moving|relocated|shifted|switched|switch|upgraded|replaced|changed|new|now|update|"
    r"these days|recently|leaving|left|just joined|no longer|anymore|instead)\b", re.I)


def is_about_user(mem) -> bool:
    """A user turn is about the user when it does not mention a third party and either names the
    user explicitly or has an implicit first-person subject ("Got a new phone, it's a OnePlus 12")
    rather than a named person as subject ("Sneha is learning Japanese")."""
    if mem.role != "user" or THIRD_PARTY.search(mem.text):
        return False
    text = mem.text.lower()
    if "the user" in text:
        return True
    first = text.split()[0].strip(",.;:'\"") if text.split() else ""
    return not any(e.split()[0] == first for e in mem.entities)


@dataclass
class Retrieved:
    id: int
    score: float
    outdated: bool = False
    components: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, cfg: MemoryConfig, store, vectors, keywords, graph, embedder, reranker=None):
        self.cfg = cfg
        self.store = store
        self.vectors = vectors
        self.keywords = keywords
        self.graph = graph
        self.embedder = embedder
        self.reranker = reranker
        self._emb_cache: dict[int, np.ndarray] = {}

    # -- helpers -------------------------------------------------------------
    def register_embedding(self, mid: int, vec: np.ndarray):
        self._emb_cache[mid] = vec

    @staticmethod
    def _rrf(rankings: list[tuple[float, list[int]]], k: int) -> dict[int, float]:
        """Weighted reciprocal rank fusion: sum_c w_c / (k + rank_c)."""
        fused: dict[int, float] = {}
        for w, ranking in rankings:
            for r, mid in enumerate(ranking):
                fused[mid] = fused.get(mid, 0.0) + w / (k + r + 1)
        return fused

    # -- main ----------------------------------------------------------------
    def retrieve(self, query: str, current_session: int, top_k: int | None = None) -> list[Retrieved]:
        cfg = self.cfg
        top_k = top_k or cfg.top_k
        comp: dict[int, dict] = {}

        # 1. first-stage candidates
        rankings: list[tuple[float, list[int]]] = []
        if cfg.use_vector and cfg.vector_candidates > 0:
            q_vec = self.embedder.encode_one(query)
            dense = self.vectors.search(q_vec, cfg.vector_candidates)
            rankings.append((cfg.w_dense, [m for m, _ in dense]))
            for r, (m, s) in enumerate(dense):
                comp.setdefault(m, {})["dense_rank"] = r + 1
                comp[m]["cosine"] = s
        if cfg.use_bm25 and cfg.bm25_candidates > 0:
            q_text = expand_query(query) if cfg.use_query_expansion else query
            sparse = self.keywords.search(q_text, cfg.bm25_candidates)
            w = cfg.w_sparse if cfg.use_vector else 1.0
            rankings.append((w, [m for m, _ in sparse]))
            for r, (m, s) in enumerate(sparse):
                comp.setdefault(m, {})["bm25_rank"] = r + 1
                comp[m]["bm25"] = s
        if not rankings:
            return []

        # 2. weighted reciprocal rank fusion
        fused = self._rrf(rankings, cfg.rrf_k)
        ordered = sorted(fused, key=fused.get, reverse=True)

        # 3. entity-graph expansion with one-step relevance propagation: a memory that shares a
        #    (non-hub) entity with a top seed inherits most of that seed's score, which is how the
        #    second hop of a multi-hop question ("Dev started at Zoho" for "the user's brother Dev")
        #    enters the candidate pool even though it never mentions the question's words.
        if cfg.use_graph and ordered:
            seeds = ordered[: cfg.graph_expand_from]
            q_ents = extract_entities(query)
            proposals: dict[int, float] = {}
            for seed in seeds:
                for mid, links in self.graph.expand([seed], q_ents, hops=1).items():
                    if mid in seeds:
                        continue
                    sc = fused[seed] * cfg.graph_propagation
                    if sc > proposals.get(mid, 0.0):
                        proposals[mid] = sc
            for mid, sc in sorted(proposals.items(), key=lambda x: -x[1])[: cfg.graph_max_expansions]:
                if sc > fused.get(mid, 0.0):
                    fused[mid] = sc
                    comp.setdefault(mid, {})["via_graph"] = True
            ordered = sorted(fused, key=fused.get, reverse=True)

        candidates = ordered[: cfg.rerank_pool]
        if not candidates:
            return []

        # 4. relevance: cross-encoder blended with first-stage rank evidence, else fused score only
        mx = max(fused[m] for m in candidates)
        fused_norm = {m: fused[m] / mx for m in candidates}
        if cfg.use_rerank and self.reranker is not None:
            texts = [self.store.get(m).text for m in candidates]
            logits = self.reranker.score(query, texts)
            ce = {m: float(1.0 / (1.0 + math.exp(-l))) for m, l in zip(candidates, logits)}
            rel = {m: cfg.rerank_blend * ce[m] + (1 - cfg.rerank_blend) * fused_norm[m] for m in candidates}
            for m in candidates:
                comp.setdefault(m, {})["ce"] = ce[m]
        else:
            rel = fused_norm

        # 5. memory scoring: relevance is the gate; recency / importance / access / graph coherence
        #    modulate it multiplicatively so an irrelevant-but-recent memory can never win
        if cfg.use_scoring:
            max_access = max((self.store.get(m).access_count for m in candidates), default=0)
            links = self.graph.shared_entity_links(candidates) if cfg.use_graph else {}
            max_links = max(links.values(), default=0)
            final = {}
            for m in candidates:
                mem = self.store.get(m)
                rec = recency_score(current_session, mem.session, cfg.recency_half_life)
                acc = access_score(mem.access_count, max_access)
                gb = (links.get(m, 0) / max_links) if max_links else 0.0
                boost = 1.0 + cfg.w_recency * rec + cfg.w_importance * mem.importance + cfg.w_access * acc + cfg.w_graph * gb
                if mem.text.rstrip().endswith("?"):      # a request to the assistant is not a fact
                    boost *= cfg.question_penalty
                final[m] = cfg.w_relevance * rel[m] * boost
                comp.setdefault(m, {}).update(rel=rel[m], recency=rec, importance=mem.importance,
                                              access=acc, graph_bonus=gb)
        else:
            final = dict(rel)
            for m in candidates:
                comp.setdefault(m, {})["rel"] = rel[m]

        # 6. conflict-aware recency (memory updates): when two first-person statements about the
        #    user are semantically close and the newer one carries an update cue, the older one is
        #    tagged as outdated and demoted.
        outdated: set[int] = set()
        if cfg.use_conflict_resolution:
            top = sorted(candidates, key=final.get, reverse=True)[: max(top_k * 3, 10)]
            for i, a in enumerate(top):
                va = self._emb_cache.get(a)
                ma = self.store.get(a)
                if va is None or not is_about_user(ma):
                    continue
                for b in top[i + 1:]:
                    vb = self._emb_cache.get(b)
                    mb = self.store.get(b)
                    if vb is None or not is_about_user(mb):
                        continue
                    if float(va @ vb) < cfg.conflict_sim_threshold:
                        continue
                    older, newer = (a, b) if (ma.session, ma.turn) < (mb.session, mb.turn) else (b, a)
                    if not UPDATE_CUE.search(self.store.get(newer).text):
                        continue
                    if older not in outdated:
                        outdated.add(older)
                        # demote the old version to just below the new one, but only when the new one
                        # is itself a strong candidate for this query (both versions are relevant); the
                        # current fact is then ranked first while the old fact stays available for
                        # "before" questions. A weakly relevant newer note only tags the older one.
                        if final[newer] >= 0.6 * final[older]:
                            final[older] = min(final[older], final[newer] * cfg.conflict_demotion)

        ranked = sorted(candidates, key=final.get, reverse=True)[:top_k]
        for m in ranked:
            self.store.get(m).access_count += 1
        return [Retrieved(id=m, score=final[m], outdated=(m in outdated), components=comp.get(m, {}))
                for m in ranked]
