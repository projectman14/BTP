from dataclasses import dataclass, replace


@dataclass
class MemoryConfig:
    """All tunable knobs of the memory layer. Flags allow ablations."""

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval sizes
    top_k: int = 5                # memories finally injected into the prompt
    vector_candidates: int = 20   # first-stage dense candidates
    bm25_candidates: int = 20     # first-stage sparse candidates
    rrf_k: int = 60               # reciprocal rank fusion constant
    w_dense: float = 1.0          # weight of the dense channel in weighted RRF
    w_sparse: float = 0.2         # weight of the BM25 channel in weighted RRF (tuned on dev split)
    rerank_pool: int = 40         # max candidates passed to the cross-encoder

    # Stage switches (for ablation)
    use_vector: bool = True
    use_bm25: bool = True
    use_query_expansion: bool = True
    use_graph: bool = True
    use_rerank: bool = True
    use_scoring: bool = True
    use_conflict_resolution: bool = True

    # Graph expansion: memories reachable through a shared entity from the top-m fused candidates
    # inherit `graph_propagation` x the seed's fused score (one-step relevance propagation)
    graph_expand_from: int = 3
    graph_max_expansions: int = 4
    graph_propagation: float = 0.85
    graph_max_entity_degree: int = 25   # ignore hub entities connected to too many memories

    # Relevance = rerank_blend * sigmoid(cross-encoder) + (1 - rerank_blend) * normalised RRF score
    rerank_blend: float = 0.5           # tuned on dev split

    # Memory scoring: final = w_rel * rel * (1 + w_rec*recency + w_imp*importance + w_acc*access + w_graph*graph_bonus)
    # (recency weight 0 for ranking was best on the dev split: recency is instead applied through the
    #  conflict-aware update rule below, so old facts stay retrievable for "before" questions)
    w_relevance: float = 1.0
    w_recency: float = 0.0
    w_importance: float = 0.10
    w_access: float = 0.02
    w_graph: float = 0.10
    recency_half_life: float = 6.0      # in sessions
    question_penalty: float = 0.7       # multiplier for memories that are questions/requests, not statements

    # Conflict-aware recency (memory updates): two statements about the user with cosine >= threshold,
    # where the newer one carries an update cue, are treated as old/new versions of one fact
    conflict_sim_threshold: float = 0.45
    conflict_demotion: float = 0.95     # old version is placed just below the new one

    # Storage policy
    store_assistant_turns: bool = True
    min_importance_to_store: float = 0.0  # >0 enables write-time pruning

    # Context injection
    context_token_budget: int = 512

    def with_(self, **kw):
        return replace(self, **kw)


# Named presets used by the benchmarks (ablation ladder)
PRESETS = {
    # Phase-1 proposal: plain embedding + vector DB + top-K (basic RAG-style memory)
    "rag_vector": dict(use_bm25=False, use_query_expansion=False, use_graph=False,
                       use_rerank=False, use_scoring=False, use_conflict_resolution=False),
    "bm25_only": dict(use_vector=False, use_query_expansion=False, use_graph=False,
                      use_rerank=False, use_scoring=False, use_conflict_resolution=False),
    "hybrid": dict(use_query_expansion=False, use_graph=False, use_rerank=False,
                   use_scoring=False, use_conflict_resolution=False),
    "hybrid_qe": dict(use_graph=False, use_rerank=False, use_scoring=False,
                      use_conflict_resolution=False),
    "hybrid_qe_rerank": dict(use_graph=False, use_scoring=False, use_conflict_resolution=False),
    "hybrid_qe_rerank_graph": dict(use_scoring=False, use_conflict_resolution=False),
    "memory_layer": dict(),
}
