# Context Window Analysis for LLMs - Phase 2 presentation content

## Slide 1: Title

- Context Window Analysis for LLMs
- Phase 2: Design and Evaluation of a Persistent Memory Layer
- B.Tech Project, Department of CCE, LNMIIT Jaipur, November 2026

## Slide 2: Acknowledgement

- Sincere gratitude to Dr. Rahul Sharma, our research supervisor, for guidance, continuous support and constructive feedback throughout both phases of the project.
- Team: Lakshya Jain (23UCS633), Monish Mathur (23UCS648), Nishit Luhadia (23UCC579)

## Slide 3: Table of Content

- Recap of Phase 1 and the problem
- Objectives of Phase 2
- Literature review (new works)
- System architecture of the memory layer
- Write path and read path
- Grounding gate
- Benchmark design and experimental setup
- Results: main comparison, question types, hallucination
- Results: ablation, pruning, context-length scaling, efficiency
- Key insights, conclusion and future work

## Slide 4: Recap of Phase 1

- LLMs are stateless: every request sees only the current context window
- Long contexts are used poorly: accuracy drops with length and position ("lost in the middle")
- RAG grounds answers but stays stateless and still hallucinates
- RAGTruth detectors: recall 0.88-0.95 but precision only 0.25-0.26 (F1 0.38-0.41)
- Phase 1 proposed a memory layer: embeddings + vector DB + top-k injection
- Phase 2 builds it, adds the missing stages and measures it

## Slide 5: Objectives of Phase 2

- Implement the complete memory layer as a modular Python package (FAISS + BM25, fusion, entity graph, re-ranking, scoring, conflict-aware updates, token-budgeted injection)
- Add a grounding gate that verifies each answer against the injected memories
- Build a long-term conversational benchmark: single-hop, temporal, multi-hop and unanswerable questions, plus a NIAH context-length benchmark
- Compare five strategies with the same local LLM: no memory, full history, sliding window, vector RAG (Phase-1 design), memory layer
- Ablate every pipeline stage and study pruning; tune only on a separate dev split

## Slide 6: Slide 6

- Paper (Year) | Approach | Core idea | Strength | Weakness / our response
- Karpukhin et al. (2020) | DPR | Learned dense bi-encoder retrieval | Beats BM25 on paraphrased questions | Misses rare exact strings; we keep a lexical channel
- Guu et al. (2020); Borgeaud et al. (2022) | REALM / RETRO | Retrieval inside pre-training | Few retrieved chunks beat 25x more parameters | Needs retraining; our layer works with a frozen LLM
- Izacard & Grave (2021) | Fusion-in-Decoder | Encode passages separately, fuse in decoder | Gains up to 100 passages | Needs enc-dec model; we inject a few high-precision notes
- Khattab & Zaharia (2020); Gao et al. (2023) | ColBERT / HyDE | Late interaction; hypothetical-answer embeddings | Near cross-encoder quality; closes vocabulary gap | Large index / extra LLM call; we use rule-based query expansion
- Cormack et al. (2009) | Reciprocal Rank Fusion | Merge ranked lists by rank only | Robust to score scales | Equal weights let a weak channel inject noise; we weight channels

## Slide 7: Slide 7

- Paper (Year) | Approach | Core idea | Strength | Weakness / our response
- Park et al. (2023) | Generative Agents | Memory stream scored by recency + importance + relevance | Simple, interpretable scoring | Additive scores let recent-but-irrelevant win; we gate by relevance
- Packer et al. (2023); Zhong et al. (2024) | MemGPT / MemoryBank | LLM pages memory in/out; Ebbinghaus forgetting | Virtually unbounded memory | LLM decides storage/recall; ours is deterministic
- Chhikara et al. (2025) | Mem0 | Extract facts; add/update/delete; graph variant | Large token savings on LoCoMo | One LLM call per turn; we use a zero-cost update rule
- Edge et al. (2024) | GraphRAG | Entity graph + traversal for multi-document questions | Solves evidence spread across documents | Graph built by many LLM calls; ours is rule-based
- Wu et al. (2022); Wang et al. (2023) | Memorizing Transformers / LongMem | kNN over past activations | Very long usable context | Opaque, model-bound memory; ours stores auditable notes

## Slide 8: Slide 8

- Paper (Year) | Approach | Core idea | Strength | Weakness / our response
- Maharana et al. (2024); Wu et al. (2025) | LoCoMo / LongMemEval | Multi-session memory benchmarks | Single/multi-hop, temporal, abstention taxonomy | Need large API models; we built a controlled benchmark with gold memory ids
- Hsieh et al. (2024) | RULER | Multi-needle, tracing, aggregation tasks | Shows effective context << window | Measures the model, not how to select memories
- Shi et al. (2023); Xu et al. (2024) | Distraction / retrieval vs long context | Irrelevant context hurts; retrieval + 4k beats 32k | Motivates selective injection | No conversational memory setting; we add distractors and updates
- Niu et al. (2024); Manakul et al. (2023) | RAGTruth / SelfCheckGPT | Hallucination corpus; consistency sampling | Standard evaluation | High recall, low precision; sampling cost
- Min et al. (2023); Asai et al. (2024) | FActScore / Self-RAG | Atomic-fact attribution; reflection tokens | Verifies support of generations | External KB / fine-tuning; our gate is deterministic and free

## Slide 9: System Architecture

- Write path: turn -> third-person normalisation -> importance scoring/pruning -> entity extraction -> FAISS index, BM25 index, entity graph
- Read path: question -> weighted RRF (dense + sparse, query expansion) -> 1-hop graph expansion -> cross-encoder re-ranking -> memory scoring + conflict-aware recency -> top-k notes within a token budget -> LLM -> grounding gate -> answer or abstention

## Slide 10: Write Path: Storing Memories

- Third-person normalisation: "I live in Pune" -> "The user lives in Pune"
    - question-memory cosine 0.3 -> 0.6; cross-encoder logit -9 -> +4
- Importance score from surface cues (speaker, entities, numbers, fact vocabulary, length; questions penalised)
    - threshold 0.3 keeps 56% of turns with no recall loss
- Three indexes per memory: all-MiniLM-L6-v2 embedding in FAISS, stemmed tokens in BM25, extracted entities in a memory-entity graph (NetworkX)
- Ingesting a 288-turn history: well under one second

## Slide 11: Read Path: Multi-Stage Retrieval

- Dense top-20 (FAISS) + sparse top-20 (BM25 on an expanded query, score floor 20% of best)
- Weighted reciprocal rank fusion: w_dense = 1, w_sparse = 0.2 (tuned on dev)
- Entity-graph expansion: neighbours of the top-3 inherit 0.85 x the seed score -> second hop of multi-hop questions enters the pool
- Cross-encoder (MS MARCO MiniLM) blended 50/50 with the fused score
- Relevance-gated scoring: rel x (1 + importance + access + graph coherence); questions x 0.7
- Conflict-aware recency: older version of an updated fact tagged "(older note)" and placed just below the new one
- Top-5 notes rendered chronologically inside a 512-token budget

## Slide 12: Grounding Gate

- After generation, the answer is checked against the injected notes with three deterministic rules:
    - Attributable: a note covers most of the answer's content words
    - Factual: that note is a statement, not a question or request
    - Same subject: a question about the user is not answered from a note about a colleague, neighbour or relative
- If any check fails the answer is replaced by an abstention
- No extra model call (Phase-1 detectors needed extra generations or fine-tuning)
- Result: precision 0.93, recall 0.55, F1 0.69 vs 0.25 / 0.92 / 0.39 in Phase 1

## Slide 13: Benchmark Design

- Public benchmarks (LoCoMo, LongMemEval) need large API models and give no gold memory ids
- Our generator: 20 users x 12 sessions x 12 user-assistant pairs = 288 turns (~3.6k tokens) per user
- Per user: 10 single-hop facts, 2 updated facts, 2 multi-hop chains, 8 distractor facts about other people, 3 unanswerable questions
- Questions are paraphrased relative to the stored statements
- Test split: 20 users, 330 questions; separate 6-user dev split (different seed) for all tuning
- NIAH-style histories of 1k-16k tokens for the scaling experiment

## Slide 14: Experimental Setup

- LLM for every strategy: Qwen2.5-1.5B-Instruct, fp16, greedy decoding, RTX 3050 (4 GB)
- Embedding: all-MiniLM-L6-v2; re-ranker: ms-marco-MiniLM-L6 cross-encoder
- Strategies: no memory | full history (chunked prefill up to 16k tokens) | sliding window (40 turns) | vector RAG top-5 (Phase-1 design) | memory layer (+ gate)
- Metrics: accuracy (gold string match / token-F1 >= 0.6; abstention required for unanswerable), hallucination rate (confident wrong answer), recall@k, MRR, prompt tokens, latency
- All hyper-parameters fixed on the dev split before touching the test split

## Slide 15: Results: Main Comparison

- Strategy | Accuracy | Halluc. rate | Prompt tokens
- Baseline LLM (no memory) | 18.2% | 0.0% | 117
- Full history in context | 57.3% | 36.1% | 4,209
- Sliding window (40 turns) | 26.4% | 11.2% | 664
- Vector RAG (Phase-1 design) | 73.6% | 17.9% | 266
- Memory layer | 78.2% | 15.5% | 267
- Memory layer + grounding gate | 80.6% | 7.0% | 267

## Slide 16: Results by Question Type

- Multi-hop: 50.0% -> 80.0% (entity graph brings in the second hop)
- Temporal (before): 50.0% -> 63.3% (old fact kept, tagged as older note); full context only 13.3%
- Temporal (current): 86.7% vs 90.0%
- Single-hop: 80.6% vs 81.2%; retrieval recall > 97%, remaining errors are reading errors of the 1.5B model
- Unanswerable: gate raises abstention to 86.7%

## Slide 17: Hallucination: Phase 1 vs Phase 2

- Detector | Precision | Recall | F1
- Phase 1: Mistral 7B (RAGTruth) | 0.25 | 0.92 | 0.39
- Phase 1: Llama 2 (RAGTruth) | 0.26 | 0.95 | 0.41
- Phase 1: TinyLlama (RAGTruth) | 0.25 | 0.88 | 0.38
- Phase 2: gate on vector RAG | 0.89 | 0.42 | 0.57
- Phase 2: gate on memory layer | 0.93 | 0.55 | 0.69
- Gate flags 30 of 330 answers; 28 are real hallucinations (precision 0.93)
- Misses 23 hallucinations that copy the wrong user fact (attributable, factual, same subject)
- Hallucination rate of the complete system: 7.0% (full context 36.1%, vector RAG 17.9%)
- Zero additional model calls

## Slide 18: Retrieval Ablation

- Vector only (Phase-1 design): recall@5 90.4%
- BM25 alone is weak (43.7%): questions are paraphrases; raw fusion hurts (87.4%), query expansion repairs it (89.6%)
- Entity graph: multi-hop recall@5 47.5% -> 92.5%
- Scoring + question penalty: overall recall@5 96.3%, single-hop 98.8%
- Temporal ordering correct: 70.0% -> 80.0%

## Slide 19: Pruning and Conflict Threshold

- Importance threshold 0.3 keeps 55.5% of turns with recall@5 95.9% (all stored: 96.3%)
- Threshold 0.45 keeps 16.9% of turns at 92.6%: six-fold smaller store for < 4 points
- Above 0.5 genuine facts are dropped and recall falls quickly
- Conflict threshold: 0.3 orders temporal questions best but tags unrelated facts; dev-chosen 0.45 keeps overall recall at its maximum

## Slide 20: Context-Length Scaling (NIAH)

- Single fact at 5 depths in 1k-16k token filler + 6 distractors; 8 needles per cell, 200 cells
- Memory layer prompt stays ~267 tokens; full context grows to 16,634 (62x)
- Latency at 16k: 81.6 s (full) vs 0.70 s (memory), 117x faster
- Accuracy: full 82.5% -> 75.0%; memory 90.0% -> 72.5% (needle retrieved 95.5%)
- One needle in filler is easy; a real history with similar facts, updates and distractors is where full context collapses (57%)

## Slide 21: NIAH Heatmap: Depth x Length

- Phase 1 showed NIAH heatmaps from the literature; Phase 2 reproduces them on our own model
- Full context: accuracy varies with depth and length (lost-in-the-middle pattern)
- Memory layer: retrieval is position-independent; remaining errors are reading errors with more distractors in a 1,118-note store
- The difficulty of long-term memory is interference, not length alone

## Slide 22: Efficiency: Tokens and Latency

- 267 prompt tokens per query vs 4,209 for full context (16x fewer)
- 0.62 s vs 3.49 s per query (6x faster)
- Retrieval stages: 30-50 ms per query on GPU, ~30 ms on CPU (plain vector search 6 ms)
- Ingest 288 turns: < 1 s; 1,118 notes: ~0.25 s
- Cost is independent of history length; full-context cost grows linearly

## Slide 23: Qualitative Example

- Q: Where does the user currently live?
    - [S1] User (older note): The user live in Jaipur ...
    - [S3] User: The user moved to Pune last month, so the user is living in Pune now.
    - Answer: Pune  (gate: supported)
- Q: What is the name of the cat adopted by the user's sister?
    - [S1] User: The user's sister Priya visited the user last weekend.
    - [S2] User: Priya adopted a cat and named it Milo.   <- reached through entity "Priya"
    - Answer: Milo  (gate: supported)
- Q: What car does the user drive?
    - [S2] User: The user's colleague Rohan lives in Delhi and drives a Honda City.
    - LLM: Honda City  ->  gate: subject mismatch  ->  I don't know

## Slide 24: Key Insights

- A bigger context window is the worst memory when facts are updated and distractors exist: full context hallucinates 36.1% of the time
- Selective retrieval is the main gain (vector RAG 73.6%); the extra stages matter where vector search fails: multi-hop 50.0% -> 80.0%, temporal-before 50.0% -> 63.3%
- Verification matters as much as retrieval: the gate halves hallucination (15.5% -> 7.0%) at precision 0.93
- Half of the turns carry all of the recall; a sixth carry 96% of it
- Remaining errors are reading errors of the small model (recall@5 96.3% vs accuracy 78.2%)

## Slide 25: Conclusion

- Phase 1 diagnosed the problem and proposed a memory layer; Phase 2 built it and measured it
- Memory layer + gate: 80.6% accuracy, 7.0% hallucination, 267 tokens/query
- vs full history: 57.3% / 36.1% / 4,209 tokens; vs vector RAG: 73.6% / 17.9%
- Grounding gate: precision 0.93, F1 0.69 (Phase 1: 0.25, 0.39), no extra model calls
- Constant cost with history length; interference, not length, is the real difficulty
- A selective, verified memory is a better foundation for long-running LLM applications than a larger context window

## Slide 26: Future Work

- Write-time fact extraction with an LLM (one turn -> several atomic memories), batched off-line
- Fine-tune the cross-encoder on conversational memory notes; learn importance and subject classifiers
- Typed entity graph (sister-of, works-at, lives-in) with explicit edge retirement on updates
- Consolidation and forgetting: summaries of related notes, decay of never-accessed notes
- Evaluate on LoCoMo / LongMemEval with 7B-70B models
- Deployment: per-user isolation, encryption, right-to-forget, and a demo that shows the notes and gate decision behind every answer

## Slide 27: References (1/2)

- Vaswani et al., Attention Is All You Need, NeurIPS 2017
- Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP, NeurIPS 2020
- Gao et al., RAG for LLMs: A Survey, arXiv 2023
- Huang et al., A Survey on Hallucination in LLMs, 2024
- Liu et al., Lost in the Middle, TACL 2024
- Modarressi et al., NoLiMa, 2025
- An et al., Why Does the Effective Context Length of LLMs Fall Short?, ICLR 2025
- Kamradt, Needle In A Haystack, 2023; Nelson, NIAH Evaluation, 2025
- Karpukhin et al., Dense Passage Retrieval, EMNLP 2020
- Guu et al., REALM, ICML 2020; Borgeaud et al., RETRO, ICML 2022
- Izacard and Grave, Fusion-in-Decoder, EACL 2021
- Khattab and Zaharia, ColBERT, SIGIR 2020
- Gao et al., HyDE, ACL 2023
- Robertson and Zaragoza, BM25, FnTIR 2009
- Cormack et al., Reciprocal Rank Fusion, SIGIR 2009
- Reimers and Gurevych, Sentence-BERT, EMNLP 2019
- Nogueira and Cho, Passage Re-ranking with BERT, 2019
- Johnson et al., FAISS, IEEE TBD 2021
- Wu et al., Memorizing Transformers, ICLR 2022

## Slide 28: References (2/2)

- Wang et al., LongMem, NeurIPS 2023
- Park et al., Generative Agents, UIST 2023
- Shinn et al., Reflexion, NeurIPS 2023
- Packer et al., MemGPT, 2023
- Zhong et al., MemoryBank, AAAI 2024
- Chhikara et al., Mem0, 2025
- Zhang et al., Survey on Memory Mechanisms of LLM Agents, 2024
- Edge et al., GraphRAG, 2024
- Shi et al., LLMs Can Be Easily Distracted by Irrelevant Context, ICML 2023
- Xu et al., Retrieval meets Long Context LLMs, ICLR 2024
- Hsieh et al., RULER, COLM 2024
- Niu et al., RAGTruth, ACL 2024
- Manakul et al., SelfCheckGPT, EMNLP 2023
- Min et al., FActScore, EMNLP 2023
- Asai et al., Self-RAG, ICLR 2024
- Maharana et al., LoCoMo, ACL 2024
- Wu et al., LongMemEval, ICLR 2025
- Bai et al., LongBench, ACL 2024
- Qwen Team, Qwen2.5 Technical Report, 2024

## Slide 29: Thank You

- "LLMs don't need more context - they need verified memory."
- Code: github.com/projectman14/BTP

