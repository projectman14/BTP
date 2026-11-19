"""Persistent memory layer for LLMs (BTP Phase 2).

Pipeline: embedding store (FAISS) + BM25 keyword index -> reciprocal-rank
fusion -> entity-graph expansion -> cross-encoder re-ranking -> memory
scoring (relevance, recency, importance, access) -> conflict-aware
selection -> token-budgeted context injection -> grounding gate.
"""
from .config import MemoryConfig, PRESETS
from .pipeline import MemoryLayer
from .llm import LocalLLM

__all__ = ["MemoryConfig", "PRESETS", "MemoryLayer", "LocalLLM"]
