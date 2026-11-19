import numpy as np


class Embedder:
    """Sentence-transformer embedder with an in-process cache (L2-normalised vectors)."""

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        todo = [(i, t) for i, t in enumerate(texts) if t not in self._cache]
        if todo:
            vecs = self.model.encode([t for _, t in todo], batch_size=batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
            for (i, t), v in zip(todo, vecs):
                self._cache[t] = v.astype(np.float32)
        for i, t in enumerate(texts):
            out[i] = self._cache[t]
        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class Reranker:
    """Cross-encoder re-ranker: (query, memory) -> relevance logit."""

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device, max_length=256)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        s = self.model.predict([(query, t) for t in texts], batch_size=64, show_progress_bar=False)
        return np.asarray(s, dtype=np.float32)
