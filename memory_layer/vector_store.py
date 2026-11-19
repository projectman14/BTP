import numpy as np


class VectorStore:
    """Flat inner-product FAISS index over normalised embeddings (exact cosine search)."""

    def __init__(self, dim: int):
        import faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[int] = []

    def add(self, ids: list[int], vecs: np.ndarray):
        if len(ids) == 0:
            return
        self.index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        self.ids.extend(ids)

    def __len__(self):
        return len(self.ids)

    def search(self, q: np.ndarray, k: int, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        if len(self.ids) == 0 or k <= 0:
            return []
        k_eff = min(len(self.ids), k + (len(exclude) if exclude else 0))
        scores, idx = self.index.search(np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32), k_eff)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            mid = self.ids[i]
            if exclude and mid in exclude:
                continue
            out.append((mid, float(s)))
            if len(out) >= k:
                break
        return out
