from .text_utils import tokenize


class KeywordIndex:
    """BM25 index over memory texts. Rebuilt lazily after new memories are added."""

    def __init__(self):
        self.ids: list[int] = []
        self.docs: list[list[str]] = []
        self._bm25 = None
        self._dirty = False

    def add(self, mid: int, text: str):
        self.ids.append(mid)
        self.docs.append(tokenize(text))
        self._dirty = True

    def _ensure(self):
        if self._dirty or self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self.docs) if self.docs else None
            self._dirty = False

    def search(self, query: str, k: int, exclude: set[int] | None = None,
               min_ratio: float = 0.2) -> list[tuple[int, float]]:
        """Top-k BM25 hits. Hits scoring below `min_ratio` x the best score are dropped so that
        near-zero matches (shared high-frequency words) are not fed into rank fusion as if they
        were real keyword matches."""
        self._ensure()
        if self._bm25 is None or k <= 0:
            return []
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = scores.argsort()[::-1]
        floor = max(1e-6, float(scores[order[0]]) * min_ratio)
        out = []
        for i in order:
            if scores[i] < floor:
                break
            mid = self.ids[i]
            if exclude and mid in exclude:
                continue
            out.append((mid, float(scores[i])))
            if len(out) >= k:
                break
        return out
