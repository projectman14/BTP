"""Entity graph over memories: memory nodes <-> entity nodes.

Entities are extracted with lightweight rules (capitalised names, product names,
dates, numbers). Retrieval uses the graph to expand candidates by 1 hop through
shared entities, which is what multi-hop questions ("the cat adopted by the
user's sister") need.
"""
import re
import networkx as nx

from .text_utils import SENTENCE_STARTERS

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_DATE_RE = re.compile(rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})(?:\s+\d{{4}})?|(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|\b20\d\d\b)", re.I)
_CAP_RE = re.compile(r"\b(?:Dr\.?\s+)?[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,2}\b")
_NUM_RE = re.compile(r"\b\d{3,}\b")
# a sentence-initial capitalised word is kept as an entity only if it behaves like a named subject
_SUBJECT_VERB_RE = re.compile(
    r"^(?:is|was|has|had|got|adopted|moved|lives|works|set|said|told|visited|bought|joined|"
    r"called|named|just|also|and|recently|finally|will|won|lost|started|loves|likes|plays|runs|"
    r"gave|sent|asked|invited|came|went|owns|drives|teaches|studies|married|turned)\b", re.I)


def extract_entities(text: str) -> list[str]:
    ents: set[str] = set()
    for m in _DATE_RE.finditer(text):
        ents.add(m.group(0).lower().strip())
    for m in _CAP_RE.finditer(text):
        span = m.group(0)
        words = span.split()
        # drop leading sentence-starter words ("Yes Priya" -> "Priya")
        while words and words[0].lower().strip(".") in SENTENCE_STARTERS:
            words = words[1:]
        if not words:
            continue
        sentence_initial = m.start() == 0 or text[max(0, m.start() - 2):m.start()].strip() in {".", "!", "?"}
        if sentence_initial and len(words) == 1:
            rest = text[m.end():].lstrip()
            if not _SUBJECT_VERB_RE.match(rest):
                continue
        ent = " ".join(words).lower().strip(".")
        if len(ent) >= 3:
            ents.add(ent)
    for m in _NUM_RE.finditer(text):
        ents.add(m.group(0))
    return sorted(ents)


class MemoryGraph:
    def __init__(self, max_entity_degree: int = 25):
        self.g = nx.Graph()
        self.max_entity_degree = max_entity_degree

    def add_memory(self, mid: int, entities: list[str]):
        self.g.add_node(("m", mid))
        for e in entities:
            self.g.add_node(("e", e))
            self.g.add_edge(("m", mid), ("e", e))

    def entities_of(self, mid: int) -> list[str]:
        return [n[1] for n in self.g.neighbors(("m", mid))] if ("m", mid) in self.g else []

    def memories_with_entity(self, ent: str) -> list[int]:
        node = ("e", ent)
        if node not in self.g:
            return []
        if self.g.degree(node) > self.max_entity_degree:   # hub entity, uninformative
            return []
        return [n[1] for n in self.g.neighbors(node)]

    def expand(self, seed_memories: list[int], query_entities: list[str], hops: int = 1) -> dict[int, int]:
        """Return {memory_id: number_of_shared_entity_links} reachable from seeds / query entities."""
        found: dict[int, int] = {}
        frontier = set(seed_memories)
        ents = set(query_entities)
        for _ in range(hops):
            for mid in list(frontier):
                ents.update(self.entities_of(mid))
            new_frontier = set()
            for e in ents:
                for mid in self.memories_with_entity(e):
                    if mid in seed_memories:
                        continue
                    found[mid] = found.get(mid, 0) + 1
                    new_frontier.add(mid)
            frontier = new_frontier
        return found

    def shared_entity_links(self, mids: list[int]) -> dict[int, int]:
        """How many *other* memories in `mids` each memory shares an entity with (coherence bonus)."""
        ent_sets = {m: set(self.entities_of(m)) for m in mids}
        out = {}
        for m in mids:
            out[m] = sum(1 for o in mids if o != m and ent_sets[m] & ent_sets[o])
        return out
