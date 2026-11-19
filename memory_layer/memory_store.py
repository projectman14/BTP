from dataclasses import dataclass, field


@dataclass
class Memory:
    id: int
    text: str                 # normalised (third-person) text used for indexing and injection
    role: str                 # "user" | "assistant"
    session: int
    turn: int
    raw_text: str = ""        # original wording of the turn
    timestamp: str = ""
    importance: float = 0.5
    access_count: int = 0
    entities: list[str] = field(default_factory=list)
    superseded_by: int | None = None

    def label(self) -> str:
        return f"[Session {self.session}{', ' + self.timestamp if self.timestamp else ''}] {self.role.capitalize()}"


class MemoryStore:
    """In-memory table of Memory records (the 'metadata' side of the vector DB)."""

    def __init__(self):
        self._items: dict[int, Memory] = {}
        self._next = 0

    def new_id(self) -> int:
        i = self._next
        self._next += 1
        return i

    def add(self, m: Memory):
        self._items[m.id] = m

    def get(self, mid: int) -> Memory:
        return self._items[mid]

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def remove(self, mid: int):
        self._items.pop(mid, None)

    def all_ids(self) -> list[int]:
        return list(self._items.keys())
