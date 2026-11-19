"""Rule-based query expansion for the sparse (BM25) channel.

Adds domain-neutral synonyms so that a paraphrased question ("Which city is the
user based in?") can lexically match the memory ("I live in Jaipur").
"""
import re

SYNONYMS = {
    "live": ["living", "lives", "based", "home", "stay", "staying", "moved", "city", "town", "reside"],
    "city": ["live", "town", "based", "moved", "home", "place"],
    "based": ["live", "living", "city"],
    "moved": ["move", "moving", "relocated", "shifted", "living", "live", "city"],
    "currently": ["now", "these days", "moved", "new"],
    "before": ["earlier", "previously", "used to", "old", "moved"],
    "profession": ["work", "job", "working", "works", "engineer", "career", "role"],
    "job": ["work", "working", "works", "profession", "role", "company", "joined"],
    "work": ["job", "working", "works", "company", "joined", "office"],
    "company": ["work", "working", "joined", "job", "office", "employer"],
    "pet": ["dog", "cat", "puppy", "kitten", "named"],
    "dog": ["pet", "puppy", "named"],
    "cat": ["pet", "kitten", "named", "adopted"],
    "name": ["named", "called", "is"],
    "named": ["name", "called"],
    "favourite": ["favorite", "love", "best", "prefer", "like"],
    "favorite": ["favourite", "love", "best", "prefer", "like"],
    "food": ["dish", "eat", "eating", "cuisine", "meal", "favourite"],
    "car": ["drive", "driving", "vehicle", "bought"],
    "drive": ["car", "driving", "vehicle"],
    "sport": ["play", "playing", "game", "weekend"],
    "play": ["sport", "playing", "game"],
    "allergic": ["allergy", "avoid", "reaction"],
    "allergy": ["allergic", "avoid"],
    "birthday": ["born", "birth", "celebrate", "turning"],
    "language": ["learning", "speak", "duolingo", "learn", "studying"],
    "learning": ["learn", "studying", "language", "course"],
    "doctor": ["dr", "physician", "clinic", "told"],
    "laptop": ["bought", "computer", "notebook", "machine"],
    "phone": ["mobile", "smartphone", "use", "using"],
    "book": ["reading", "read", "novel", "author"],
    "reading": ["book", "read", "novel"],
    "gym": ["workout", "exercise", "fitness", "train"],
    "team": ["support", "fan", "cricket", "football", "club"],
    "support": ["team", "fan", "cheer"],
    "deadline": ["due", "submit", "submission", "date", "set"],
    "wedding": ["marriage", "married", "getting married", "engaged"],
    "sister": ["sibling", "sis"],
    "brother": ["sibling", "bro"],
    "friend": ["buddy", "mate", "college"],
    "manager": ["boss", "lead", "supervisor"],
    "hobby": ["weekend", "free time", "enjoy", "love", "play"],
    "study": ["studying", "college", "degree", "university", "course"],
    "visit": ["visited", "trip", "travel", "travelled", "went"],
    "trip": ["visit", "travel", "travelled", "went", "holiday", "vacation"],
    "bought": ["buy", "purchased", "got", "new"],
    "adopted": ["adopt", "got", "new", "pet"],
}

_WORD = re.compile(r"[a-zA-Z]+")


def expand_query(query: str, max_added: int = 12) -> str:
    """Return query + synonyms of its content words (used for BM25 only)."""
    words = [w.lower() for w in _WORD.findall(query)]
    added: list[str] = []
    for w in words:
        for syn in SYNONYMS.get(w, []):
            if syn not in words and syn not in added:
                added.append(syn)
        if len(added) >= max_added:
            break
    return query + " " + " ".join(added[:max_added]) if added else query
