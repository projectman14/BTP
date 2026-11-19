"""Write-time importance scoring and read-time recency / access scoring."""
import math
import re

FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|my|me|mine|myself|we|our|the user|the user's)\b", re.I)
NUMBER_OR_DATE = re.compile(
    r"\b(\d+|january|february|march|april|may|june|july|august|september|october|november|december|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
CAPITALISED = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-zA-Z]{2,}\b")
# generic "personal fact" cues (not tied to any particular dataset)
FACT_CUES = re.compile(
    r"\b(live|lives|living|moved|moving|work|works|working|job|joined|company|named|called|name is|"
    r"favou?rite|allergic|birthday|born|bought|own|drive|study|studying|learning|married|wedding|"
    r"deadline|appointment|doctor|sister|brother|cousin|friend|manager|boss|wife|husband|daughter|son|"
    r"prefer|hate|love|usually|always|every)\b", re.I)


def importance_score(text: str, role: str) -> float:
    """Heuristic importance in [0,1]: personal, entity-bearing, specific statements score high;
    small talk and generic assistant replies score low."""
    s = 0.20 if role == "user" else 0.05
    if FIRST_PERSON.search(text):
        s += 0.20
    if CAPITALISED.search(text):
        s += 0.25
    if NUMBER_OR_DATE.search(text):
        s += 0.10
    if FACT_CUES.search(text):
        s += 0.15
    if len(text.split()) >= 6:
        s += 0.10
    if text.strip().endswith("?"):
        s -= 0.15
    return max(0.0, min(1.0, s))


def recency_score(current_session: int, memory_session: int, half_life: float) -> float:
    gap = max(0, current_session - memory_session)
    return 0.5 ** (gap / max(half_life, 1e-6))


def access_score(access_count: int, max_access: int) -> float:
    if max_access <= 0:
        return 0.0
    return math.log1p(access_count) / math.log1p(max_access)
