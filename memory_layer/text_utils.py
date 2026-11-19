import re

STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from as is are was were be been being
am do does did doing have has had having i me my mine myself you your yours we us our ours they them
their he him his she her hers it its this that these those there here what which who whom whose when
where why how can could would should will shall may might must not no yes so very just also too only
about into over under again further than more most some any each both few all own same other such
up down out off please thanks thank hey hi hello okay ok sure got get let lets like really now today
tomorrow yesterday still ever never always often sometimes already yet user users assistant
""".split())

# words that are capitalised only because they start a sentence
SENTENCE_STARTERS = set("""
i my the a an this that these those what which who where when why how can could would should will
please also thanks thank yes no ok okay sure hey hi hello so and but or if just let actually honestly
maybe anyway by for in on at to it its is are do does did have has had we you they he she there here
last next every one two three today tomorrow yesterday after before since during while sometimes
good great nice cool wow oh well sounds glad happy sorry got thank user assistant question answer
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = text.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def light_stem(tok: str) -> str:
    if len(tok) > 4:
        if tok.endswith("ies"):
            return tok[:-3] + "y"
        for suf in ("ing", "ed", "es", "s"):
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                return tok[: -len(suf)]
    return tok


def tokenize(text: str, stem: bool = True, drop_stop: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower().replace("'", ""))
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    if stem:
        toks = [light_stem(t) for t in toks]
    return [t for t in toks if t]


def content_tokens(text: str) -> set[str]:
    return set(tokenize(text, stem=True, drop_stop=True))


_PERSON_RULES_USER = [
    (re.compile(r"\bI am\b"), "the user is"), (re.compile(r"\bI'm\b"), "the user is"),
    (re.compile(r"\bI've\b"), "the user has"), (re.compile(r"\bI'd\b"), "the user would"),
    (re.compile(r"\bI'll\b"), "the user will"), (re.compile(r"\bI\b"), "the user"),
    (re.compile(r"\bmy\b", re.I), "the user's"), (re.compile(r"\bmine\b", re.I), "the user's"),
    (re.compile(r"\bmyself\b", re.I), "themselves"), (re.compile(r"\bme\b", re.I), "the user"),
    (re.compile(r"\bwe\b"), "the user and others"), (re.compile(r"\bour\b", re.I), "the user's"),
]
_PERSON_RULES_ASSISTANT = [
    (re.compile(r"\byou're\b", re.I), "the user is"), (re.compile(r"\byou've\b", re.I), "the user has"),
    (re.compile(r"\byour\b", re.I), "the user's"), (re.compile(r"\byours\b", re.I), "the user's"),
    (re.compile(r"\byou\b", re.I), "the user"),
]


def to_third_person(text: str, role: str) -> str:
    """Memory normalisation: rewrite a conversational turn so that the speaker is referred to
    as 'the user'. Questions are later asked about 'the user', so this aligns the lexical and
    semantic form of memories with queries (large gains for BM25, embeddings and re-ranking)."""
    rules = _PERSON_RULES_USER if role == "user" else _PERSON_RULES_ASSISTANT
    out = text
    for pat, rep in rules:
        out = pat.sub(rep, out)
    # fix simple verb agreement after "the user": "the user live" -> "the user lives" is not
    # attempted; embeddings and BM25 are robust to this, and the LLM reads it fine.
    out = out.replace("the user the user", "the user")
    return out[0].upper() + out[1:] if out else out


def approx_tokens(text: str) -> int:
    """Cheap token estimate used when no tokenizer is supplied (~1.3 tokens / word)."""
    return int(len(text.split()) * 1.3) + 1
