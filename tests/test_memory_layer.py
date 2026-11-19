"""Quick functional tests for the memory layer (no LLM needed).

python tests/test_memory_layer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_layer import MemoryLayer, MemoryConfig
from memory_layer.text_utils import to_third_person, tokenize
from memory_layer.graph_memory import extract_entities
from memory_layer.scoring import importance_score
from memory_layer.memory_store import Memory
from memory_layer.retriever import is_about_user


def test_normalisation():
    assert to_third_person("I live in Jaipur and my dog is Bruno.", "user") == \
        "The user live in Jaipur and the user's dog is Bruno."
    assert to_third_person("I'm learning Spanish.", "user") == "The user is learning Spanish."
    assert "the user" in to_third_person("Your appointment is on Monday.", "assistant").lower()


def test_entities():
    assert extract_entities("My sister Priya adopted a cat named Milo.") == ["milo", "priya"]
    assert "congrats" not in extract_entities("Congrats on the move!")
    assert "dev" in extract_entities("Dev started working at Zoho.")
    assert any("march" in e for e in extract_entities("The deadline is 14 March."))


def test_importance():
    assert importance_score("I live in Pune.", "user") > importance_score("Can you suggest a movie?", "user")
    assert importance_score("Sure, here is a summary.", "assistant") < 0.3


def test_about_user():
    m = Memory(0, "The user's neighbour's dog is called Luna.", "user", 1, 0, entities=["luna"])
    assert not is_about_user(m)
    m = Memory(1, "Got a new phone, it's a OnePlus 12.", "user", 1, 0, entities=["oneplus 12"])
    assert is_about_user(m)
    m = Memory(2, "Sneha is learning Japanese for a trip.", "user", 1, 0, entities=["japanese", "sneha"])
    assert not is_about_user(m)


def test_pipeline():
    ml = MemoryLayer(MemoryConfig())
    ml.ingest_session([{"role": "user", "text": "I live in Jaipur and I work as a data analyst."},
                       {"role": "assistant", "text": "Nice!"},
                       {"role": "user", "text": "My sister Priya visited me last weekend."},
                       {"role": "user", "text": "Can you suggest a good movie for tonight?"}], 1, "2026-01-05")
    ml.ingest_session([{"role": "user", "text": "Priya adopted a cat and named it Milo."},
                       {"role": "user", "text": "My colleague Rohan lives in Delhi."}], 2, "2026-01-12")
    ml.ingest_session([{"role": "user", "text": "I moved to Pune last month, so I'm living in Pune now."}], 3)
    texts = lambda rs: [ml.store.get(r.id).text for r in rs]
    r = ml.retrieve("Where does the user currently live?", 4)
    assert "Pune" in texts(r)[0], texts(r)
    assert any(x.outdated and "Jaipur" in ml.store.get(x.id).text for x in r), "old city should be tagged outdated"
    r = ml.retrieve("What is the name of the cat adopted by the user's sister?", 4)
    assert any("Milo" in t for t in texts(r)[:2])
    ctx = ml.build_context(r)
    assert "[Session" in ctx
    supported, reason = ml.verify("Delhi", ml.retrieve("Where does the user live?", 4), "Where does the user live?")
    assert not supported and reason == "subject_mismatch", (supported, reason)
    supported, _ = ml.verify("Pune", ml.retrieve("Where does the user live?", 4), "Where does the user live?")
    assert supported
    removed = ml.prune(0.4)
    assert removed >= 1 and len(ml.store) < 8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all tests passed")
