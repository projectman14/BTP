"""Build the Phase-2 BTP presentation (.pptx) and a markdown content file.

python presentation/make_deck.py
Numbers come from report/numbers.tex; figures from report/figures/*.png.
"""
import os
import re
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures")
OUT_DIR = os.path.join(ROOT, "presentation")
BLUE = RGBColor(0x1F, 0x4E, 0xC9)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xEE, 0xF2, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"

# ------------------------------------------------------------------ numbers
N = {}
with open(os.path.join(ROOT, "report", "numbers.tex"), encoding="utf-8") as f:
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", f.read()):
        N[m.group(1)] = m.group(2)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height
CONTENT_MD = []          # (title, [lines]) for the markdown export


# ------------------------------------------------------------------ helpers
def _tf(shape, text, size, bold=False, color=DARK, align=PP_ALIGN.LEFT, font=FONT):
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return tf


def chrome(slide, number=None):
    """Header label, header rule and footer rule, like the Phase-1 deck."""
    lbl = slide.shapes.add_textbox(Inches(0.9), Inches(0.25), Inches(3), Inches(0.35))
    _tf(lbl, "BTP PRESENTATION", 10, bold=True, color=BLUE)
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(3.0), Inches(0.43), W - Inches(0.6), Inches(0.43))
    line.line.color.rgb = BLUE
    line.line.width = Pt(1)
    foot = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.6), H - Inches(0.35), W - Inches(0.6), H - Inches(0.35))
    foot.line.color.rgb = BLUE
    foot.line.width = Pt(2)
    if number is not None:
        nb = slide.shapes.add_textbox(W - Inches(1.2), H - Inches(0.33), Inches(0.6), Inches(0.3))
        _tf(nb, str(number), 9, color=BLUE, align=PP_ALIGN.RIGHT)


def title(slide, text, top=0.7, left=0.9, width=6.2, size=40):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(1.6))
    _tf(tb, text, size, bold=True, color=BLUE)
    return tb


def bullets(slide, items, left=0.9, top=2.5, width=6.0, height=4.3, size=18, gap=6):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        sub = it.startswith("  ")
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.level = 1 if sub else 0
        r = p.add_run()
        r.text = ("– " if sub else "• ") + it.strip()
        r.font.size = Pt(size - 2 if sub else size)
        r.font.name = FONT
        r.font.color.rgb = DARK
    return tb


def picture(slide, name, left, top, width=None, height=None):
    path = os.path.join(FIG, name)
    kw = {}
    if width:
        kw["width"] = Inches(width)
    if height:
        kw["height"] = Inches(height)
    return slide.shapes.add_picture(path, Inches(left), Inches(top), **kw)


def table(slide, rows, left, top, width, col_widths=None, size=11, header=True, row_h=0.4):
    nrows, ncols = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nrows, ncols, Inches(left), Inches(top), Inches(width), Inches(row_h * nrows))
    t = shp.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            t.columns[i].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = FONT
            r.font.bold = (i == 0 and header) or (j == 0 and i > 0 and header)
            r.font.color.rgb = WHITE if (i == 0 and header) else DARK
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = BLUE if (i == 0 and header) else (LIGHT if i % 2 == 0 else WHITE)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
    return shp


def box(slide, text, left, top, width, height, fill=LIGHT, size=11, bold=False, color=DARK, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(left), Inches(top), Inches(width), Inches(height))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = BLUE
    s.line.width = Pt(1)
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = FONT
    r.font.color.rgb = color
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return s


def arrow(slide, x1, y1, x2, y2, dashed=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = GREY
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    from lxml import etree
    if dashed:                       # prstDash must precede tailEnd in the schema
        dash = etree.SubElement(ln, qn("a:prstDash"))
        dash.set("val", "dash")
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def new_slide(number, ttl, md_lines, title_kw=None):
    s = prs.slides.add_slide(BLANK)
    chrome(s, number)
    if ttl:
        kw = dict(title_kw or {})
        if kw.get("size", 40) <= 32 and "width" not in kw:
            kw["width"] = 11.6          # single-line titles above full-width content
        title(s, ttl, **kw)
    CONTENT_MD.append((ttl or f"Slide {number}", md_lines))
    return s


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ================================================================== slides
n = 0

# 1. Title ------------------------------------------------------------------
n += 1
s = prs.slides.add_slide(BLANK)
chrome(s)
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(2.4))
tf = _tf(tb, "Context Window Analysis for LLMs", 48, bold=True, color=BLUE)
p = tf.add_paragraph()
r = p.add_run()
r.text = "Phase 2: Design and Evaluation of a Persistent Memory Layer"
r.font.size = Pt(28)
r.font.name = FONT
r.font.color.rgb = DARK
ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.9), Inches(4.75), Inches(5.2), Inches(4.75))
ln.line.color.rgb = BLUE
ln.line.width = Pt(3)
tb = s.shapes.add_textbox(Inches(0.9), Inches(5.0), Inches(11), Inches(1.2))
_tf(tb, "B.Tech Project  |  Department of Communication and Computer Engineering, LNMIIT Jaipur  |  November 2026", 14, color=GREY)
CONTENT_MD.append(("Title", ["Context Window Analysis for LLMs", "Phase 2: Design and Evaluation of a Persistent Memory Layer",
                             "B.Tech Project, Department of CCE, LNMIIT Jaipur, November 2026"]))

# 2. Acknowledgement --------------------------------------------------------
n += 1
s = new_slide(n, "Acknowledgement", [
    "Sincere gratitude to Dr. Rahul Sharma, our research supervisor, for guidance, continuous support and constructive feedback throughout both phases of the project.",
    "Team: Lakshya Jain (23UCS633), Monish Mathur (23UCS648), Nishit Luhadia (23UCC579)"])
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.4), Inches(7.2), Inches(2.2))
_tf(tb, "We extend our sincere gratitude to Dr. Rahul Sharma, our research supervisor, for his invaluable "
        "guidance, continuous support and constructive feedback throughout both phases of this project.", 18, color=DARK)
for i, (name, roll) in enumerate([("Lakshya Jain", "23UCS633"), ("Monish Mathur", "23UCS648"), ("Nishit Luhadia", "23UCC579")]):
    box(s, f"{name}\n({roll})", 8.7, 2.3 + i * 1.25, 3.6, 0.95, size=16, bold=True, color=BLUE, fill=LIGHT)

# 3. Table of contents --------------------------------------------------------
toc = ["Recap of Phase 1 and the problem", "Objectives of Phase 2", "Literature review (new works)",
       "System architecture of the memory layer", "Write path and read path", "Grounding gate",
       "Benchmark design and experimental setup", "Results: main comparison, question types, hallucination",
       "Results: ablation, pruning, context-length scaling, efficiency", "Key insights, conclusion and future work"]
n += 1
s = new_slide(n, "Table of Content", toc)
bullets(s, toc, left=5.0, top=1.0, width=7.6, height=6.0, size=18, gap=8)

# 4. Recap of Phase 1 ---------------------------------------------------------
n += 1
rec = ["LLMs are stateless: every request sees only the current context window",
       "Long contexts are used poorly: accuracy drops with length and position (\"lost in the middle\")",
       "RAG grounds answers but stays stateless and still hallucinates",
       "RAGTruth detectors: recall 0.88-0.95 but precision only 0.25-0.26 (F1 0.38-0.41)",
       "Phase 1 proposed a memory layer: embeddings + vector DB + top-k injection",
       "Phase 2 builds it, adds the missing stages and measures it"]
s = new_slide(n, "Recap of Phase 1", rec)
bullets(s, rec, left=0.9, top=2.3, width=6.3, size=17)
table(s, [["Model (RAGTruth)", "Precision", "Recall", "F1"], ["Mistral 7B", "0.25", "0.92", "0.39"],
          ["Llama 2", "0.26", "0.95", "0.41"], ["TinyLlama", "0.25", "0.88", "0.38"]],
      left=7.8, top=2.5, width=4.8, col_widths=[2.1, 0.9, 0.9, 0.9], size=13)
tb = s.shapes.add_textbox(Inches(7.8), Inches(4.4), Inches(4.8), Inches(1.6))
_tf(tb, "Phase-1 result: hallucination detectors flag almost everything. The question for Phase 2: can a "
        "memory layer be both more accurate and more trustworthy than a bigger context window?", 13, color=GREY)

# 5. Objectives ---------------------------------------------------------------
n += 1
obj = ["Implement the complete memory layer as a modular Python package (FAISS + BM25, fusion, entity graph, re-ranking, scoring, conflict-aware updates, token-budgeted injection)",
       "Add a grounding gate that verifies each answer against the injected memories",
       "Build a long-term conversational benchmark: single-hop, temporal, multi-hop and unanswerable questions, plus a NIAH context-length benchmark",
       "Compare five strategies with the same local LLM: no memory, full history, sliding window, vector RAG (Phase-1 design), memory layer",
       "Ablate every pipeline stage and study pruning; tune only on a separate dev split"]
s = new_slide(n, "Objectives of Phase 2", obj)
bullets(s, obj, left=0.9, top=2.2, width=11.5, size=18, gap=10)

# 6-8. Literature review tables ------------------------------------------------
lit = [
    [["Paper (Year)", "Approach", "Core idea", "Strength", "Weakness / our response"],
     ["Karpukhin et al. (2020)", "DPR", "Learned dense bi-encoder retrieval", "Beats BM25 on paraphrased questions", "Misses rare exact strings; we keep a lexical channel"],
     ["Guu et al. (2020); Borgeaud et al. (2022)", "REALM / RETRO", "Retrieval inside pre-training", "Few retrieved chunks beat 25x more parameters", "Needs retraining; our layer works with a frozen LLM"],
     ["Izacard & Grave (2021)", "Fusion-in-Decoder", "Encode passages separately, fuse in decoder", "Gains up to 100 passages", "Needs enc-dec model; we inject a few high-precision notes"],
     ["Khattab & Zaharia (2020); Gao et al. (2023)", "ColBERT / HyDE", "Late interaction; hypothetical-answer embeddings", "Near cross-encoder quality; closes vocabulary gap", "Large index / extra LLM call; we use rule-based query expansion"],
     ["Cormack et al. (2009)", "Reciprocal Rank Fusion", "Merge ranked lists by rank only", "Robust to score scales", "Equal weights let a weak channel inject noise; we weight channels"]],
    [["Paper (Year)", "Approach", "Core idea", "Strength", "Weakness / our response"],
     ["Park et al. (2023)", "Generative Agents", "Memory stream scored by recency + importance + relevance", "Simple, interpretable scoring", "Additive scores let recent-but-irrelevant win; we gate by relevance"],
     ["Packer et al. (2023); Zhong et al. (2024)", "MemGPT / MemoryBank", "LLM pages memory in/out; Ebbinghaus forgetting", "Virtually unbounded memory", "LLM decides storage/recall; ours is deterministic"],
     ["Chhikara et al. (2025)", "Mem0", "Extract facts; add/update/delete; graph variant", "Large token savings on LoCoMo", "One LLM call per turn; we use a zero-cost update rule"],
     ["Edge et al. (2024)", "GraphRAG", "Entity graph + traversal for multi-document questions", "Solves evidence spread across documents", "Graph built by many LLM calls; ours is rule-based"],
     ["Wu et al. (2022); Wang et al. (2023)", "Memorizing Transformers / LongMem", "kNN over past activations", "Very long usable context", "Opaque, model-bound memory; ours stores auditable notes"]],
    [["Paper (Year)", "Approach", "Core idea", "Strength", "Weakness / our response"],
     ["Maharana et al. (2024); Wu et al. (2025)", "LoCoMo / LongMemEval", "Multi-session memory benchmarks", "Single/multi-hop, temporal, abstention taxonomy", "Need large API models; we built a controlled benchmark with gold memory ids"],
     ["Hsieh et al. (2024)", "RULER", "Multi-needle, tracing, aggregation tasks", "Shows effective context << window", "Measures the model, not how to select memories"],
     ["Shi et al. (2023); Xu et al. (2024)", "Distraction / retrieval vs long context", "Irrelevant context hurts; retrieval + 4k beats 32k", "Motivates selective injection", "No conversational memory setting; we add distractors and updates"],
     ["Niu et al. (2024); Manakul et al. (2023)", "RAGTruth / SelfCheckGPT", "Hallucination corpus; consistency sampling", "Standard evaluation", "High recall, low precision; sampling cost"],
     ["Min et al. (2023); Asai et al. (2024)", "FActScore / Self-RAG", "Atomic-fact attribution; reflection tokens", "Verifies support of generations", "External KB / fine-tuning; our gate is deterministic and free"]],
]
for k, rows in enumerate(lit):
    n += 1
    s = new_slide(n, None, [" | ".join(r) for r in rows])
    tb = s.shapes.add_textbox(Inches(0.9), Inches(0.6), Inches(11.5), Inches(0.8))
    _tf(tb, f"Literature Review ({k + 1}/3)", 32, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    table(s, rows, left=0.7, top=1.5, width=12.0, col_widths=[2.3, 1.9, 2.7, 2.5, 2.6], size=11, row_h=0.85)

# 9. Architecture --------------------------------------------------------------
n += 1
s = new_slide(n, "System Architecture", [
    "Write path: turn -> third-person normalisation -> importance scoring/pruning -> entity extraction -> FAISS index, BM25 index, entity graph",
    "Read path: question -> weighted RRF (dense + sparse, query expansion) -> 1-hop graph expansion -> cross-encoder re-ranking -> memory scoring + conflict-aware recency -> top-k notes within a token budget -> LLM -> grounding gate -> answer or abstention"],
              title_kw=dict(size=32))
bx = 2.55
xs = [0.9, 0.9 + 1 * (bx + 0.5), 0.9 + 2 * (bx + 0.5), 0.9 + 3 * (bx + 0.5)]
y1, y2, y3, y4 = 1.75, 3.05, 4.45, 5.65
for x, t in zip(xs, ["Conversation turn\n(user / assistant)", "Normalisation\n(third person)", "Importance scoring\n+ pruning", "Entity\nextraction"]):
    box(s, t, x, y1, bx, 0.8, size=12)
for i in range(3):
    arrow(s, xs[i] + bx, y1 + 0.4, xs[i + 1], y1 + 0.4)
for x, t in zip(xs[1:], ["FAISS vector index", "BM25 keyword index", "Entity graph"]):
    box(s, t, x + 0.3, y2, bx - 0.6, 0.7, size=12, fill=WHITE, shape=MSO_SHAPE.CAN)
    arrow(s, x + bx / 2, y1 + 0.8, x + bx / 2, y2)
for x, t in zip(xs, ["Question", "Weighted RRF\n(dense + sparse, query exp.)", "Graph expansion\n(1 hop)", "Cross-encoder\nre-ranking"]):
    box(s, t, x, y3, bx, 0.8, size=12)
for i in range(3):
    arrow(s, xs[i] + bx, y3 + 0.4, xs[i + 1], y3 + 0.4)
for x, t in zip(xs, ["Grounding gate\n-> answer / \"I don't know\"", "LLM\n(Qwen2.5-1.5B)", "Top-k notes\n(token budget)", "Memory scoring +\nconflict-aware recency"]):
    box(s, t, x, y4, bx, 0.8, size=12)
arrow(s, xs[3] + bx / 2, y3 + 0.8, xs[3] + bx / 2, y4)
for i in (3, 2, 1):
    arrow(s, xs[i], y4 + 0.4, xs[i - 1] + bx, y4 + 0.4)
arrow(s, xs[1] + bx / 2, y2 + 0.7, xs[1] + bx / 2, y3, dashed=True)
arrow(s, xs[2] + bx / 2, y2 + 0.7, xs[1] + bx, y3 + 0.2, dashed=True)
arrow(s, xs[3] + bx / 2, y2 + 0.7, xs[2] + bx / 2, y3, dashed=True)
tb = s.shapes.add_textbox(Inches(0.9), Inches(6.55), Inches(11), Inches(0.4))
_tf(tb, "Top: write path (ingest every turn).  Bottom: read path (answer a question).  Dashed = index look-ups.", 11, color=GREY)

# 10. Write path ---------------------------------------------------------------
n += 1
wp = ["Third-person normalisation: \"I live in Pune\" -> \"The user lives in Pune\"",
      "  question-memory cosine 0.3 -> 0.6; cross-encoder logit -9 -> +4",
      "Importance score from surface cues (speaker, entities, numbers, fact vocabulary, length; questions penalised)",
      "  threshold 0.3 keeps 56% of turns with no recall loss",
      "Three indexes per memory: all-MiniLM-L6-v2 embedding in FAISS, stemmed tokens in BM25, extracted entities in a memory-entity graph (NetworkX)",
      "Ingesting a 288-turn history: well under one second"]
s = new_slide(n, "Write Path: Storing Memories", wp)
bullets(s, wp, left=0.9, top=2.3, width=7.2, size=17)
rows = [["Stored memory (normalised)", "Imp.", "Entities"],
        ["The user live in Jaipur and the user work as a data analyst.", "0.90", "jaipur"],
        ["The user's sister Priya visited the user last weekend.", "0.90", "priya"],
        ["Can you suggest a good movie for tonight?", "0.15", "-"],
        ["Priya adopted a cat and named it Milo.", "0.70", "milo, priya"],
        ["The user moved to Pune last month, so the user is living in Pune now.", "0.90", "pune"]]
table(s, rows, left=8.3, top=2.4, width=4.4, col_widths=[2.9, 0.6, 0.9], size=10, row_h=0.6)

# 11. Read path ----------------------------------------------------------------
n += 1
rp = ["Dense top-20 (FAISS) + sparse top-20 (BM25 on an expanded query, score floor 20% of best)",
      "Weighted reciprocal rank fusion: w_dense = 1, w_sparse = 0.2 (tuned on dev)",
      "Entity-graph expansion: neighbours of the top-3 inherit 0.85 x the seed score -> second hop of multi-hop questions enters the pool",
      "Cross-encoder (MS MARCO MiniLM) blended 50/50 with the fused score",
      "Relevance-gated scoring: rel x (1 + importance + access + graph coherence); questions x 0.7",
      "Conflict-aware recency: older version of an updated fact tagged \"(older note)\" and placed just below the new one",
      "Top-5 notes rendered chronologically inside a 512-token budget"]
s = new_slide(n, "Read Path: Multi-Stage Retrieval", rp)
bullets(s, rp, left=0.9, top=2.2, width=11.6, size=17, gap=8)

# 12. Grounding gate -----------------------------------------------------------
n += 1
gg = ["After generation, the answer is checked against the injected notes with three deterministic rules:",
      "  Attributable: a note covers most of the answer's content words",
      "  Factual: that note is a statement, not a question or request",
      "  Same subject: a question about the user is not answered from a note about a colleague, neighbour or relative",
      "If any check fails the answer is replaced by an abstention",
      "No extra model call (Phase-1 detectors needed extra generations or fine-tuning)",
      f"Result: precision {N['gatePMem']}, recall {N['gateRMem']}, F1 {N['gateFMem']} vs 0.25 / 0.92 / 0.39 in Phase 1"]
s = new_slide(n, "Grounding Gate", gg)
bullets(s, gg, left=0.9, top=2.2, width=7.0, size=16, gap=7)
ex = ["Q: What car does the user drive?", "Notes include: \"The user's colleague Rohan lives in Delhi and drives a Honda City.\"",
      "LLM raw answer: Honda City", "Gate: subject mismatch -> REJECTED", "Final answer: I don't know (not supported by my memory notes)"]
tb = s.shapes.add_textbox(Inches(8.2), Inches(2.4), Inches(4.5), Inches(3.5))
tf = tb.text_frame
tf.word_wrap = True
for i, line in enumerate(ex):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    r = p.add_run()
    r.text = line
    r.font.size = Pt(13)
    r.font.name = "Consolas"
    r.font.color.rgb = BLUE if "REJECTED" in line else DARK
    p.space_after = Pt(6)
tb.fill.solid()
tb.fill.fore_color.rgb = LIGHT

# 13. Benchmark ----------------------------------------------------------------
n += 1
bm = ["Public benchmarks (LoCoMo, LongMemEval) need large API models and give no gold memory ids",
      "Our generator: 20 users x 12 sessions x 12 user-assistant pairs = 288 turns (~3.6k tokens) per user",
      "Per user: 10 single-hop facts, 2 updated facts, 2 multi-hop chains, 8 distractor facts about other people, 3 unanswerable questions",
      "Questions are paraphrased relative to the stored statements",
      f"Test split: {N['nUsers']} users, {N['nQuestions']} questions; separate 6-user dev split (different seed) for all tuning",
      "NIAH-style histories of 1k-16k tokens for the scaling experiment"]
s = new_slide(n, "Benchmark Design", bm)
bullets(s, bm, left=0.9, top=2.2, width=7.3, size=16, gap=8)
table(s, [["Question type", "Count", "Example"], ["Single-hop", "170", "Which gym does the user go to?"],
          ["Temporal (current)", "30", "Where does the user live now?"], ["Temporal (before)", "30", "Where did the user live before moving?"],
          ["Multi-hop", "40", "What did the user's sister name her cat?"], ["Unanswerable", "60", "What is the user's dog called? (never said)"]],
      left=8.3, top=2.4, width=4.5, col_widths=[1.6, 0.7, 2.2], size=10, row_h=0.55)

# 14. Experimental setup -------------------------------------------------------
n += 1
es = ["LLM for every strategy: Qwen2.5-1.5B-Instruct, fp16, greedy decoding, RTX 3050 (4 GB)",
      "Embedding: all-MiniLM-L6-v2; re-ranker: ms-marco-MiniLM-L6 cross-encoder",
      "Strategies: no memory | full history (chunked prefill up to 16k tokens) | sliding window (40 turns) | vector RAG top-5 (Phase-1 design) | memory layer (+ gate)",
      "Metrics: accuracy (gold string match / token-F1 >= 0.6; abstention required for unanswerable), hallucination rate (confident wrong answer), recall@k, MRR, prompt tokens, latency",
      "All hyper-parameters fixed on the dev split before touching the test split"]
s = new_slide(n, "Experimental Setup", es)
bullets(s, es, left=0.9, top=2.2, width=11.6, size=17, gap=10)

# 15. Main results -------------------------------------------------------------
n += 1
rows = [["Strategy", "Accuracy", "Halluc. rate", "Prompt tokens"],
        ["Baseline LLM (no memory)", N["accNone"] + "%", N["halNone"] + "%", N["tokNone"]],
        ["Full history in context", N["accFull"] + "%", N["halFull"] + "%", N["tokFull"]],
        ["Sliding window (40 turns)", N["accWin"] + "%", N["halWin"] + "%", N["tokWin"]],
        ["Vector RAG (Phase-1 design)", N["accRag"] + "%", N["halRag"] + "%", N["tokRag"]],
        ["Memory layer", N["accMem"] + "%", N["halMem"] + "%", N["tokMem"]],
        ["Memory layer + grounding gate", N["accMemGate"] + "%", N["halMemGate"] + "%", N["tokMemGate"]]]
s = new_slide(n, "Results: Main Comparison", [" | ".join(r) for r in rows], title_kw=dict(size=32))
table(s, rows, left=0.8, top=1.7, width=5.6, col_widths=[2.6, 1.0, 1.0, 1.0], size=12, row_h=0.5)
picture(s, "fig_main_accuracy.png", 6.6, 1.6, width=6.3)
tb = s.shapes.add_textbox(Inches(0.8), Inches(5.4), Inches(5.6), Inches(1.6))
_tf(tb, f"Full history: most tokens and the most hallucination ({N['halFull']}%). Memory layer with gate: "
        f"{N['accMemGate']}% accuracy, {N['halMemGate']}% hallucination, {N['tokRatioFullMem']}x fewer tokens than full context.", 13, color=GREY)

# 16. By question type ---------------------------------------------------------
n += 1
qt = [f"Multi-hop: {N['accRagMH']}% -> {N['accMemMH']}% (entity graph brings in the second hop)",
      f"Temporal (before): {N['accRagTB']}% -> {N['accMemTB']}% (old fact kept, tagged as older note); full context only {N['accFullTB']}%",
      f"Temporal (current): {N['accRagTC']}% vs {N['accMemTC']}%",
      f"Single-hop: {N['accRagSH']}% vs {N['accMemSH']}%; retrieval recall > 97%, remaining errors are reading errors of the 1.5B model",
      f"Unanswerable: gate raises abstention to {N['absMemGate']}%"]
s = new_slide(n, "Results by Question Type", qt, title_kw=dict(size=32))
picture(s, "fig_type_accuracy.png", 0.7, 1.6, width=7.4)
bullets(s, qt, left=8.3, top=1.7, width=4.6, size=14, gap=8)

# 17. Hallucination ------------------------------------------------------------
n += 1
rows = [["Detector", "Precision", "Recall", "F1"],
        ["Phase 1: Mistral 7B (RAGTruth)", "0.25", "0.92", "0.39"],
        ["Phase 1: Llama 2 (RAGTruth)", "0.26", "0.95", "0.41"],
        ["Phase 1: TinyLlama (RAGTruth)", "0.25", "0.88", "0.38"],
        ["Phase 2: gate on vector RAG", N["gatePRag"], N["gateRRag"], N["gateFRag"]],
        ["Phase 2: gate on memory layer", N["gatePMem"], N["gateRMem"], N["gateFMem"]]]
hl = [f"Gate flags {N['gateFlaggedMem']} of {N['nQuestions']} answers; {N['gateTPMem']} are real hallucinations (precision {N['gatePMem']})",
      f"Misses {N['gateFNMem']} hallucinations that copy the wrong user fact (attributable, factual, same subject)",
      f"Hallucination rate of the complete system: {N['halMemGate']}% (full context {N['halFull']}%, vector RAG {N['halRag']}%)",
      "Zero additional model calls"]
s = new_slide(n, "Hallucination: Phase 1 vs Phase 2", [" | ".join(r) for r in rows] + hl, title_kw=dict(size=32))
table(s, rows, left=0.8, top=1.7, width=6.2, col_widths=[3.2, 1.0, 1.0, 1.0], size=13, row_h=0.5)
bullets(s, hl, left=7.4, top=1.8, width=5.4, size=15, gap=10)

# 18. Ablation -----------------------------------------------------------------
n += 1
ab = [f"Vector only (Phase-1 design): recall@5 {N['rFiveVec']}%",
      f"BM25 alone is weak ({N['rFiveBm']}%): questions are paraphrases; raw fusion hurts ({N['rFiveHyb']}%), query expansion repairs it ({N['rFiveQe']}%)",
      f"Entity graph: multi-hop recall@5 {N['rFiveVecMH']}% -> {N['rFiveMemMH']}%",
      f"Scoring + question penalty: overall recall@5 {N['rFiveMem']}%, single-hop {N['rFiveMemSH']}%",
      f"Temporal ordering correct: {N['tempVec']}% -> {N['tempMem']}%"]
s = new_slide(n, "Retrieval Ablation", ab, title_kw=dict(size=32))
picture(s, "fig_ablation.png", 0.7, 1.6, width=7.6)
bullets(s, ab, left=8.5, top=1.7, width=4.4, size=14, gap=8)

# 19. Pruning + conflict -------------------------------------------------------
n += 1
pr = [f"Importance threshold 0.3 keeps {N['pruneKeptThree']}% of turns with recall@5 {N['pruneRecThree']}% (all stored: {N['pruneRecZero']}%)",
      f"Threshold 0.45 keeps {N['pruneKeptFortyFive']}% of turns at {N['pruneRecFortyFive']}%: six-fold smaller store for < 4 points",
      "Above 0.5 genuine facts are dropped and recall falls quickly",
      "Conflict threshold: 0.3 orders temporal questions best but tags unrelated facts; dev-chosen 0.45 keeps overall recall at its maximum"]
s = new_slide(n, "Pruning and Conflict Threshold", pr, title_kw=dict(size=32))
picture(s, "fig_pruning.png", 0.7, 1.7, width=6.8)
bullets(s, pr, left=7.8, top=1.8, width=5.0, size=15, gap=10)

# 20. NIAH lines ---------------------------------------------------------------
n += 1
ni = [f"Single fact at 5 depths in 1k-{N['niahMaxLen']} token filler + 6 distractors; {N['niahNeedles']} needles per cell, {N['niahCells']} cells",
      f"Memory layer prompt stays ~{N['niahTokMemLast']} tokens; full context grows to {N['niahTokFullLast']} ({N['niahTokRatioLast']}x)",
      f"Latency at {N['niahMaxLen']}: {N['niahLatFullLast']} s (full) vs {N['niahLatMemLast']} s (memory), {N['niahLatRatioLast']}x faster",
      f"Accuracy: full {N['niahFullAccFirst']}% -> {N['niahFullAccLast']}%; memory {N['niahMemAccFirst']}% -> {N['niahMemAccLast']}% (needle retrieved {N['niahMemRecallMean']}%)",
      "One needle in filler is easy; a real history with similar facts, updates and distractors is where full context collapses (57%)"]
s = new_slide(n, "Context-Length Scaling (NIAH)", ni, title_kw=dict(size=32))
picture(s, "fig_niah_lines.png", 0.7, 1.6, width=7.6)
bullets(s, ni, left=8.5, top=1.7, width=4.4, size=13, gap=8)

# 21. NIAH heatmap -------------------------------------------------------------
n += 1
hm = ["Phase 1 showed NIAH heatmaps from the literature; Phase 2 reproduces them on our own model",
      "Full context: accuracy varies with depth and length (lost-in-the-middle pattern)",
      "Memory layer: retrieval is position-independent; remaining errors are reading errors with more distractors in a 1,118-note store",
      "The difficulty of long-term memory is interference, not length alone"]
s = new_slide(n, "NIAH Heatmap: Depth x Length", hm, title_kw=dict(size=32))
picture(s, "fig_niah_heatmap.png", 0.7, 1.7, width=7.9)
bullets(s, hm, left=8.8, top=1.8, width=4.1, size=14, gap=8)

# 22. Efficiency ---------------------------------------------------------------
n += 1
ef = [f"{N['tokMem']} prompt tokens per query vs {N['tokFull']} for full context ({N['tokRatioFullMem']}x fewer)",
      f"{N['latMem']} s vs {N['latFull']} s per query ({N['latRatioFullMem']}x faster)",
      "Retrieval stages: 30-50 ms per query on GPU, ~30 ms on CPU (plain vector search 6 ms)",
      "Ingest 288 turns: < 1 s; 1,118 notes: ~0.25 s",
      "Cost is independent of history length; full-context cost grows linearly"]
s = new_slide(n, "Efficiency: Tokens and Latency", ef, title_kw=dict(size=32))
picture(s, "fig_tokens_latency.png", 0.7, 1.7, width=7.4)
bullets(s, ef, left=8.3, top=1.8, width=4.6, size=15, gap=10)

# 23. Qualitative example ------------------------------------------------------
n += 1
qe = ["Q: Where does the user currently live?",
      "  [S1] User (older note): The user live in Jaipur ...",
      "  [S3] User: The user moved to Pune last month, so the user is living in Pune now.",
      "  Answer: Pune  (gate: supported)",
      "Q: What is the name of the cat adopted by the user's sister?",
      "  [S1] User: The user's sister Priya visited the user last weekend.",
      "  [S2] User: Priya adopted a cat and named it Milo.   <- reached through entity \"Priya\"",
      "  Answer: Milo  (gate: supported)",
      "Q: What car does the user drive?",
      "  [S2] User: The user's colleague Rohan lives in Delhi and drives a Honda City.",
      "  LLM: Honda City  ->  gate: subject mismatch  ->  I don't know"]
s = new_slide(n, "Qualitative Example", qe, title_kw=dict(size=32))
tb = s.shapes.add_textbox(Inches(0.9), Inches(1.6), Inches(11.6), Inches(5.3))
tf = tb.text_frame
tf.word_wrap = True
for i, line in enumerate(qe):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    r = p.add_run()
    r.text = line
    r.font.size = Pt(14 if line.startswith("Q:") else 13)
    r.font.bold = line.startswith("Q:")
    r.font.name = "Consolas"
    r.font.color.rgb = BLUE if line.startswith("Q:") else DARK
    p.space_after = Pt(4)

# 24. Key insights -------------------------------------------------------------
n += 1
ki = [f"A bigger context window is the worst memory when facts are updated and distractors exist: full context hallucinates {N['halFull']}% of the time",
      f"Selective retrieval is the main gain (vector RAG {N['accRag']}%); the extra stages matter where vector search fails: multi-hop {N['accRagMH']}% -> {N['accMemMH']}%, temporal-before {N['accRagTB']}% -> {N['accMemTB']}%",
      f"Verification matters as much as retrieval: the gate halves hallucination ({N['halMem']}% -> {N['halMemGate']}%) at precision {N['gatePMem']}",
      "Half of the turns carry all of the recall; a sixth carry 96% of it",
      f"Remaining errors are reading errors of the small model (recall@5 {N['recMem']}% vs accuracy {N['accMem']}%)"]
s = new_slide(n, "Key Insights", ki)
bullets(s, ki, left=0.9, top=2.2, width=11.6, size=17, gap=12)

# 25. Conclusion ---------------------------------------------------------------
n += 1
co = ["Phase 1 diagnosed the problem and proposed a memory layer; Phase 2 built it and measured it",
      f"Memory layer + gate: {N['accMemGate']}% accuracy, {N['halMemGate']}% hallucination, {N['tokMem']} tokens/query",
      f"vs full history: {N['accFull']}% / {N['halFull']}% / {N['tokFull']} tokens; vs vector RAG: {N['accRag']}% / {N['halRag']}%",
      f"Grounding gate: precision {N['gatePMem']}, F1 {N['gateFMem']} (Phase 1: 0.25, 0.39), no extra model calls",
      "Constant cost with history length; interference, not length, is the real difficulty",
      "A selective, verified memory is a better foundation for long-running LLM applications than a larger context window"]
s = new_slide(n, "Conclusion", co)
bullets(s, co, left=0.9, top=2.2, width=11.6, size=17, gap=11)

# 26. Future work --------------------------------------------------------------
n += 1
fw = ["Write-time fact extraction with an LLM (one turn -> several atomic memories), batched off-line",
      "Fine-tune the cross-encoder on conversational memory notes; learn importance and subject classifiers",
      "Typed entity graph (sister-of, works-at, lives-in) with explicit edge retirement on updates",
      "Consolidation and forgetting: summaries of related notes, decay of never-accessed notes",
      "Evaluate on LoCoMo / LongMemEval with 7B-70B models",
      "Deployment: per-user isolation, encryption, right-to-forget, and a demo that shows the notes and gate decision behind every answer"]
s = new_slide(n, "Future Work", fw)
bullets(s, fw, left=0.9, top=2.2, width=11.6, size=17, gap=11)

# 27-28. References -------------------------------------------------------------
refs = ["Vaswani et al., Attention Is All You Need, NeurIPS 2017", "Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP, NeurIPS 2020",
        "Gao et al., RAG for LLMs: A Survey, arXiv 2023", "Huang et al., A Survey on Hallucination in LLMs, 2024",
        "Liu et al., Lost in the Middle, TACL 2024", "Modarressi et al., NoLiMa, 2025", "An et al., Why Does the Effective Context Length of LLMs Fall Short?, ICLR 2025",
        "Kamradt, Needle In A Haystack, 2023; Nelson, NIAH Evaluation, 2025", "Karpukhin et al., Dense Passage Retrieval, EMNLP 2020",
        "Guu et al., REALM, ICML 2020; Borgeaud et al., RETRO, ICML 2022", "Izacard and Grave, Fusion-in-Decoder, EACL 2021",
        "Khattab and Zaharia, ColBERT, SIGIR 2020", "Gao et al., HyDE, ACL 2023", "Robertson and Zaragoza, BM25, FnTIR 2009",
        "Cormack et al., Reciprocal Rank Fusion, SIGIR 2009", "Reimers and Gurevych, Sentence-BERT, EMNLP 2019",
        "Nogueira and Cho, Passage Re-ranking with BERT, 2019", "Johnson et al., FAISS, IEEE TBD 2021",
        "Wu et al., Memorizing Transformers, ICLR 2022", "Wang et al., LongMem, NeurIPS 2023",
        "Park et al., Generative Agents, UIST 2023", "Shinn et al., Reflexion, NeurIPS 2023", "Packer et al., MemGPT, 2023",
        "Zhong et al., MemoryBank, AAAI 2024", "Chhikara et al., Mem0, 2025", "Zhang et al., Survey on Memory Mechanisms of LLM Agents, 2024",
        "Edge et al., GraphRAG, 2024", "Shi et al., LLMs Can Be Easily Distracted by Irrelevant Context, ICML 2023",
        "Xu et al., Retrieval meets Long Context LLMs, ICLR 2024", "Hsieh et al., RULER, COLM 2024",
        "Niu et al., RAGTruth, ACL 2024", "Manakul et al., SelfCheckGPT, EMNLP 2023", "Min et al., FActScore, EMNLP 2023",
        "Asai et al., Self-RAG, ICLR 2024", "Maharana et al., LoCoMo, ACL 2024", "Wu et al., LongMemEval, ICLR 2025",
        "Bai et al., LongBench, ACL 2024", "Qwen Team, Qwen2.5 Technical Report, 2024"]
half = (len(refs) + 1) // 2
for k, chunk in enumerate((refs[:half], refs[half:])):
    n += 1
    s = new_slide(n, f"References ({k + 1}/2)", chunk, title_kw=dict(size=32))
    bullets(s, chunk, left=0.9, top=1.6, width=11.8, height=5.5, size=12, gap=3)

# 29. Thank you ----------------------------------------------------------------
n += 1
s = prs.slides.add_slide(BLANK)
chrome(s)
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.4), Inches(11.5), Inches(1.6))
_tf(tb, "Thank You", 54, bold=True, color=BLUE)
tb = s.shapes.add_textbox(Inches(0.9), Inches(4.2), Inches(11.5), Inches(1.4))
_tf(tb, "\"LLMs don't need more context - they need verified memory.\"", 22, color=DARK)
tb = s.shapes.add_textbox(Inches(0.9), Inches(5.3), Inches(11.5), Inches(0.8))
_tf(tb, "Code, benchmarks and report: github.com/projectman14/BTP", 14, color=GREY)
CONTENT_MD.append(("Thank You", ["\"LLMs don't need more context - they need verified memory.\"", "Code: github.com/projectman14/BTP"]))

# ------------------------------------------------------------------ save
os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "BTP_Phase2_Presentation.pptx")
prs.save(out)
with open(os.path.join(OUT_DIR, "slides_content.md"), "w", encoding="utf-8") as f:
    f.write("# Context Window Analysis for LLMs - Phase 2 presentation content\n\n")
    for i, (t, lines) in enumerate(CONTENT_MD, 1):
        f.write(f"## Slide {i}: {t}\n\n")
        for l in lines:
            f.write(("    - " if l.startswith("  ") else "- ") + l.strip() + "\n")
        f.write("\n")
print("saved", out, "slides:", len(prs.slides))
