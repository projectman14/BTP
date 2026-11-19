"""Generate report figures (PDF + PNG) and LaTeX tables from the result JSON files.

python benchmarks/plots.py --results results --figdir report/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated categorical palette (fixed order) + neutral ink
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})
METHOD_LABELS = {"no_memory": "Baseline LLM\n(no memory)", "full_context": "Full context\n(whole history)",
                 "sliding_window": "Sliding window\n(last 40 turns)", "rag_vector": "Vector RAG\n(Phase-1 design)",
                 "memory_layer": "Memory layer\n(proposed)", "memory_layer_gated": "Memory layer\n+ grounding gate",
                 "rag_vector_gated": "Vector RAG\n+ gate"}
TYPE_LABELS = {"single": "Single-hop", "temporal_current": "Temporal\n(current)", "temporal_before": "Temporal\n(before)",
               "multihop": "Multi-hop", "unanswerable": "Unanswerable"}
LADDER_LABELS = {"rag_vector": "Vector only", "bm25_only": "BM25 only", "hybrid": "+ RRF hybrid",
                 "hybrid_qe": "+ query exp.", "hybrid_qe_rerank": "+ re-ranking",
                 "hybrid_qe_rerank_graph": "+ entity graph", "memory_layer": "+ scoring\n(full)"}


def save(fig, figdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, name + ".pdf"))
    fig.savefig(os.path.join(figdir, name + ".png"))
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.2f}", pct=False):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, (f"{100 * v:.0f}%" if pct else fmt.format(v)),
                ha="center", va="bottom", fontsize=7.5, color=INK)


# ---------------------------------------------------------------------------------------
def fig_main(main, figdir):
    s = main["summary"]
    methods = [m for m in ["no_memory", "full_context", "sliding_window", "rag_vector", "memory_layer", "memory_layer_gated"] if m in s]
    y = np.arange(len(methods))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, key, title, col in [(axes[0], "accuracy", "Answer accuracy (all questions)", C[0]),
                                (axes[1], "hallucination_rate", "Hallucination rate (lower is better)", C[1])]:
        vals = [s[m][key] for m in methods]
        cols = [col if m != "memory_layer_gated" else C[2] for m in methods]
        bars = ax.barh(y, vals, height=0.62, color=cols, edgecolor="none")
        for b, v in zip(bars, vals):
            ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{100 * v:.0f}%", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([METHOD_LABELS[m].replace("\n", " ") for m in methods], fontsize=7.5)
    save(fig, figdir, "fig_main_accuracy")


def fig_types(main, figdir):
    s = main["summary"]
    methods = [m for m in ["full_context", "sliding_window", "rag_vector", "memory_layer"] if m in s]
    types = [t for t in ["single", "temporal_current", "temporal_before", "multihop", "unanswerable"] if t in s[methods[0]]["by_type"]]
    x = np.arange(len(types))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, m in enumerate(methods):
        vals = [s[m]["by_type"][t]["accuracy"] for t in types]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, width=w * 0.92, color=C[i],
               label=METHOD_LABELS[m].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[t] for t in types])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, figdir, "fig_type_accuracy")


def fig_tokens(main, figdir):
    s = main["summary"]
    methods = [m for m in ["full_context", "sliding_window", "rag_vector", "memory_layer"] if m in s]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    x = np.arange(len(methods))
    toks = [s[m]["avg_prompt_tokens"] for m in methods]
    bars = axes[0].bar(x, toks, width=0.6, color=C[0])
    for b, v in zip(bars, toks):
        axes[0].text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:,.0f}", ha="center", fontsize=7.5)
    axes[0].set_yscale("log")
    axes[0].set_ylim(50, max(toks) * 3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=6.8)
    axes[0].set_title("Prompt tokens per query (log scale)", fontsize=9, loc="left")
    axes[0].grid(axis="x", visible=False)
    lat = [1000 * s[m]["avg_latency_s"] for m in methods]
    bars = axes[1].bar(x, lat, width=0.6, color=C[2])
    for b, v in zip(bars, lat):
        axes[1].text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:,.0f}", ha="center", fontsize=7.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=6.8)
    axes[1].set_title("End-to-end latency per query (ms)", fontsize=9, loc="left")
    axes[1].set_ylim(0, max(lat) * 1.18)
    axes[1].grid(axis="x", visible=False)
    save(fig, figdir, "fig_tokens_latency")


def fig_niah(niah, figdir):
    L = [int(k) for k in niah["summary"]]
    full = [niah["summary"][str(l)]["full"]["accuracy"] for l in L]
    mem = [niah["summary"][str(l)]["memory"]["accuracy"] for l in L]
    tf = [niah["summary"][str(l)]["full"]["avg_prompt_tokens"] for l in L]
    tm = [niah["summary"][str(l)]["memory"]["avg_prompt_tokens"] for l in L]
    lf = [niah["summary"][str(l)]["full"]["avg_latency_s"] for l in L]
    lm = [niah["summary"][str(l)]["memory"]["avg_latency_s"] for l in L]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    ax.plot(L, full, "-o", color=C[1], lw=2, ms=5, label="Full context")
    ax.plot(L, mem, "-o", color=C[0], lw=2, ms=5, label="Memory layer")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Needle accuracy", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="lower left")
    ax = axes[1]
    ax.plot(L, tf, "-o", color=C[1], lw=2, ms=5)
    ax.plot(L, tm, "-o", color=C[0], lw=2, ms=5)
    ax.set_yscale("log")
    ax.set_title("Prompt tokens (log)", fontsize=9, loc="left")
    ax = axes[2]
    ax.plot(L, lf, "-o", color=C[1], lw=2, ms=5)
    ax.plot(L, lm, "-o", color=C[0], lw=2, ms=5)
    ax.set_yscale("log")
    ax.set_title("Latency per query, s (log)", fontsize=9, loc="left")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(L)
        ax.set_xticklabels([f"{l // 1000}k" for l in L])
        ax.set_xlabel("History length (tokens)")
    save(fig, figdir, "fig_niah_lines")

    # heatmaps (depth x length), one sequential hue, light -> dark
    depths = niah["meta"]["depths"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, m, title in [(axes[0], "full", "Full context"), (axes[1], "memory", "Memory layer")]:
        grid = np.array([[niah["heatmap"][m][str(l)][str(d)] for l in L] for d in depths])
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(L)))
        ax.set_xticklabels([f"{l // 1000}k" for l in L])
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{int(d * 100)}%" for d in depths])
        ax.set_xlabel("History length (tokens)")
        ax.set_title(f"{title}: accuracy by needle depth", fontsize=9, loc="left")
        ax.grid(False)
        for i in range(len(depths)):
            for j in range(len(L)):
                v = grid[i, j]
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else INK)
    axes[0].set_ylabel("Needle depth in history")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="accuracy")
    fig.savefig(os.path.join(figdir, "fig_niah_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(figdir, "fig_niah_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def fig_ablation(abl, figdir):
    names = [n for n in LADDER_LABELS if n in abl["ladder"]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.27
    for i, (key, lab) in enumerate([("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("mrr", "MRR")]):
        vals = [abl["ladder"][n][key] for n in names]
        bars = ax.bar(x + (i - 1) * w, vals, width=w * 0.92, color=C[i], label=lab)
        if key == "recall@5":
            bar_labels(ax, bars, pct=True)
    ax.set_xticks(x)
    ax.set_xticklabels([LADDER_LABELS[n] for n in names], fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, figdir, "fig_ablation")

    # per-type recall@5 for first vs last rung
    first, last = names[0], names[-1]
    types = [t for t in ["single", "temporal_current", "temporal_before", "multihop"] if t in abl["ladder"][last]["by_type"]]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    x = np.arange(len(types))
    for i, (n, lab) in enumerate([(first, "Vector only (Phase-1 design)"), (last, "Memory layer (full pipeline)")]):
        vals = [abl["ladder"][n]["by_type"][t]["recall@5"] for t in types]
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, width=0.33, color=C[i], label=lab)
        bar_labels(ax, bars, pct=True)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[t] for t in types], fontsize=7.5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, figdir, "fig_ablation_types")

    # pruning curve
    thr = sorted(abl["pruning"], key=float)
    kept = [100 * abl["pruning"][t]["memories_stored"] / abl["pruning"][t]["memories_seen"] for t in thr]
    rec = [100 * abl["pruning"][t]["recall@5"] for t in thr]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(kept, rec, "-o", color=C[0], lw=2, ms=5)
    for k, r, t in zip(kept, rec, thr):
        ax.annotate(f"θ={t}", (k, r), textcoords="offset points", xytext=(4, -10), fontsize=7, color=INK2)
    ax.set_xlabel("Memories kept after write-time pruning (% of turns)")
    ax.set_ylabel("Recall@5 (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 105)
    ax.set_title("Importance-based pruning: storage vs. recall", fontsize=9, loc="left")
    save(fig, figdir, "fig_pruning")


# ---------------------------------------------------------------------------------------
def pct(v: float) -> str:
    return f"{100 * v:.1f}\\%"


def tables(main, abl, niah, outpath):
    s = main["summary"]
    L = []
    L.append("% auto-generated by benchmarks/plots.py\n")
    # main table
    L.append("\\begin{table}[htbp]\\centering\\small\\resizebox{\\textwidth}{!}{%")
    L.append("\\begin{tabular}{lccccc}\\hline")
    L.append("\\textbf{Method} & \\textbf{Accuracy} & \\textbf{Acc. (answerable)} & \\textbf{Halluc. rate} & \\textbf{Abstain (unans.)} & \\textbf{Prompt tokens}\\\\\\hline")
    for m in ["no_memory", "full_context", "sliding_window", "rag_vector", "memory_layer", "memory_layer_gated"]:
        if m not in s:
            continue
        r = s[m]
        name = METHOD_LABELS[m].replace("\n", " ")
        bold = m.startswith("memory_layer")
        cells = [name, pct(r["accuracy"]), pct(r["accuracy_answerable"]), pct(r["hallucination_rate"]),
                 pct(r["abstain_rate_unanswerable"]), f"{r['avg_prompt_tokens']:,.0f}"]
        if bold:
            cells = ["\\textbf{" + c + "}" for c in cells]
        L.append(" & ".join(cells) + "\\\\")
    L.append("\\hline\\end{tabular}}")
    L.append("\\caption{End-to-end comparison on the long-term conversational memory benchmark "
             f"({main['meta']['users']} users, {s['memory_layer']['n']} questions, {main['meta']['model'].split('/')[-1]}).}}")
    L.append("\\label{tab:main}\\end{table}\n")

    # by type
    methods = [m for m in ["full_context", "sliding_window", "rag_vector", "memory_layer"] if m in s]
    types = ["single", "temporal_current", "temporal_before", "multihop", "unanswerable"]
    L.append("\\begin{table}[htbp]\\centering\\small\\resizebox{\\textwidth}{!}{%")
    L.append("\\begin{tabular}{l" + "c" * len(types) + "}\\hline")
    L.append("\\textbf{Method} & " + " & ".join("\\textbf{" + TYPE_LABELS[t].replace("\n", " ") + "}" for t in types) + "\\\\\\hline")
    for m in methods:
        L.append(METHOD_LABELS[m].replace("\n", " ") + " & " +
                 " & ".join(f"{100 * s[m]['by_type'][t]['accuracy']:.1f}\\%" for t in types) + "\\\\")
    L.append("\\hline\\end{tabular}}")
    L.append("\\caption{Accuracy by question type.}\\label{tab:types}\\end{table}\n")

    # hallucination detection (compare with Phase-1 Table 4.1)
    d = main["hallucination_detection"]
    L.append("\\begin{table}[htbp]\\centering\\small")
    L.append("\\begin{tabular}{lccc}\\hline")
    L.append("\\textbf{Detector} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1}\\\\\\hline")
    L.append("Phase 1: Mistral 7B (RAGTruth) & 0.25 & 0.92 & 0.39\\\\")
    L.append("Phase 1: Llama 2 (RAGTruth) & 0.26 & 0.95 & 0.41\\\\")
    L.append("Phase 1: TinyLlama (RAGTruth) & 0.25 & 0.88 & 0.38\\\\\\hline")
    for k, name in [("rag_vector", "Phase 2: grounding gate on vector RAG"), ("memory_layer", "Phase 2: grounding gate on memory layer")]:
        if k in d:
            L.append(f"{name} & {d[k]['precision']:.2f} & {d[k]['recall']:.2f} & {d[k]['f1']:.2f}\\\\")
    L.append("\\hline\\end{tabular}")
    L.append("\\caption{Hallucination detection: Phase-1 results versus the Phase-2 grounding gate.}\\label{tab:halluc}\\end{table}\n")

    # ablation ladder
    L.append("\\begin{table}[htbp]\\centering\\small")
    L.append("\\begin{tabular}{lccccc}\\hline")
    L.append("\\textbf{Configuration} & \\textbf{R@1} & \\textbf{R@3} & \\textbf{R@5} & \\textbf{MRR} & \\textbf{Temporal ok}\\\\\\hline")
    for n in LADDER_LABELS:
        if n in abl["ladder"]:
            a = abl["ladder"][n]
            L.append(f"{LADDER_LABELS[n].replace(chr(10), ' ')} & {100 * a['recall@1']:.1f}\\% & {100 * a['recall@3']:.1f}\\% & "
                     f"{100 * a['recall@5']:.1f}\\% & {a['mrr']:.3f} & {100 * a.get('temporal_ok', 0):.1f}\\%\\\\")
    L.append("\\hline\\end{tabular}")
    L.append("\\caption{Retrieval ablation: each row adds one stage to the previous one. "
             "R@k = all gold memories within the top-k; Temporal ok = the current fact outranks the outdated one.}"
             "\\label{tab:ablation}\\end{table}\n")

    # pruning
    L.append("\\begin{table}[htbp]\\centering\\small")
    L.append("\\begin{tabular}{lccc}\\hline")
    L.append("\\textbf{Importance threshold $\\theta$} & \\textbf{Memories kept} & \\textbf{R@5} & \\textbf{MRR}\\\\\\hline")
    for t in sorted(abl["pruning"], key=float):
        a = abl["pruning"][t]
        L.append(f"{t} & {100 * a['memories_stored'] / a['memories_seen']:.1f}\\% & {100 * a['recall@5']:.1f}\\% & {a['mrr']:.3f}\\\\")
    L.append("\\hline\\end{tabular}")
    L.append("\\caption{Write-time pruning: fraction of turns kept as memories and the resulting retrieval quality.}\\label{tab:pruning}\\end{table}\n")

    # niah
    L.append("\\begin{table}[htbp]\\centering\\small")
    L.append("\\begin{tabular}{lcccccc}\\hline")
    L.append("\\textbf{History length} & \\multicolumn{3}{c}{\\textbf{Full context}} & \\multicolumn{3}{c}{\\textbf{Memory layer}}\\\\")
    L.append(" & Acc. & Tokens & Latency & Acc. & Tokens & Latency\\\\\\hline")
    for l, r in niah["summary"].items():
        L.append(f"{int(l) // 1000}k & {100 * r['full']['accuracy']:.0f}\\% & {r['full']['avg_prompt_tokens']:,.0f} & {r['full']['avg_latency_s']:.2f}s & "
                 f"{100 * r['memory']['accuracy']:.0f}\\% & {r['memory']['avg_prompt_tokens']:,.0f} & {r['memory']['avg_latency_s']:.2f}s\\\\")
    L.append("\\hline\\end{tabular}")
    L.append("\\caption{Needle-in-a-haystack scaling: accuracy, prompt size and latency versus history length.}\\label{tab:niah}\\end{table}\n")

    text = "\n".join(L)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(text)
    # also one file per table so the report can place each next to its discussion
    tabdir = os.path.join(os.path.dirname(outpath), "tables")
    os.makedirs(tabdir, exist_ok=True)
    for block in text.split("\\begin{table}")[1:]:
        block = "\\begin{table}" + block
        label = block.split("\\label{tab:")[1].split("}")[0]
        with open(os.path.join(tabdir, f"tab_{label}.tex"), "w", encoding="utf-8") as f:
            f.write(block.strip() + "\n")


def macros(main, abl, niah, outpath):
    """LaTeX macros with the key numbers so that the report prose always matches the results."""
    s = main["summary"]
    d = main["hallucination_detection"]
    M = {}

    def p(v):
        return f"{100 * v:.1f}"

    M["nUsers"] = str(main["meta"]["users"])
    M["nQuestions"] = str(s["memory_layer"]["n"])
    M["llmName"] = main["meta"]["model"].split("/")[-1]
    keymap = {"no_memory": "None", "full_context": "Full", "sliding_window": "Win", "rag_vector": "Rag",
              "memory_layer": "Mem", "memory_layer_gated": "MemGate", "rag_vector_gated": "RagGate"}
    tmap = {"single": "SH", "temporal_current": "TC", "temporal_before": "TB", "multihop": "MH", "unanswerable": "UN"}
    for m, k in keymap.items():
        if m not in s:
            continue
        r = s[m]
        M["acc" + k] = p(r["accuracy"])
        M["accAns" + k] = p(r["accuracy_answerable"])
        M["hal" + k] = p(r["hallucination_rate"])
        M["halAns" + k] = p(r["hallucination_rate_answerable"])
        M["abs" + k] = p(r["abstain_rate_unanswerable"])
        M["tok" + k] = f"{r['avg_prompt_tokens']:,.0f}"
        M["lat" + k] = f"{r['avg_latency_s']:.2f}"
        if "recall_at_k" in r:
            M["rec" + k] = p(r["recall_at_k"])
            M["mrr" + k] = f"{r['mrr']:.3f}"
        for t, tk in tmap.items():
            if t in r["by_type"]:
                M["acc" + k + tk] = p(r["by_type"][t]["accuracy"])
                M["hal" + k + tk] = p(r["by_type"][t]["hallucination_rate"])
    for m, k in [("memory_layer", "Mem"), ("rag_vector", "Rag")]:
        if m in d:
            M["gateP" + k] = f"{d[m]['precision']:.2f}"
            M["gateR" + k] = f"{d[m]['recall']:.2f}"
            M["gateF" + k] = f"{d[m]['f1']:.2f}"
            M["gateFlagged" + k] = str(d[m]["tp"] + d[m]["fp"])
            M["gateTP" + k] = str(d[m]["tp"])
            M["gateFP" + k] = str(d[m]["fp"])
            M["gateFN" + k] = str(d[m]["fn"])
            M["gatePos" + k] = str(d[m]["positives"])
    M["tokRatioFullMem"] = f"{s['full_context']['avg_prompt_tokens'] / s['memory_layer']['avg_prompt_tokens']:.0f}"
    M["latRatioFullMem"] = f"{s['full_context']['avg_latency_s'] / s['memory_layer']['avg_latency_s']:.0f}"

    lad = abl["ladder"]
    for n, k in [("rag_vector", "Vec"), ("bm25_only", "Bm"), ("hybrid", "Hyb"), ("hybrid_qe", "Qe"),
                 ("hybrid_qe_rerank", "Rr"), ("hybrid_qe_rerank_graph", "Gr"), ("memory_layer", "Mem")]:
        M["rFive" + k] = p(lad[n]["recall@5"])
        M["rOne" + k] = p(lad[n]["recall@1"])
        M["rThree" + k] = p(lad[n]["recall@3"])
        M["mrrL" + k] = f"{lad[n]['mrr']:.3f}"
        M["temp" + k] = p(lad[n].get("temporal_ok", 0))
        for t, tk in tmap.items():
            if t in lad[n]["by_type"]:
                M["rFive" + k + tk] = p(lad[n]["by_type"][t]["recall@5"])
    for thr, k in [("0.3", "Three"), ("0.45", "FortyFive"), ("0.55", "FiftyFive")]:
        a = abl["pruning"][thr]
        M["pruneKept" + k] = p(a["memories_stored"] / a["memories_seen"])
        M["pruneRec" + k] = p(a["recall@5"])
    M["pruneRecZero"] = p(abl["pruning"]["0.0"]["recall@5"])

    ns = niah["summary"]
    Ls = sorted(int(x) for x in ns)
    first, last = str(Ls[0]), str(Ls[-1])
    M["niahMaxLen"] = f"{Ls[-1] // 1000}k"
    M["niahNeedles"] = str(niah["meta"]["needles"])
    M["niahCells"] = str(len(niah["cells"]))
    M["niahFullAccFirst"] = p(ns[first]["full"]["accuracy"])
    M["niahFullAccLast"] = p(ns[last]["full"]["accuracy"])
    M["niahMemAccFirst"] = p(ns[first]["memory"]["accuracy"])
    M["niahMemAccLast"] = p(ns[last]["memory"]["accuracy"])
    M["niahFullAccMean"] = p(sum(ns[l]["full"]["accuracy"] for l in ns) / len(ns))
    M["niahMemAccMean"] = p(sum(ns[l]["memory"]["accuracy"] for l in ns) / len(ns))
    M["niahFullAccMin"] = p(min(ns[l]["full"]["accuracy"] for l in ns))
    M["niahMemAccMin"] = p(min(ns[l]["memory"]["accuracy"] for l in ns))
    M["niahMemRecallMean"] = p(sum(ns[l]["memory"]["needle_recall"] for l in ns) / len(ns))
    M["niahTokFullLast"] = f"{ns[last]['full']['avg_prompt_tokens']:,.0f}"
    M["niahTokMemLast"] = f"{ns[last]['memory']['avg_prompt_tokens']:,.0f}"
    M["niahTokRatioLast"] = f"{ns[last]['full']['avg_prompt_tokens'] / ns[last]['memory']['avg_prompt_tokens']:.0f}"
    M["niahLatFullLast"] = f"{ns[last]['full']['avg_latency_s']:.1f}"
    M["niahLatMemLast"] = f"{ns[last]['memory']['avg_latency_s']:.2f}"
    M["niahLatFullFirst"] = f"{ns[first]['full']['avg_latency_s']:.2f}"
    M["niahLatRatioLast"] = f"{ns[last]['full']['avg_latency_s'] / ns[last]['memory']['avg_latency_s']:.0f}"
    M["niahMemMemories"] = f"{max(c['memory']['memories'] for c in niah['cells'])}"

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("% auto-generated by benchmarks/plots.py -- do not edit\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figdir", default="report/figures")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    main_r = json.load(open(os.path.join(args.results, "main_results.json"), encoding="utf-8"))
    abl = json.load(open(os.path.join(args.results, "ablation.json"), encoding="utf-8"))
    niah = json.load(open(os.path.join(args.results, "niah.json"), encoding="utf-8"))
    fig_main(main_r, args.figdir)
    fig_types(main_r, args.figdir)
    fig_tokens(main_r, args.figdir)
    fig_niah(niah, args.figdir)
    fig_ablation(abl, args.figdir)
    report_dir = os.path.dirname(args.figdir.rstrip("/\\"))
    tables(main_r, abl, niah, os.path.join(report_dir, "tables.tex"))
    macros(main_r, abl, niah, os.path.join(report_dir, "numbers.tex"))
    print("figures written to", args.figdir)


if __name__ == "__main__":
    main()
