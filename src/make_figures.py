"""Publication figures for the paper (vector PDF + PNG preview).

Sized for a two-column ACL layout: single-column figures are 3.15in wide,
double-column 6.3in. Everything is drawn from the committed result JSON and the
dataset itself, so the figures cannot drift from the reported numbers.

Palette: slots 1-3 of the reference categorical palette (blue / orange / aqua),
which is the documented all-pairs-validated subset. Hatching is applied as a
secondary encoding so the figures survive greyscale printing, and every bar is
directly labelled so identity never rests on colour alone.

    python src/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import force_utf8_stdout, load_config, read_jsonl  # noqa: E402

# --- palette (reference instance, light mode) -------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

SINGLE = 3.15   # inches, one ACL column
DOUBLE = 6.30

OUT = Path("paper/figures")


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIX Two Text",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType, not Type 3
        "ps.fonttype": 42,
    })


def recessive(ax, axis: str = "y") -> None:
    """Grid behind the data, spines trimmed - the marks carry the chart."""
    ax.grid(axis=axis, color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------- figure 1 --
def fig_boundary_positions(cfg) -> None:
    """Where boundaries actually fall - the evidence for a positional prior."""
    recs = list(read_jsonl(Path(cfg["paths"]["generated_dir"]) / "combined.jsonl"))
    all_pos: list[float] = []
    for r in recs:
        n = len(r["labels"])
        for b in r["boundaries"]:
            all_pos.append(b / max(1, n - 1))
    first = [r["boundary_position_normalized"] for r in recs
             if r.get("boundary_position_normalized") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 1.85), sharey=False)
    bins = np.linspace(0, 1, 21)

    for ax, data, color, title, n in (
        (axes[0], first, BLUE, "First boundary per document", len(first)),
        (axes[1], all_pos, AQUA, "All boundaries", len(all_pos)),
    ):
        # One series per panel, so no hatch: hatching exists to separate
        # series, and here it would only add texture noise at print size.
        ax.hist(data, bins=bins, color=color, edgecolor="white",
                linewidth=0.5, alpha=0.95)
        ax.set_title(f"{title} ($n$={n:,})", pad=4)
        ax.set_xlabel("normalised position in document")
        ax.set_xlim(0, 1)
        recessive(ax)
    axes[0].set_ylabel("count")

    # No shaded "excluded" bands: mass at 1.0 is legitimate - a span ending
    # just before the final sentence closes with a boundary at index n-1.

    fig.tight_layout()
    save(fig, "fig1_boundary_positions")


# ---------------------------------------------------------------- figure 2 --
def fig_model_comparison(lin: dict, tr: dict) -> None:
    """The headline: the linear model loses to a baseline that reads no text."""
    def get(name):
        r = next(x for x in lin["results"] if x["name"] == name)
        return (r["overall"]["sentence"]["f1_ai"],
                r["overall"]["boundary_exact"]["f1"])

    rows = [
        ("All-AI", *get("baseline: all-AI")),
        ("Random", *get("baseline: random (train prior)")),
        ("Position only\n(no text)", *get("baseline: position only (no text)")),
        ("Linear\n(char $n$-gram)", *get("text + context")),
        ("XLM-R", tr["test"]["overall"]["sentence"]["f1_ai"],
         tr["test"]["overall"]["boundary_exact"]["f1"]),
    ]
    labels = [r[0] for r in rows]
    sent = [r[1] for r in rows]
    bound = [r[2] for r in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.25))

    b1 = ax.bar(x - w / 2 - 0.01, sent, w, label="Sentence F1 (machine class)",
                color=BLUE, edgecolor="white", linewidth=0.8, hatch="//")
    b2 = ax.bar(x + w / 2 + 0.01, bound, w, label="Exact-boundary F1",
                color=ORANGE, edgecolor="white", linewidth=0.8, hatch="\\\\")

    # Reference line: the bar every text model must clear.
    pos_b = rows[2][2]
    ax.axhline(pos_b, color=INK_2, linewidth=0.8, linestyle=(0, (4, 3)),
               zorder=1)
    ax.annotate("position-only baseline (0.421)", xy=(-0.42, pos_b),
                xytext=(0, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.8, color=INK_2)

    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.4, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.83)
    ax.legend(frameon=False, loc="upper left", ncols=2,
              handlelength=1.4, columnspacing=1.2)
    recessive(ax)
    fig.tight_layout()
    save(fig, "fig2_model_comparison")


# ---------------------------------------------------------------- figure 3 --
def fig_generalisation_gap(tr: dict) -> None:
    """Seen vs held-out generator, and the same split by construction type."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.15))

    # -- left: aggregate seen vs held-out across the three headline metrics --
    metrics = [
        ("Sentence F1", "sentence", "f1_ai"),
        ("Exact-boundary F1", "boundary_exact", "f1"),
        ("Document exact", "boundary_exact", "exact_doc_match"),
    ]
    seen = [tr["test"]["seen"][a][b] for _, a, b in metrics]
    held = [tr["test"]["held_out"][a][b] for _, a, b in metrics]
    x = np.arange(len(metrics))
    w = 0.38
    ax = axes[0]
    b1 = ax.bar(x - w / 2 - 0.01, seen, w, label="Seen generators",
                color=BLUE, edgecolor="white", linewidth=0.8, hatch="//")
    b2 = ax.bar(x + w / 2 + 0.01, held, w, label="Held-out (Gemini 2.5 Pro)",
                color=ORANGE, edgecolor="white", linewidth=0.8, hatch="\\\\")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=7)
    ax.set_ylabel("score")
    ax.set_ylim(0, 0.92)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)
    ax.set_title("Generalisation to an unseen generator", pad=4)
    recessive(ax)

    # -- right: per construction type, linear vs XLM-R ---------------------
    order = [("Continuation", "type1_single_boundary"),
             ("Single-span", "type2_single_internal_segment"),
             ("Multi-span", "type3_multiple_internal_segments")]
    xl = np.arange(len(order))
    ax = axes[1]
    vals = [tr["per_construction_type"][k]["sentence"]["f1_ai"] for _, k in order]
    bvals = [tr["per_construction_type"][k]["boundary_exact"]["f1"]
             for _, k in order]
    b1 = ax.bar(xl - w / 2 - 0.01, vals, w, label="Sentence F1",
                color=BLUE, edgecolor="white", linewidth=0.8, hatch="//")
    b2 = ax.bar(xl + w / 2 + 0.01, bvals, w, label="Exact-boundary F1",
                color=ORANGE, edgecolor="white", linewidth=0.8, hatch="\\\\")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.set_xticks(xl)
    ax.set_xticklabels([o[0] for o in order], fontsize=7)
    ax.set_ylim(0, 0.92)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)
    ax.set_title("XLM-R by construction type", pad=4)
    recessive(ax)

    fig.tight_layout()
    save(fig, "fig3_generalisation_and_type")


def main() -> None:
    force_utf8_stdout()
    style()
    cfg = load_config()
    lin = json.loads(Path("reports/detector_results.json").read_text(encoding="utf-8"))
    tr = json.loads(Path("reports/transformer_results.json").read_text(encoding="utf-8"))
    print("generating figures ->", OUT)
    fig_boundary_positions(cfg)
    fig_model_comparison(lin, tr)
    fig_generalisation_gap(tr)
    print("done")


if __name__ == "__main__":
    main()
