"""Dataset figures, written to reports/figures/.

A data paper's figures should describe the *data*, not detector performance, so
these are: the three construction schemas, and the distribution of boundary
positions (evidence that the corpus carries no trivial positional shortcut).

Sized for the NeurIPS text block (5.5in wide), Times-matched, vector PDF.

    python src/make_abd_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import force_utf8_stdout, load_config, read_jsonl  # noqa: E402

HUMAN = "#2a78d6"      # categorical slot 1
AI = "#eb6834"         # categorical slot 2
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

WIDTH = 5.5            # NeurIPS text width, inches
OUT = Path("reports/figures")


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIX Two Text",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": INK_2, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
        print(f"  wrote {OUT / f'{name}.{ext}'}")
    plt.close(fig)


# ------------------------------------------------------ construction schema --
def fig_schema() -> None:
    """The three document types as label patterns, with real statistics."""
    rows = [
        ("Continuation", [0, 0, 0, 0, 1, 1, 1, 1],
         "1,675 docs · 1 boundary · 49.4% machine"),
        ("Single-span replacement", [0, 0, 0, 1, 1, 0, 0, 0],
         "1,367 docs · 2 boundaries · 24.5% machine"),
        ("Multi-span replacement", [0, 0, 1, 0, 1, 1, 0, 0],
         "1,202 docs · 4.97 boundaries · 33.0% machine"),
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.15))
    ax.set_xlim(-0.4, 8.4)
    ax.set_ylim(-0.3, len(rows) * 1.5)
    ax.axis("off")

    bw, bh = 0.86, 0.44
    for r, (name, labels, stats) in enumerate(rows):
        y = (len(rows) - 1 - r) * 1.5 + 0.30
        ax.text(-0.35, y + bh + 0.30, name, fontsize=8.5, fontweight="bold",
                va="bottom", ha="left")
        ax.text(-0.35, y + bh + 0.06, stats, fontsize=7.2, color=INK_2,
                va="bottom", ha="left")
        for i, lab in enumerate(labels):
            ax.add_patch(FancyBboxPatch(
                (i, y), bw, bh,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=AI if lab else HUMAN, edgecolor="white",
                linewidth=0.8))
            ax.text(i + bw / 2, y + bh / 2, "A" if lab else "H",
                    color="white", fontsize=7.5, fontweight="bold",
                    ha="center", va="center")
        # mark every authorship change
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                ax.plot([i - 0.07, i - 0.07], [y - 0.10, y + bh + 0.10],
                        color=INK, linewidth=1.3, solid_capstyle="butt")
    ax.legend(handles=[Patch(facecolor=HUMAN, label="human sentence"),
                       Patch(facecolor=AI, label="machine sentence")],
              loc="lower right", frameon=False, ncols=2,
              bbox_to_anchor=(1.0, -0.10), handlelength=1.1)
    ax.text(-0.35, -0.12, "vertical rule = authorship boundary",
            fontsize=7.2, color=INK_2, va="bottom")
    fig.tight_layout()
    save(fig, "abd_fig1_constructions")


# ------------------------------------------------------- boundary positions --
def fig_positions(cfg) -> None:
    """Boundary locations, and machine-sentence share by construction."""
    recs = list(read_jsonl(Path(cfg["paths"]["generated_dir"]) / "combined.jsonl"))
    allpos = [b / max(1, len(r["labels"]) - 1)
              for r in recs for b in r["boundaries"]]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.85),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    ax.hist(allpos, bins=np.linspace(0, 1, 21), color=HUMAN,
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("normalised position in document")
    ax.set_ylabel("boundaries")
    ax.set_xlim(0, 1)
    ax.set_title(f"All boundaries ($n$={len(allpos):,})", pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    order = [("Cont.", "type1_single_boundary"),
             ("Single", "type2_single_internal_segment"),
             ("Multi", "type3_multiple_internal_segments")]
    vals = []
    for _, key in order:
        sub = [r for r in recs if r["construction_type"] == key]
        vals.append(np.mean([sum(r["labels"]) / len(r["labels"]) for r in sub]))
    bars = ax.bar(range(3), vals, 0.62, color=AI, edgecolor="white",
                  linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1%}", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([o[0] for o in order])
    ax.set_ylabel("machine sentences")
    ax.set_ylim(0, 0.62)
    ax.set_title("Machine share by type", pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    save(fig, "abd_fig2_positions")


def main() -> None:
    force_utf8_stdout()
    style()
    cfg = load_config()
    print("generating Sinhala-ABD figures ->", OUT)
    fig_schema()
    fig_positions(cfg)
    print("done")


if __name__ == "__main__":
    main()
