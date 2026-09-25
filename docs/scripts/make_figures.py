"""Regenerate every README figure from the versioned results.

Each figure is written twice, docs/assets/<name>-light.png and -dark.png, so
the README can serve the variant matching the reader's GitHub theme. Inputs
are the result files under results/ only; nothing here needs the dataset,
the models or a GPU.

Colour: categorical slots 1-4 of the reference palette (blue, orange, aqua,
yellow), each with a separately stepped dark-mode value, validated for colour
vision deficiency in both modes. Every series is also direct-labelled, so no
identity rests on colour alone.

    python docs/scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "detection"
OUT = ROOT / "docs" / "assets"

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}
T: dict = {}   # the active theme


def load(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text(encoding="utf-8"))


# --------------------------------------------------------------- styling ---
def style(theme: str) -> None:
    T.clear()
    T.update(THEMES[theme])
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "Liberation Sans",
                            "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
        "figure.facecolor": T["surface"], "axes.facecolor": T["surface"],
        "savefig.facecolor": T["surface"],
        "text.color": T["ink"], "axes.labelcolor": T["ink2"],
        "axes.titlecolor": T["ink"], "axes.edgecolor": T["axis"],
        "xtick.color": T["muted"], "ytick.color": T["muted"],
        "xtick.labelcolor": T["ink2"], "ytick.labelcolor": T["ink2"],
        "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    })


def recessive(ax, axis: str = "y") -> None:
    """Hairline grid behind the data; only the baseline axis kept."""
    ax.grid(axis=axis, color=T["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if axis == "y":
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def bar_labels(ax, bars, fmt="{:.3f}") -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=T["ink2"])


def bars(ax, x, h, w, color, label=None):
    # 2px surface gap between adjacent bars comes from the edge colour.
    return ax.bar(x, h, w, color=color, label=label,
                  edgecolor=T["surface"], linewidth=1.5)


def save(fig, name: str, theme: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}-{theme}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


# ------------------------------------------------ 1. construction types ---
def fig_constructions(theme: str) -> None:
    human, machine = T["series"][0], T["series"][1]
    rows = [
        ("Type 1 · continuation", [0, 0, 0, 0, 1, 1, 1, 1],
         "1,675 documents · 1 boundary · 49.4% machine sentences"),
        ("Type 2 · single-span replacement", [0, 0, 0, 1, 1, 0, 0, 0],
         "1,367 documents · 2 boundaries · 24.5% machine sentences"),
        ("Type 3 · multi-span replacement", [0, 0, 1, 0, 1, 1, 0, 0],
         "1,202 documents · 4.97 boundaries on average · 33.0% machine"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.set_xlim(-0.2, 8.1)
    ax.set_ylim(-0.75, len(rows) * 1.55)
    ax.axis("off")
    bw, bh = 0.88, 0.5
    for r, (name, labels, stats) in enumerate(rows):
        y = (len(rows) - 1 - r) * 1.55
        ax.text(0, y + bh + 0.42, name, fontsize=10.5, fontweight="bold",
                color=T["ink"], va="bottom")
        ax.text(0, y + bh + 0.1, stats, fontsize=9, color=T["ink2"],
                va="bottom")
        for i, lab in enumerate(labels):
            ax.add_patch(FancyBboxPatch(
                (i, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=machine if lab else human, edgecolor=T["surface"],
                linewidth=1.5))
            # letter = secondary encoding, so identity is not colour alone
            ax.text(i + bw / 2, y + bh / 2, "M" if lab else "H",
                    color="#ffffff", fontsize=9, fontweight="bold",
                    ha="center", va="center")
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                ax.plot([i - 0.06] * 2, [y - 0.12, y + bh + 0.12],
                        color=T["ink"], linewidth=2, solid_capstyle="round")
    ax.text(0, -0.62, "H  human sentence      M  machine sentence      "
            "|  authorship boundary", fontsize=9, color=T["ink2"])
    save(fig, "fig1-constructions", theme)


# ---------------------------------------------------- 2. model comparison ---
def fig_models(theme: str) -> None:
    lin = load("baselines/linear.json")
    xl = load("xlmr/xlmr.json")

    def row(name):
        r = next(x for x in lin["results"] if x["name"] == name)
        return r["overall"]["sentence"]["f1_ai"], r["overall"]["boundary_exact"]["f1"]

    models = [
        ("Position only\n(reads no text)", *row("baseline: position only (no text)")),
        ("Linear\nchar n-grams", *row("text + context")),
        ("Linear\n+ likelihood", *row("text + context + likelihood")),
        ("XLM-R\ntagger", xl["test_threshold"]["overall"]["sentence"]["f1_ai"],
         xl["test_threshold"]["overall"]["boundary_exact"]["f1"]),
    ]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    b1 = bars(ax, x - w / 2, [m[1] for m in models], w, T["series"][0],
              "Sentence F1 (machine class)")
    b2 = bars(ax, x + w / 2, [m[2] for m in models], w, T["series"][1],
              "Exact-boundary F1 (primary)")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in models])
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("F1 on the test set")
    ax.legend(loc="upper left", ncols=2)
    recessive(ax)
    save(fig, "fig2-models", theme)


# ------------------------------------ 3. held-out generator, construction ---
def fig_breakdown(theme: str) -> None:
    xl = load("xlmr/xlmr.json")
    # Threshold decoding: the reported headline (Viterbi ties it on
    # exact-boundary F1 and loses on the other metrics).
    t = xl["test_threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    w = 0.36

    metrics = [("Sentence F1", "sentence", "f1_ai"),
               ("Exact-boundary F1", "boundary_exact", "f1"),
               ("Documents fully correct", "boundary_exact", "exact_doc_match")]
    x = np.arange(len(metrics))
    ax = axes[0]
    b1 = bars(ax, x - w / 2, [t["seen"][a][b] for _, a, b in metrics], w,
              T["series"][0], "Seen generators")
    b2 = bars(ax, x + w / 2, [t["held_out"][a][b] for _, a, b in metrics], w,
              T["series"][1], "Held-out (Gemini 2.5 Pro)")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Unseen generator", loc="left")
    ax.legend(loc="upper right")
    recessive(ax)

    order = [("Continuation", "type1_single_boundary"),
             ("Single span", "type2_single_internal_segment"),
             ("Multi span", "type3_multiple_internal_segments")]
    per = xl["per_construction_type"]
    x = np.arange(len(order))
    ax = axes[1]
    b1 = bars(ax, x - w / 2, [per[k]["sentence"]["f1_ai"] for _, k in order],
              w, T["series"][0], "Sentence F1")
    b2 = bars(ax, x + w / 2, [per[k]["boundary_exact"]["f1"] for _, k in order],
              w, T["series"][1], "Exact-boundary F1")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([o[0] for o in order], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Construction type", loc="left")
    ax.legend(loc="upper right")
    recessive(ax)
    fig.tight_layout(w_pad=3)
    save(fig, "fig3-breakdown", theme)


# ------------------------------------------------- 4. twin trade-off ---
def fig_tradeoff(theme: str) -> None:
    tr = load("twins/twin_tradeoff_laptop.json")
    runs = [  # (key, label) in fixed palette order
        ("base", "XLM-R baseline"),
        ("twin_ft", "twins, fine-tuned"),
        ("twin_plain", "twins, no paired terms"),
        ("twin_warm", "twins + paired terms"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for (key, label), color in zip(runs, T["series"]):
        c = sorted(tr[key]["test_curve"], key=lambda r: r["threshold"])
        xs = [100 * r["false_alarm"] for r in c]
        ys = [r["mixed_bE"] for r in c]
        ax.plot(xs, ys, color=color, linewidth=2, label=label, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")
        # direct label just left of the curve's lowest-false-alarm point
        i = int(np.argmin(xs))
        ax.plot(xs[i], ys[i], "o", ms=5, color=color, zorder=4)
        ax.annotate(label, (xs[i], ys[i]), xytext=(-9, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=9, color=T["ink2"])
    ax.set_xlim(0, 102)
    ax.set_ylim(0.44, 0.63)
    ax.set_xlabel("Human documents with a false alarm (%)  ← fewer is better")
    ax.set_ylabel("Exact-boundary F1, mixed documents")
    ax.legend(loc="lower right", ncols=2)
    recessive(ax, axis="both")
    ax.spines["left"].set_visible(True)
    save(fig, "fig4-twin-tradeoff", theme)


# ----------------------------------------------------- 5. three seeds ---
def fig_seeds(theme: str) -> None:
    """Each seed as a dot, the mean as a large marker with a +/- 1 sd range.

    The F1 panels need a zoomed axis to show seed spread, and bars on a
    truncated axis exaggerate differences, so every panel uses dots.
    """
    s = load("twins/twin_ft_seeds.json")["metrics"]
    panels = [
        ("twin false-alarm rate", "Human documents with\na false alarm (%)",
         100, "{:.1f}%", (0, 105)),
        ("exact-boundary F1, mixed + twins",
         "Exact-boundary F1,\nmixed + human documents", 1, "{:.3f}", None),
        ("exact-boundary F1, mixed (Viterbi)",
         "Exact-boundary F1,\nmixed documents only", 1, "{:.3f}", None),
    ]
    groups = [("control", "More training\n(control)", T["series"][0]),
              ("twin_ft", "More training\nwith twins", T["series"][1])]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6))
    for ax, (key, title, scale, fmt, ylim) in zip(axes, panels):
        for j, (g, _, color) in enumerate(groups):
            vals = scale * np.array(s[key][g])
            m, sd = vals.mean(), vals.std(ddof=1)
            ax.plot([j, j], [m - sd, m + sd], color=color, linewidth=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(j, m, "o", ms=11, color=color,
                    markeredgecolor=T["surface"], markeredgewidth=2, zorder=4)
            ax.scatter(np.full(len(vals), j + 0.2), vals, s=24,
                       facecolor=T["surface"], edgecolor=color, linewidth=1.4,
                       zorder=3)
            ax.annotate(fmt.format(m), (j, m), xytext=(-12, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=9, color=T["ink2"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([g[1] for g in groups], fontsize=9)
        ax.set_xlim(-0.75, 1.5)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            lo = min(scale * min(s[key][g]) for g, _, _ in groups)
            hi = max(scale * max(s[key][g]) for g, _, _ in groups)
            pad = (hi - lo) * 0.35
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, loc="left", fontsize=10)
        recessive(ax)
    fig.tight_layout(w_pad=2.5)
    fig.text(0.01, -0.03, "Large dot: mean of 3 seeds  ·  line: ±1 standard "
             "deviation  ·  small dots: individual seeds", fontsize=8.5,
             color=T["ink2"])
    save(fig, "fig5-seeds", theme)


# ----------------------------------------------- 6. four conditions ---
def fig_conditions(theme: str) -> None:
    """Is it the twin, or any human text? Same encoding as fig5."""
    s = load("twins/four_conditions.json")["metrics"]
    panels = [
        ("unrelated-human false-alarm rate",
         "Unrelated human documents with\na false alarm, threshold (%)", 100,
         "{:.0f}%", (0, 105)),
        ("unrelated-human false-alarm rate (Viterbi)",
         "Unrelated human documents with\na false alarm, Viterbi (%)", 100,
         "{:.0f}%", (0, 105)),
        ("exact-boundary F1, mixed (Viterbi)",
         "Exact-boundary F1,\nmixed documents only", 1, "{:.3f}", None),
    ]
    groups = [("control", "no human\ndocuments"),
              ("unrelated", "unrelated\nhuman"),
              ("twins_unpaired", "twins,\nCE only"),
              ("twins_paired", "twins +\npaired terms")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8))
    for ax, (key, title, scale, fmt, ylim) in zip(axes, panels):
        for j, ((g, _), color) in enumerate(zip(groups, T["series"])):
            vals = scale * np.array(s[key][g])
            m, sd = vals.mean(), vals.std(ddof=1)
            ax.plot([j, j], [m - sd, m + sd], color=color, linewidth=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(j, m, "o", ms=10, color=color,
                    markeredgecolor=T["surface"], markeredgewidth=2, zorder=4)
            ax.scatter(np.full(len(vals), j + 0.22), vals, s=20,
                       facecolor=T["surface"], edgecolor=color, linewidth=1.3,
                       zorder=3)
            ax.annotate(fmt.format(m), (j, m), xytext=(-9, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8.5, color=T["ink2"])
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[1] for g in groups], fontsize=8.5)
        ax.set_xlim(-0.8, len(groups) - 0.4)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            lo = min(scale * min(s[key][g]) for g, _ in groups)
            hi = max(scale * max(s[key][g]) for g, _ in groups)
            pad = (hi - lo) * 0.35
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, loc="left", fontsize=10)
        recessive(ax)
    fig.tight_layout(w_pad=2.0)
    fig.text(0.01, -0.03, "Large dot: mean of 3 seeds  ·  line: ±1 standard "
             "deviation  ·  small dots: individual seeds", fontsize=8.5,
             color=T["ink2"])
    save(fig, "fig6-conditions", theme)


def main() -> None:
    for theme in THEMES:
        style(theme)
        fig_constructions(theme)
        fig_models(theme)
        fig_breakdown(theme)
        fig_tradeoff(theme)
        fig_seeds(theme)
        fig_conditions(theme)


if __name__ == "__main__":
    main()
