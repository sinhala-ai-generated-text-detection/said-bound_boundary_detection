"""Balance and boundary-position audit.

The headline risk this catches: boundary positions spiking near the start of
the document. If they do, a classifier can win by guessing position rather than
detecting a stylistic change, and the dataset is worthless. The plot and the
per-decile table exist to make that visible.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import force_utf8_stdout, load_config, read_jsonl  # noqa: E402


def _hist(ax, values, title, xlabel, bins=20, color="#4C72B0"):
    ax.hist(values, bins=bins, range=(0, 1), color=color, edgecolor="white")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)


def run(cfg: dict) -> dict:
    force_utf8_stdout()
    gdir = Path(cfg["paths"]["generated_dir"])
    rdir = Path(cfg["paths"]["reports_dir"])
    rdir.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(gdir / "combined.jsonl"))
    if not records:
        raise SystemExit("No records in generated/combined.jsonl; "
                         "run src/build_dataset.py first.")

    lines: list[str] = ["# Dataset audit", ""]
    lines.append(f"Total records: **{len(records)}**")
    lines.append(f"Unique source articles: **{len({r['source_id'] for r in records})}**")
    lines.append("")

    # ---- counts by domain x generator x type ------------------------------
    cube: dict = defaultdict(Counter)
    for r in records:
        cube[(r["domain"], r["generator"])][r["construction_type"]] += 1

    types = ["type1_single_boundary", "type2_single_internal_segment",
             "type3_multiple_internal_segments"]
    lines.append("## Counts by domain x generator x construction type")
    lines.append("")
    lines.append("| domain | generator | role | " + " | ".join(
        t.replace("_", " ") for t in types) + " | total |")
    lines.append("|---|---|---|" + "---|" * (len(types) + 1))
    for (domain, gen), counts in sorted(cube.items()):
        role = cfg["generators"].get(gen, {}).get("role", "?")
        row = [str(counts.get(t, 0)) for t in types]
        lines.append(f"| {domain} | {gen} | {role} | " + " | ".join(row)
                     + f" | {sum(counts.values())} |")
    lines.append("")

    # ---- split balance ----------------------------------------------------
    lines.append("## Split balance")
    lines.append("")
    lines.append("| generator | role | train | dev | test |")
    lines.append("|---|---|---|---|---|")
    by_gs: dict = defaultdict(Counter)
    for r in records:
        by_gs[r["generator"]][r["split"]] += 1
    for gen in sorted(by_gs):
        role = cfg["generators"].get(gen, {}).get("role", "?")
        c = by_gs[gen]
        lines.append(f"| {gen} | {role} | {c.get('train',0)} | "
                     f"{c.get('dev',0)} | {c.get('test',0)} |")
    lines.append("")
    held_out_in_train = [g for g in by_gs
                         if cfg["generators"].get(g, {}).get("role") == "held_out"
                         and by_gs[g].get("train", 0) > 0]
    lines.append(f"Held-out generators appearing in train: "
                 f"**{held_out_in_train or 'none (correct)'}**")
    lines.append("")

    # ---- label balance ----------------------------------------------------
    total_sent = sum(len(r["labels"]) for r in records)
    ai_sent = sum(sum(r["labels"]) for r in records)
    lines.append("## Sentence-level label balance")
    lines.append("")
    lines.append(f"- Total sentences: **{total_sent}**")
    lines.append(f"- AI sentences: **{ai_sent}** ({ai_sent/total_sent:.1%})")
    lines.append(f"- Human sentences: **{total_sent-ai_sent}** "
                 f"({1-ai_sent/total_sent:.1%})")
    lines.append("")

    # ---- boundary position distribution -----------------------------------
    first_bounds = [r["boundary_position_normalized"] for r in records
                    if r.get("boundary_position_normalized") is not None]
    all_bounds = [b for r in records
                  for b in (r.get("boundary_positions_normalized") or [])]

    lines.append("## Boundary-position distribution")
    lines.append("")
    lines.append("Normalised position of the first boundary, by decile. A "
                 "healthy dataset is spread across deciles; a spike in the "
                 "first decile means position alone predicts the boundary.")
    lines.append("")
    deciles = Counter(min(9, int(b * 10)) for b in first_bounds)
    lines.append("| decile | range | count | share |")
    lines.append("|---|---|---|---|")
    for d in range(10):
        n = deciles.get(d, 0)
        share = n / len(first_bounds) if first_bounds else 0
        bar = "#" * int(share * 50)
        lines.append(f"| {d} | {d/10:.1f}-{(d+1)/10:.1f} | {n} | "
                     f"{share:.1%} {bar} |")
    lines.append("")
    if first_bounds:
        srt = sorted(first_bounds)
        p = lambda q: srt[min(len(srt) - 1, int(len(srt) * q))]  # noqa: E731
        lines.append(f"- min={srt[0]:.3f} p25={p(.25):.3f} p50={p(.50):.3f} "
                     f"p75={p(.75):.3f} max={srt[-1]:.3f}")
        spike = deciles.get(0, 0) / len(first_bounds)
        verdict = ("OK - spread across the passage" if spike < 0.25
                   else "WARNING - spiked near the start")
        lines.append(f"- First-decile share: {spike:.1%} -> **{verdict}**")
    lines.append("")

    # ---- span positions for types 2/3 -------------------------------------
    span_mid: list[float] = []
    for r in records:
        if r["construction_type"] == "type1_single_boundary":
            continue
        n = len(r["labels"])
        for s, e in (r.get("spans") or []):
            span_mid.append(((s + e) / 2) / n)
    lines.append("## Span-position distribution (Types 2 and 3)")
    lines.append("")
    if span_mid:
        sd = Counter(min(9, int(x * 10)) for x in span_mid)
        lines.append("| decile | count |")
        lines.append("|---|---|")
        for d in range(10):
            lines.append(f"| {d} | {sd.get(d,0)} |")
    else:
        lines.append("_No Type 2/3 records yet._")
    lines.append("")

    # ---- retries ----------------------------------------------------------
    retries = Counter(r.get("retry_count", 0) for r in records)
    lines.append("## Retry counts")
    lines.append("")
    lines.append("| retries | records |")
    lines.append("|---|---|")
    for k in sorted(retries):
        lines.append(f"| {k} | {retries[k]} |")
    lines.append("")

    # ---- plots ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _hist(axes[0], first_bounds, "First boundary position (all types)",
          "normalised position")
    _hist(axes[1], all_bounds, "All boundary positions", "normalised position",
          color="#DD8452")
    if span_mid:
        _hist(axes[2], span_mid, "Span midpoints (Types 2/3)",
              "normalised position", color="#55A868")
    else:
        axes[2].set_visible(False)
    fig.suptitle("Boundary-position audit", fontsize=12)
    fig.tight_layout()
    plot_path = rdir / "boundary_positions.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)

    # per-generator boundary spread
    gens = sorted({r["generator"] for r in records})
    if gens:
        fig, axes = plt.subplots(1, len(gens), figsize=(5 * len(gens), 3.6),
                                 squeeze=False)
        for ax, g in zip(axes[0], gens):
            vals = [r["boundary_position_normalized"] for r in records
                    if r["generator"] == g
                    and r.get("boundary_position_normalized") is not None]
            _hist(ax, vals, f"{g} (n={len(vals)})", "normalised position")
        fig.suptitle("First-boundary position by generator", fontsize=12)
        fig.tight_layout()
        gen_plot = rdir / "boundary_positions_by_generator.png"
        fig.savefig(gen_plot, dpi=130)
        plt.close(fig)
        lines.append(f"![by generator]({gen_plot.name})")

    lines.insert(2, f"![boundary positions]({plot_path.name})\n")

    out = rdir / "audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 70)
    print("AUDIT")
    print("=" * 70)
    print(f"  records      {len(records)}")
    print(f"  AI sentences {ai_sent}/{total_sent} ({ai_sent/total_sent:.1%})")
    if first_bounds:
        print(f"  first-decile boundary share {deciles.get(0,0)/len(first_bounds):.1%}")
    print(f"  wrote -> {out}")
    print(f"  wrote -> {plot_path}")
    return {"records": len(records)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    run(load_config(a.config))


if __name__ == "__main__":
    main()
