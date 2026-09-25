"""Mean and spread across seeds, and paired differences between two methods.

Reads the result JSON that run_transformer.py writes to results/detection/twins/.
Each group lists its runs in seed order; when two groups are given, runs are
paired position by position (same seed), and the paired difference is the
headline: seed-to-seed noise that both methods share cancels out.

    python src/detect/summarize_seeds.py \\
        twin_ft=twin_ft,twin_ft_s43,twin_ft_s44 \\
        control=xlmr_ft,xlmr_ft_s43,xlmr_ft_s44 \\
        --out results/detection/twins/twin_ft_seeds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_config  # noqa: E402

RESULTS = Path("results/detection/twins")

# (label, path into the results JSON, higher is better)
METRICS = [
    ("exact-boundary F1, mixed (Viterbi)",
     ("test_viterbi", "overall", "boundary_exact", "f1"), True),
    ("exact-boundary F1, mixed, held-out (Viterbi)",
     ("test_viterbi", "held_out", "boundary_exact", "f1"), True),
    ("exact-boundary F1, mixed (threshold)",
     ("test_threshold", "overall", "boundary_exact", "f1"), True),
    ("sentence F1 (Viterbi)",
     ("test_viterbi", "overall", "sentence", "f1_ai"), True),
    ("twin false-alarm rate",
     ("test_twins", "threshold", "overall", "doc_false_alarm"), False),
    ("boundaries per twin",
     ("test_twins", "threshold", "overall", "boundaries_per_twin"), False),
    ("sentence FPR on twins",
     ("test_twins", "threshold", "overall", "sentence_fpr"), False),
    ("exact-boundary F1, mixed + twins",
     ("test_twins", "threshold", "overall", "combined_boundary_exact", "f1"),
     True),
    ("exact-boundary F1, mixed + twins, held-out",
     ("test_twins", "threshold", "held_out", "combined_boundary_exact", "f1"),
     True),
]


def _get(d, path):
    for k in path:
        d = d[k]
    return float(d)


def load(runs):
    # Runs from before --seed existed used the config seed and did not record it.
    default_seed = load_config()["seed"]
    out = []
    for r in runs:
        p = RESULTS / f"{r}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({lbl: _get(d, path) for lbl, path, _ in METRICS}
                   | {"_seed": d.get("seed", default_seed)})
    return out


def fmt(xs) -> str:
    xs = np.asarray(xs)
    sd = xs.std(ddof=1) if len(xs) > 1 else 0.0
    return f"{xs.mean():.3f} ± {sd:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="name=run1,run2,...")
    ap.add_argument("--out", required=True, help="output path without suffix")
    a = ap.parse_args()

    groups = {}
    for g in a.groups:
        name, runs = g.split("=", 1)
        groups[name] = (runs.split(","), load(runs.split(",")))
    names = list(groups)
    n = {len(v[0]) for v in groups.values()}

    L = ["# Results across seeds", "",
         "Test set. Mean ± sample standard deviation across seeds. Twin "
         "metrics use threshold decoding; mixed-only metrics are labelled by "
         "decoder.", ""]
    for name in names:
        runs, rows = groups[name]
        L.append(f"- **{name}**: " + ", ".join(
            f"`{r}` (seed {x['_seed']})" for r, x in zip(runs, rows)))
    L.append("")

    head = "| metric | " + " | ".join(names)
    div = "|---|" + "---|" * len(names)
    paired = len(names) == 2 and len(n) == 1
    if paired:
        head += f" | paired diff ({names[0]} − {names[1]}) | better in"
        div += "---|---|"
    L += [head + " |", div]
    summary = {}
    for lbl, _, higher in METRICS:
        cells = [fmt([x[lbl] for x in groups[g][1]]) for g in names]
        row = f"| {lbl} | " + " | ".join(cells)
        entry = {g: [x[lbl] for x in groups[g][1]] for g in names}
        if paired:
            diff = np.array(entry[names[0]]) - np.array(entry[names[1]])
            wins = int(((diff > 0) if higher else (diff < 0)).sum())
            row += f" | {fmt(diff)} | {wins}/{len(diff)} seeds"
            entry["paired_diff"] = diff.tolist()
        L.append(row + " |")
        summary[lbl] = entry
    L += ["", "*Better in* counts the seeds where the first method beats the "
          "second on that metric (lower is better for false alarms, "
          "boundaries per twin and FPR).", ""]

    out = Path(a.out)
    out.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"groups": {g: groups[g][0] for g in names},
                    "metrics": summary}, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
