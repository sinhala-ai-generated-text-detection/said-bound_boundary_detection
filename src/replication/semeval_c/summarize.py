"""Mean ± sd across seeds, and seed-paired differences between conditions.

The logic of src/detect/summarize_seeds.py applied to the Subtask C runs:
groups list their runs in seed order, and each comparison pairs runs by
position (same seed), so noise both runs share cancels out.

    python src/replication/semeval_c/summarize.py \\
        A=deberta_control_s42,deberta_control_s43,deberta_control_s44 \\
        B=deberta_human_s42,... C=deberta_twins_s42,... \\
        --pairs B-A C-A C-B --out results/replication/semeval_c/seeds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "detect"))
from summarize_seeds import fmt  # noqa: E402

RESULTS = Path("results/replication/semeval_c")

FW, CP = "first_word", "change_point"
# (label, path into the run JSON, higher is better, scale for display)
METRICS = [
    ("test MAE", (FW, "test", "overall", "mae"), False, 1),
    ("test exact-boundary accuracy", (FW, "test", "overall", "exact"), True, 1),
    ("test MAE, PeerRead", (FW, "test", "by_domain", "peerread", "mae"), False, 1),
    ("test MAE, OUTFOX", (FW, "test", "by_domain", "outfox", "mae"), False, 1),
    ("false alarm, ASAP reviews (%)",
     (FW, "human", "eval_asap", "false_alarm"), False, 100),
    ("false alarm, OUTFOX essays (%)",
     (FW, "human", "eval_outfox", "false_alarm"), False, 100),
    ("false alarm, test twins PeerRead (%)",
     (FW, "human", "eval_twins_peerread", "false_alarm"), False, 100),
    ("false alarm, test twins OUTFOX (%)",
     (FW, "human", "eval_twins_outfox", "false_alarm"), False, 100),
    ("words flagged, ASAP reviews (%)",
     (FW, "human", "eval_asap", "frac_flagged"), False, 100),
    ("words flagged, OUTFOX essays (%)",
     (FW, "human", "eval_outfox", "frac_flagged"), False, 100),
    ("presence accuracy, mixed + human (balanced)",
     (FW, "deployment", "balanced_accuracy"), True, 1),
    ("change point: test MAE", (CP, "test", "overall", "mae"), False, 1),
    ("change point: false alarm, ASAP (%)",
     (CP, "human", "eval_asap", "false_alarm"), False, 100),
    ("change point: false alarm, OUTFOX (%)",
     (CP, "human", "eval_outfox", "false_alarm"), False, 100),
    ("change point: false alarm, twins PeerRead (%)",
     (CP, "human", "eval_twins_peerread", "false_alarm"), False, 100),
]


def _get(d, path):
    for k in path:
        d = d[k]
    return float(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="name=run1,run2,...")
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="comparisons as X-Y (X minus Y)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    groups, runs = {}, {}
    for g in a.groups:
        name, rs = g.split("=", 1)
        runs[name] = rs.split(",")
        rows = [json.loads((RESULTS / f"{r}.json").read_text()) for r in runs[name]]
        groups[name] = {lbl: [s * _get(x, p) for x in rows]
                        for lbl, p, _, s in METRICS}
        groups[name]["_seeds"] = [x["seed"] for x in rows]
    pairs = [p.split("-") for p in a.pairs]
    for x, y in pairs:
        assert groups[x]["_seeds"] == groups[y]["_seeds"], (x, y)

    names = list(groups)
    L = ["# Subtask C: results across seeds", "",
         "Test set, mean ± sample standard deviation over seeds. The first-"
         "word rule at threshold 0.5 unless labelled *change point*. False "
         "alarms and flagged words in percent.", ""]
    for n in names:
        L.append(f"- **{n}**: " + ", ".join(
            f"`{r}`" for r in runs[n]))
    L += ["", "| metric | " + " | ".join(names) + " |",
          "|---|" + "---|" * len(names)]
    for lbl, *_ in METRICS:
        L.append(f"| {lbl} | " + " | ".join(fmt(groups[n][lbl]) for n in names)
                 + " |")
    summary = {"groups": runs, "metrics": {
        lbl: {n: groups[n][lbl] for n in names} for lbl, *_ in METRICS},
        "paired": {}}
    if pairs:
        L += ["", "## Seed-paired differences", "",
              "Each cell: mean ± sd of the per-seed difference, and the seeds "
              "on which the first condition is better.", "",
              "| metric | " + " | ".join(f"{x} − {y}" for x, y in pairs) + " |",
              "|---|" + "---|" * len(pairs)]
        for lbl, _, higher, _ in METRICS:
            cells = []
            for x, y in pairs:
                d = np.array(groups[x][lbl]) - np.array(groups[y][lbl])
                wins = int(((d > 0) if higher else (d < 0)).sum())
                cells.append(f"{fmt(d)} ({wins}/{len(d)})")
                summary["paired"].setdefault(f"{x}-{y}", {})[lbl] = d.tolist()
            L.append(f"| {lbl} | " + " | ".join(cells) + " |")
    L.append("")
    out = Path(a.out)
    out.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
