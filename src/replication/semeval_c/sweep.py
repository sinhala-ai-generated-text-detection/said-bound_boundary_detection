"""Threshold sweep: Subtask C test MAE against false alarms on human text.

The English counterpart of src/detect/twin_tradeoff.py. For every saved run
it varies the per-word machine-probability threshold of the first-word rule
from 0.05 to 0.95 and records test MAE and the false-alarm rate on each human
set. The question is whether a model can be made quiet on human documents at
any threshold. Inference results only (data/english/preds/<tag>.pkl).

    python src/replication/semeval_c/sweep.py deberta_control_s42 ... \\
        --out results/replication/semeval_c/threshold_sweep
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import read, words_of  # noqa: E402
from run import EVAL_SETS, NAMES, PREDS  # noqa: E402

GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)
# Past the grid: if a model is still flagging human text at 0.95, does it
# go quiet at all before it stops finding machine text?
BEYOND = (0.975, 0.99, 0.995, 0.999)
LEVELS = (0.1, 0.3, 0.5)
MAE_LEVELS = (18, 20, 22, 25)


def first_word_curve(prob, thresholds) -> np.ndarray:
    """First-word boundary at every threshold at once (``prev`` rule).

    The first scored word at or above t is the first whose running maximum
    reaches t. Tokenless words before the first scored word inherit its
    label, so if that word is machine the boundary is 0.
    """
    n = len(prob)
    idx = np.flatnonzero(~np.isnan(prob))
    if len(idx) == 0:
        return np.full(len(thresholds), n)
    run = np.maximum.accumulate(prob[idx].astype(np.float64))
    k = np.searchsorted(run, np.asarray(thresholds, dtype=np.float64), "left")
    pos = np.append(idx, n)[k]
    return np.where(k == 0, 0, pos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preds-dir", default=str(PREDS))
    a = ap.parse_args()

    test = read("test")
    gold = np.array([min(d["boundary"], len(words_of(d["text"])))
                     for d in test])
    dom = np.array([d["domain"] for d in test])
    sets = {s: read(s) for s in EVAL_SETS}
    n_h = {s: np.array([len(words_of(d["text"])) for d in v])
           for s, v in sets.items()}

    out = {}
    for tag in a.tags:
        with open(Path(a.preds_dir) / f"{tag}.pkl", "rb") as fh:
            P = pickle.load(fh)
        B = np.stack([first_word_curve(p.astype(np.float32), GRID)
                      for p in P["test"]])            # docs x thresholds
        err = np.abs(B - gold[:, None])
        rows = []
        for j, t in enumerate(GRID):
            r = {"threshold": float(t), "mae": float(err[:, j].mean()),
                 "exact": float((err[:, j] == 0).mean()),
                 "mae_peerread": float(err[dom == "peerread", j].mean()),
                 "mae_outfox": float(err[dom == "outfox", j].mean())}
            rows.append(r)
        for s in EVAL_SETS:
            H = np.stack([first_word_curve(p.astype(np.float32), GRID)
                          for p in P[s]])
            alarm = (H < n_h[s][:, None]).mean(0)
            for j in range(len(GRID)):
                rows[j][f"fa_{s}"] = float(alarm[j])
        summary = {}
        for s in EVAL_SETS:
            fa = [r[f"fa_{s}"] for r in rows]
            summary[s] = {"lowest_fa": min(fa),
                          "at_threshold": rows[int(np.argmin(fa))]["threshold"],
                          "mae_there": rows[int(np.argmin(fa))]["mae"],
                          "best_mae_at": {}}
            for lvl in LEVELS:
                ok = [r for r in rows if r[f"fa_{s}"] <= lvl]
                summary[s]["best_mae_at"][str(lvl)] = (
                    min(ok, key=lambda r: r["mae"]) if ok else None)
        beyond = []
        for t in BEYOND:
            b = np.array([first_word_curve(p.astype(np.float32), [t])[0]
                          for p in P["test"]])
            r = {"threshold": t, "mae": float(np.abs(b - gold).mean())}
            for s in EVAL_SETS:
                h = np.array([first_word_curve(p.astype(np.float32), [t])[0]
                              for p in P[s]])
                r[f"fa_{s}"] = float((h < n_h[s]).mean())
            beyond.append(r)
        # Lowest false alarm among all thresholds (grid and beyond) whose test
        # MAE is at most a level: the two models compared at equal accuracy.
        pts = rows + beyond
        at_mae = {}
        for lvl in MAE_LEVELS:
            ok = [r for r in pts if r["mae"] <= lvl]
            at_mae[str(lvl)] = ({s: min(r[f"fa_{s}"] for r in ok)
                                 for s in EVAL_SETS} if ok else None)
        out[tag] = {"curve": rows, "summary": summary, "beyond_grid": beyond,
                    "fa_at_mae": at_mae,
                    "best_mae": min(r["mae"] for r in rows)}
        print(tag, {s: round(v["lowest_fa"], 3) for s, v in summary.items()},
              flush=True)

    o = Path(a.out)
    o.with_suffix(".json").write_text(json.dumps(out, indent=2))
    write_report(o.with_suffix(".md"), out)


def write_report(path: Path, out: dict) -> None:
    L = ["# Threshold sweep: Subtask C MAE against false alarms", "",
         f"First-word rule, threshold swept over {GRID[0]:.2f}–{GRID[-1]:.2f}. "
         "*False alarm*: share of human-only documents with any word at or "
         "above the threshold. Read off the test curves, so these describe "
         "the trade-off each model offers rather than a tuned result.", "",
         "## Lowest false-alarm rate reachable at any threshold", "",
         "| run | best test MAE | " + " | ".join(NAMES[s] for s in EVAL_SETS)
         + " |", "|---|---|" + "---|" * len(EVAL_SETS)]
    for tag, r in out.items():
        cells = [f"{r['summary'][s]['lowest_fa']:.3f} "
                 f"(thr {r['summary'][s]['at_threshold']:.2f}, MAE "
                 f"{r['summary'][s]['mae_there']:.1f})" for s in EVAL_SETS]
        L.append(f"| {tag} | {r['best_mae']:.2f} | " + " | ".join(cells) + " |")
    L += ["", "## Beyond the grid", "",
          "The same rule at thresholds above 0.95: test MAE, then false alarms "
          "on ASAP reviews / OUTFOX essays.", "",
          "| run | " + " | ".join(f"thr {t}" for t in BEYOND) + " |",
          "|---|" + "---|" * len(BEYOND)]
    for tag, r in out.items():
        L.append(f"| {tag} | " + " | ".join(
            f"{b['mae']:.1f}; {b['fa_eval_asap']:.2f} / "
            f"{b['fa_eval_outfox']:.2f}" for b in r["beyond_grid"]) + " |")
    L += ["", "## Lowest false alarm at a given test MAE", "",
          "Over every threshold (grid and beyond) whose test MAE is at most "
          "the level: false alarms on ASAP reviews / OUTFOX essays. This "
          "compares models at equal accuracy on the official metric.", "",
          "| run | " + " | ".join(f"MAE ≤ {l}" for l in MAE_LEVELS) + " |",
          "|---|" + "---|" * len(MAE_LEVELS)]
    for tag, r in out.items():
        cells = []
        for lvl in MAE_LEVELS:
            m = r["fa_at_mae"][str(lvl)]
            cells.append(f"{m['eval_asap']:.2f} / {m['eval_outfox']:.2f}"
                         if m else "unreachable")
        L.append(f"| {tag} | " + " | ".join(cells) + " |")
    for s in ("eval_asap", "eval_outfox"):
        L += ["", f"## Best test MAE at a false-alarm rate on {NAMES[s]}", "",
              "| run | " + " | ".join(f"≤ {l:.0%}" for l in LEVELS) + " |",
              "|---|" + "---|" * len(LEVELS)]
        for tag, r in out.items():
            cells = []
            for lvl in LEVELS:
                m = r["summary"][s]["best_mae_at"][str(lvl)]
                cells.append(f"{m['mae']:.1f} (thr {m['threshold']:.2f})"
                             if m else "unreachable")
            L.append(f"| {tag} | " + " | ".join(cells) + " |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
