"""Operating-point analysis: mixed-document accuracy against false alarms.

A single tuned threshold compares models at whatever operating point the
selection rule happened to pick, and the rule used in run_transformer.py sees
only mixed documents, so it cannot reward a model for staying quiet on human
text. This sweeps the decision threshold for every saved model and reports:

* the full test trade-off curve (mixed exact-boundary F1 vs twin false alarm),
* each model at matched false-alarm levels, and
* a *deployment-aware* selection: threshold chosen on dev-seen mixed documents
  plus their twins, scored as exact-boundary F1 over both together.

Inference only; nothing is trained.

    python src/detect/twin_tradeoff.py base=models/xlmr_tagger \\
        twin_plain=models/xlmr_twin_plain twin_warm=models/xlmr_twin_warm
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_dataset, load_twins, role_map  # noqa: E402
from metrics import boundary_metrics, twin_metrics  # noqa: E402
from run_transformer import _per_doc  # noqa: E402
from utils import force_utf8_stdout, load_config  # noqa: E402

GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)
ALARM_LEVELS = (0.3, 0.5, 0.7)


def _unique(twins, probs):
    seen, out = set(), []
    for t, p in zip(twins, probs):
        if t.window_id not in seen:
            seen.add(t.window_id)
            out.append((t, p))
    return out


def sweep(docs, twins, pm, pt):
    """Metrics at every threshold, from cached per-document P(AI)."""
    uniq = _unique(twins, pt)
    rows = []
    for thr in GRID:
        ym = [(np.asarray(p) >= thr).astype(int).tolist() for p in pm]
        yt = [(np.asarray(p) >= thr).astype(int).tolist() for p in pt]
        mixed = boundary_metrics([list(d.labels) for d in docs], ym, 0)
        tm = twin_metrics(docs, twins, ym, yt, pm, pt)
        rows.append({
            "threshold": float(thr),
            "mixed_bE": mixed.f1,
            "mixed_doc_exact": mixed.exact_doc_match,
            "false_alarm": tm["doc_false_alarm"],
            "boundaries_per_twin": tm["boundaries_per_twin"],
            "sentence_fpr": tm["sentence_fpr"],
            "combined_bE": tm["combined_boundary_exact"]["f1"],
        })
    return rows, len(uniq)


def main() -> None:
    force_utf8_stdout()
    import torch  # noqa: F401
    from transformer import TransformerDetector

    runs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    if not runs:
        raise SystemExit(__doc__)
    cfg = load_config()
    rm = role_map(cfg)
    ds = load_dataset(cfg)
    dev = ds.filter(split="dev").filter(roles=["seen"], role_map=rm)
    test = ds.filter(split="test")
    dev_tw, _ = load_twins(dev.docs, cfg)
    test_tw, _ = load_twins(test.docs, cfg)
    held = [i for i, d in enumerate(test.docs)
            if rm.get(d.generator) == "held_out"]

    out = {}
    for name, path in runs.items():
        print(f"\n== {name} ({path})", flush=True)
        det = TransformerDetector(use_pair_head=False).load(path)
        P = {}
        for key, docs in (("dev", dev.docs), ("dev_tw", dev_tw),
                          ("test", test.docs), ("test_tw", test_tw)):
            P[key] = _per_doc(docs, det.predict_proba(docs), float)
        del det
        torch.cuda.empty_cache()

        dev_rows, _ = sweep(dev.docs, dev_tw, P["dev"], P["dev_tw"])
        test_rows, n_tw = sweep(test.docs, test_tw, P["test"], P["test_tw"])
        held_rows, _ = sweep([test.docs[i] for i in held],
                             [test_tw[i] for i in held],
                             [P["test"][i] for i in held],
                             [P["test_tw"][i] for i in held])

        def pick(key):
            j = int(np.argmax([r[key] for r in dev_rows]))
            return {"threshold": dev_rows[j]["threshold"],
                    "dev": dev_rows[j], "test": test_rows[j],
                    "held_out": held_rows[j]}

        matched = {}
        for lvl in ALARM_LEVELS:
            ok = [r for r in test_rows if r["false_alarm"] <= lvl]
            matched[str(lvl)] = (max(ok, key=lambda r: r["mixed_bE"])
                                 if ok else None)
        out[name] = {
            "path": path, "n_test_twins": n_tw,
            "select_mixed": pick("mixed_bE"),
            "select_deploy": pick("combined_bE"),
            "matched_false_alarm": matched,
            "test_curve": test_rows, "held_out_curve": held_rows,
            "dev_curve": dev_rows,
        }
        for lbl, key in (("mixed-only selection", "select_mixed"),
                         ("deployment selection", "select_deploy")):
            s = out[name][key]
            t, h = s["test"], s["held_out"]
            print(f"  {lbl:21s} thr={s['threshold']:.3f}  "
                  f"mixed bE={t['mixed_bE']:.3f} held bE={h['mixed_bE']:.3f} "
                  f"false-alarm={t['false_alarm']:.3f} "
                  f"bnd/twin={t['boundaries_per_twin']:.2f} "
                  f"bE+twins={t['combined_bE']:.3f} "
                  f"held bE+twins={h['combined_bE']:.3f}")
        for lvl, r in matched.items():
            print(f"  best mixed bE at false-alarm <= {lvl}: "
                  + (f"{r['mixed_bE']:.3f} (thr {r['threshold']:.3f}, "
                     f"alarm {r['false_alarm']:.3f})" if r else "unreachable"))

    Path("results/detection/twins/twin_tradeoff.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    write_report(Path("results/detection/twins/twin_tradeoff.md"), out)
    print("\nwrote -> results/detection/twins/twin_tradeoff.{md,json}")


def write_report(path: Path, out: dict) -> None:
    L = ["# Operating-point analysis: counterfactual twins", "",
         "Every model's decision threshold is swept over "
         f"{GRID[0]:.2f}-{GRID[-1]:.2f} (threshold decoding). *Mixed bE* is "
         "exact-boundary F1 on the mixed test documents; *false alarm* is the "
         "share of all-human twins with any sentence flagged; *bE + twins* "
         "scores the mixed documents and the twins together, where every "
         "boundary predicted on a twin is a false positive.", "",
         "## Selected operating points", "",
         "*Mixed-only* picks the threshold with the best dev-seen exact-"
         "boundary F1 on mixed documents (the original protocol, on a wider "
         "grid). *Deployment* picks it on dev-seen mixed documents plus their "
         "twins. Neither selection sees test or the held-out generator.", "",
         "| model | selection | thr | mixed bE | held-out bE | false alarm | "
         "boundaries/twin | bE + twins | held-out bE + twins |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in out.items():
        for lbl, key in (("mixed-only", "select_mixed"),
                         ("deployment", "select_deploy")):
            s = r[key]
            t, h = s["test"], s["held_out"]
            L.append(f"| {name} | {lbl} | {s['threshold']:.3f} | "
                     f"{t['mixed_bE']:.3f} | {h['mixed_bE']:.3f} | "
                     f"{t['false_alarm']:.3f} | "
                     f"{t['boundaries_per_twin']:.2f} | "
                     f"{t['combined_bE']:.3f} | {h['combined_bE']:.3f} |")
    L += ["", "## Mixed-document accuracy at matched false-alarm rates", "",
          "Best test mixed exact-boundary F1 among thresholds whose twin "
          "false-alarm rate is at most the given level. This is read off the "
          "test curve, so it describes the trade-off each model offers rather "
          "than a tuned result.", "",
          "| model | " + " | ".join(f"alarm <= {l}" for l in ALARM_LEVELS)
          + " |", "|---|" + "---|" * len(ALARM_LEVELS)]
    for name, r in out.items():
        cells = []
        for lvl in ALARM_LEVELS:
            m = r["matched_false_alarm"][str(lvl)]
            cells.append(f"{m['mixed_bE']:.3f} (thr {m['threshold']:.2f})"
                         if m else "unreachable")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
