"""How much of each detector's score rests on formatting cues?

Human Wikipedia sentences carry typographical noise that generator output
lacks (docs/limitations.md). run_transformer.py --normalize-surface re-scores a
saved model with those cues removed from every sentence of both classes
(data.normalize_surface), re-tuning threshold and bias on the normalized dev
set. This script puts each pair of runs side by side, and re-measures the
feature table on the test set before and after normalization to show the cues
are gone.

Inference results only; run the --eval-only commands first.

    python src/detect/surface_noise.py \\
        base=xlmr_rescored_spark:xlmr_normsurface \\
        control=ft_control_s42:ft_control_s42_normsurface \\
        --out results/detection/surface_noise
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import FINAL_PUNCT, load_dataset, normalize_surface  # noqa: E402
from utils import force_utf8_stdout, load_config  # noqa: E402

RESULTS = Path("results/detection/twins")

# The feature table of docs/limitations.md. These definitions reproduce its
# numbers on the full corpus.
FEATURES = {
    "space before punctuation": lambda s: bool(re.search(r"\s[.,?!:;]", s)),
    "parentheses": lambda s: "(" in s,
    "no final punctuation": lambda s: not s.rstrip().endswith(FINAL_PUNCT),
    "Latin characters": lambda s: bool(re.search(r"[A-Za-z]", s)),
    "zero-width non-joiner": lambda s: "‌" in s,
    "curly quotes": lambda s: bool(re.search("[‘’“”]", s)),
}
CHANGED = ("space before punctuation", "no final punctuation",
           "zero-width non-joiner", "curly quotes")

# (label, path into the result JSON, higher is better)
METRICS = [
    ("sentence F1 (Viterbi)",
     ("test_viterbi", "overall", "sentence", "f1_ai"), True),
    ("sentence F1 (threshold)",
     ("test_threshold", "overall", "sentence", "f1_ai"), True),
    ("exact-boundary F1, mixed (Viterbi)",
     ("test_viterbi", "overall", "boundary_exact", "f1"), True),
    ("exact-boundary F1, mixed (threshold)",
     ("test_threshold", "overall", "boundary_exact", "f1"), True),
    ("exact-boundary F1, held-out (Viterbi)",
     ("test_viterbi", "held_out", "boundary_exact", "f1"), True),
    ("twin false-alarm rate",
     ("test_twins", "threshold", "overall", "doc_false_alarm"), False),
    ("unrelated-human false-alarm rate",
     ("test_unrelated_human", "threshold", "doc_false_alarm"), False),
    ("sentence FPR, unrelated human",
     ("test_unrelated_human", "threshold", "sentence_fpr"), False),
]


def feature_table(docs, norm: bool) -> dict:
    out = {}
    for y, cls in ((0, "human"), (1, "machine")):
        sents = [normalize_surface(s) if norm else s
                 for d in docs for s, lab in zip(d.sentences, d.labels)
                 if lab == y]
        out[cls] = {"n": len(sents),
                    **{k: sum(f(s) for s in sents) / len(sents)
                       for k, f in FEATURES.items()}}
    return out


def _get(d, path):
    for k in path:
        d = d[k]
    return float(d)


def main() -> None:
    force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="name=raw_run:normalized_run")
    ap.add_argument("--out", required=True, help="output path without suffix")
    a = ap.parse_args()

    test = load_dataset(load_config()).filter(split="test")
    feats = {"raw": feature_table(test.docs, False),
             "normalized": feature_table(test.docs, True)}

    models = {}
    for arg in a.pairs:
        name, runs = arg.split("=", 1)
        raw, norm = runs.split(":")
        R = [json.loads((RESULTS / f"{r}.json").read_text(encoding="utf-8"))
             for r in (raw, norm)]
        assert not R[0].get("normalize_surface") and R[1]["normalize_surface"]
        models[name] = {"raw_run": raw, "normalized_run": norm, "metrics": {
            lbl: {"raw": _get(R[0], p), "normalized": _get(R[1], p),
                  "change": _get(R[1], p) - _get(R[0], p)}
            for lbl, p, _ in METRICS}}

    L = ["# Surface noise: scores with formatting cues removed", "",
         "Each model is scored twice on the test set: as is, and after "
         "`data.normalize_surface` has been applied to every sentence of "
         "every document (both classes; dev, test, twins and unrelated human "
         "documents). Normalization removes whitespace before `. , ? ! : ;`, "
         "removes U+200B, U+200C (ZWNJ), U+2060, U+FEFF and U+00AD, maps "
         "curly quotes to straight ones, and appends `.` to a sentence that "
         "does not end in `. ? ! ෴ …`. U+200D (ZWJ) is kept because Sinhala "
         "conjuncts are spelled with it. **Parentheses and Latin text are "
         "left alone**: they are content, not formatting, so their imbalance "
         "remains. The threshold and Viterbi bias are re-tuned on the "
         "normalized dev set; the models are not retrained.", "",
         "## The cues before and after normalization", "",
         f"Test set, {feats['raw']['human']['n']:,} human and "
         f"{feats['raw']['machine']['n']:,} machine sentences.", "",
         "| feature | human, raw | machine, raw | human, normalized | "
         "machine, normalized |", "|---|---|---|---|---|"]
    for k in FEATURES:
        cells = [f"{feats[v][c][k]:.2%}" for v in ("raw", "normalized")
                 for c in ("human", "machine")]
        mark = "" if k in CHANGED else " (not normalized)"
        L.append(f"| {k}{mark} | " + " | ".join(cells) + " |")
    L += ["", "## Scores", "",
          "Each cell is raw → normalized (change). Twin and unrelated-human "
          "false alarms use threshold decoding.", ""]
    names = list(models)
    for title, group in (("Mixed documents", METRICS[:5]),
                         ("Human documents", METRICS[5:])):
        L += [f"### {title}", "",
              "| model | " + " | ".join(lbl for lbl, _, _ in group) + " |",
              "|---|" + "---|" * len(group)]
        for n in names:
            cells = []
            for lbl, _, _ in group:
                m = models[n]["metrics"][lbl]
                cells.append(f"{m['raw']:.3f} → {m['normalized']:.3f} "
                             f"({m['change']:+.3f})")
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Runs: " + "; ".join(
        f"{n}: `{m['raw_run']}` / `{m['normalized_run']}`"
        for n, m in models.items()), ""]

    out = Path(a.out)
    out.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"features_test": feats, "models": models}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
