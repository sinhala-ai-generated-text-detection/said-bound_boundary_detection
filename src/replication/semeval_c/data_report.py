"""Publish the data build report: counts and length statistics, no text.

    python src/replication/semeval_c/data_report.py \\
        --out results/replication/semeval_c/data_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BUILD = Path("data/english/processed/build_report.json")


def row(name, d):
    return (f"| {name} | {d['n']:,} | {d['mean']:.0f} | {d['p5']:.0f} | "
            f"{d['p25']:.0f} | {d['p50']:.0f} | {d['p75']:.0f} | "
            f"{d['p95']:.0f} | {d['total_words']:,} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.loads(BUILD.read_text())
    sc, tr = r["subtask_c"], r["tokenless_rule"]
    L = ["# English replication: data", "",
         "Counts and word-length statistics (words = `text.split(' ')`, the "
         "official Subtask C convention) for every set built by "
         "`src/replication/semeval_c/corpus.py`.", "",
         "## Subtask C (official)", "",
         f"Train {sc['train']:,}, dev {sc['dev']:,}, test {sc['test']:,}. "
         "Fully machine documents (boundary 0): " + ", ".join(
             f"{k} {v}" for k, v in sc["fully_machine"].items())
         + ". Boundary at or past the last word (no machine word): "
         + ", ".join(f"{k} {v}" for k, v in sc["boundary_at_end"].items())
         + ".", "",
         "| test domain / generator | docs |", "|---|---|"]
    for k, v in sorted(sc["test_by_domain_generator"].items()):
        L.append(f"| {k} | {v:,} |")
    L += ["", "## Words with no token", "",
          "Words that are empty or whitespace only get no subword token, so "
          "their label is inherited from a neighbour. The rule was chosen on "
          "train by the error it forces with every other word right:", "",
          "| rule | train MAE | train docs with error |", "|---|---|---|"]
    for k, v in tr["train"].items():
        L.append(f"| {k}{' (chosen)' if k == tr['chosen'] else ''} | "
                 f"{v['mae']:.3f} | {v['docs_with_error']} |")
    L += ["", "Forced error of the chosen rule:", "",
          "| split | MAE | docs with error | docs | docs with tokenless words |",
          "|---|---|---|---|---|"]
    for k, v in tr["forced_error"].items():
        L.append(f"| {k} | {v['mae']:.3f} | {v['docs_with_error']} | {v['n']:,} "
                 f"| {tr['docs_with_tokenless_words'][k]:,} |")
    pr, asap, ofx = r["peerread"], r["asap"], r["outfox"]
    L += ["", "## Human-only sources and the 8-gram filter", "",
          "Any document sharing a lower-cased word 8-gram with any Subtask C "
          "text (every file, split and generator; mixed, prefix, continuation "
          "and full human original) is removed.", "",
          f"- **PeerRead** (ICLR 2017, ACL 2017, CoNLL 2016): {pr['reviews']:,} "
          f"reviews and comments, **{pr['kept']} survive** "
          f"({pr['kept_distinct']} distinct, median "
          f"{pr['kept_median_words']:.1f} words). Not usable.",
          f"- **OUTFOX** human essays: {ofx['essays']:,}, "
          f"{ofx['kept']:,} survive, all from OUTFOX's train split "
          "(none of its valid or test essays: the Subtask C test uses them).",
          "",
          "| ASAP-Review venue | reviews | after filter | kept |",
          "|---|---|---|---|"]
    for v in sorted(asap["before"]):
        b = asap["before"][v]
        if v == "ICLR_2017":
            k = asap.get("iclr_2017_passing_filter")
            L.append(f"| {v} | {b:,} | {k if k is not None else '—'} | "
                     f"excluded (PeerRead's ICLR year) |")
            continue
        k = asap["after"].get(v, 0)
        L.append(f"| {v} | {b:,} | {k:,} | {k / b:.0%} |")
    rest = [v for v in asap["before"] if v != "ICLR_2017"]
    tb = sum(asap["before"][v] for v in rest)
    ta = sum(asap["after"].get(v, 0) for v in rest)
    L += ["", f"Outside ICLR 2017 the filter removes {1 - ta / tb:.0%} of "
          f"reviews ({tb - ta:,} of {tb:,}), from venues and years Subtask C "
          "never drew on. These are generic reviewing phrases, not leakage. "
          "The filter is kept strict anyway."]
    tw, ht = r["twins_train"], r["human_train"]
    L += ["", "## Human documents added in training", "",
          f"Condition C adds each distinct full human original behind the "
          f"{sc['train']:,} training documents once: {tw['distinct']:,} "
          f"distinct, minus {tw['also_in_test']} that also appear in test = "
          f"**{tw['used']:,}**. Condition B adds the same number of ASAP "
          f"reviews from train-side papers, matched one to one to the twins' "
          f"word counts ({ht['asap_matching'].get('whole', 0)} whole, "
          f"{ht['asap_matching'].get('truncated', 0)} cut at a sentence "
          f"boundary). "
          f"Every human document has weight 1 in both. Human-to-mixed ratio: "
          f"{ht['ratio_docs']:.2f} by documents, "
          f"{ht['twins']['total_words'] / ht['mixed_train']['total_words']:.2f}"
          f" by words.", "",
          "| set | docs | mean | p5 | p25 | p50 | p75 | p95 | total words |",
          "|---|---|---|---|---|---|---|---|---|",
          row("Subtask C train (mixed)", ht["mixed_train"]),
          row("C: twins", ht["twins"]), row("B: ASAP", ht["asap"])]
    ev = r["eval"]
    L += ["", "## Human evaluation sets", "",
          "ASAP evaluation reviews come from papers with no review in "
          "training. ASAP and OUTFOX sets are length-matched to the Subtask C "
          "test documents of their domain. Test twins are the full human "
          "originals of the test documents, minus any that appear in "
          "train or dev.", "",
          "| set | docs | mean | p5 | p25 | p50 | p75 | p95 | total words |",
          "|---|---|---|---|---|---|---|---|---|",
          row("Subtask C test, PeerRead (target)", ev["asap"]["target"]),
          row("ASAP eval", ev["asap"]["matched"]),
          row("Subtask C test, OUTFOX (target)", ev["outfox"]["target"]),
          row("OUTFOX eval", ev["outfox"]["matched"]),
          row("test twins, PeerRead", ev["twins"]["peerread"]),
          row("test twins, OUTFOX", ev["twins"]["outfox"]), "",
          "Matching (whole / cut at a sentence boundary / shorter than the "
          "target because no unused document was long enough): "
          + "; ".join(f"{k} " + " / ".join(
              str(ev[k]["matching"].get(x, 0))
              for x in ("whole", "truncated", "short"))
              for k in ("asap", "outfox")) + ".", ""]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(r, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
