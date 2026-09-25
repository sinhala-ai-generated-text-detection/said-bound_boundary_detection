"""Assemble per-type JSONL files into combined.jsonl and verify integrity.

generate.py writes one record per generated document into the per-type files.
This stage deduplicates by record_id (so re-runs are idempotent), re-checks the
label invariants, enforces split inheritance, and emits combined.jsonl.

Integrity checks (a failure here means a bug upstream, not a bad generation):
  - len(sentences) == len(labels)
  - text == " ".join(sentences)
  - boundaries are exactly the label-change indices
  - every record's split matches its source_id's partition
  - a held-out generator never appears in train
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import OUT_FILES  # noqa: E402
from split import load_split_map  # noqa: E402
from utils import force_utf8_stdout, load_config, read_jsonl, write_jsonl  # noqa: E402

REQUIRED_FIELDS = [
    "record_id", "source_id", "domain", "generator", "construction_type",
    "split", "text", "sentences", "labels", "boundaries",
    "boundary_position_normalized", "prompt_id", "raw_output",
    "cleaned_output", "temperature", "top_p", "retry_count",
    "validation_passed", "validation_errors",
]


def check_record(rec: dict, split_map: dict[str, str],
                 cfg: dict) -> list[str]:
    errs: list[str] = []
    for f in REQUIRED_FIELDS:
        if f not in rec:
            errs.append(f"missing field '{f}'")
    if errs:
        return errs

    sents, labels = rec["sentences"], rec["labels"]
    if len(sents) != len(labels):
        errs.append(f"len(sentences)={len(sents)} != len(labels)={len(labels)}")
    if rec["text"] != " ".join(sents):
        errs.append("text does not equal ' '.join(sentences)")
    if set(labels) - {0, 1}:
        errs.append(f"labels contain non-binary values: {set(labels)}")

    expected = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    if rec["boundaries"] != expected:
        errs.append(f"boundaries {rec['boundaries']} != label changes {expected}")

    if not any(labels):
        errs.append("record contains no AI sentences")
    if all(labels):
        errs.append("record contains no human sentences")

    ctype = rec["construction_type"]
    n_bounds = len(rec["boundaries"])
    if ctype == "type1_single_boundary" and n_bounds != 1:
        errs.append(f"type1 must have exactly 1 boundary, got {n_bounds}")
    if ctype == "type2_single_internal_segment" and n_bounds != 2:
        errs.append(f"type2 must have exactly 2 boundaries, got {n_bounds}")
    if ctype == "type3_multiple_internal_segments" and n_bounds not in (4, 6):
        errs.append(f"type3 must have 4 or 6 boundaries, got {n_bounds}")

    true_split = split_map.get(rec["source_id"])
    if true_split is None:
        errs.append(f"source_id {rec['source_id']} not in any split")
    elif true_split != rec["split"]:
        errs.append(f"split '{rec['split']}' != source partition '{true_split}'")

    gcfg = cfg["generators"].get(rec["generator"])
    if gcfg and rec["split"] not in gcfg["splits"]:
        errs.append(f"generator {rec['generator']} ({gcfg['role']}) must not "
                    f"write to split '{rec['split']}'")
    return errs


def run(cfg: dict, strict: bool = False) -> dict:
    force_utf8_stdout()
    gdir = Path(cfg["paths"]["generated_dir"])
    split_map = load_split_map(cfg)

    seen: dict[str, dict] = {}
    per_type_counts: Counter = Counter()
    dupes = 0
    bad: list[tuple[str, list[str]]] = []

    for ctype, fname in OUT_FILES.items():
        for rec in read_jsonl(gdir / fname):
            rid = rec.get("record_id")
            if rid in seen:
                dupes += 1
                continue                      # idempotent: keep the first
            errs = check_record(rec, split_map, cfg)
            if errs:
                bad.append((rid, errs))
                continue
            seen[rid] = rec
            per_type_counts[ctype] += 1

    records = sorted(seen.values(), key=lambda r: r["record_id"])
    out = gdir / "combined.jsonl"
    write_jsonl(out, records)

    # ---- report -----------------------------------------------------------
    print("=" * 70)
    print("DATASET BUILD")
    print("=" * 70)
    print(f"  records written  : {len(records)}  -> {out}")
    print(f"  duplicate ids    : {dupes} (skipped)")
    print(f"  integrity failures: {len(bad)}")
    for rid, errs in bad[:10]:
        print(f"     {rid}: {errs}")
    if bad and strict:
        raise SystemExit("Integrity failures with --strict; aborting.")

    if records:
        print("\n  by construction type:")
        for k, v in sorted(per_type_counts.items()):
            print(f"    {k:<38s} {v}")
        by_gen: Counter = Counter(r["generator"] for r in records)
        print("\n  by generator:")
        for k, v in by_gen.most_common():
            print(f"    {k:<38s} {v}")
        by_split: Counter = Counter(r["split"] for r in records)
        print("\n  by split:")
        for k in ("train", "dev", "test"):
            print(f"    {k:<38s} {by_split.get(k, 0)}")

        cross: dict = defaultdict(Counter)
        for r in records:
            cross[r["generator"]][r["split"]] += 1
        print("\n  generator x split:")
        for g in sorted(cross):
            role = cfg["generators"][g]["role"]
            row = "  ".join(f"{s}={cross[g].get(s,0)}"
                            for s in ("train", "dev", "test"))
            print(f"    {g:<18s} ({role:<8s}) {row}")

        n_src = len({r["source_id"] for r in records})
        print(f"\n  unique source articles used: {n_src}")
    return {"records": len(records), "bad": len(bad), "duplicates": dupes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any integrity check fails")
    a = ap.parse_args()
    run(load_config(a.config), strict=a.strict)


if __name__ == "__main__":
    main()
