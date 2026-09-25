"""70/15/15 split over SOURCE ARTICLE IDs, seeded.

Run BEFORE any generation. Splitting at the source level (not the record
level) is what prevents leakage: every derivative of a source - all
construction types, all generators - inherits that source's partition, so no
sentence from a train article can ever appear in dev or test.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import force_utf8_stdout, load_config, read_jsonl  # noqa: E402

SPLITS = ("train", "dev", "test")


def make_split(source_ids: list[str], cfg: dict) -> dict[str, list[str]]:
    """Deterministic partition of unique source ids."""
    ids = sorted(set(source_ids))          # sort first: order-independent
    rng = random.Random(cfg["seed"])
    rng.shuffle(ids)

    frac = cfg["split"]
    n = len(ids)
    n_train = int(n * frac["train"])
    n_dev = int(n * frac["dev"])
    return {
        "train": ids[:n_train],
        "dev": ids[n_train:n_train + n_dev],
        "test": ids[n_train + n_dev:],     # remainder -> test, nothing dropped
    }


def load_split_map(cfg: dict) -> dict[str, str]:
    """source_id -> split name, read back from splits/*.txt."""
    d = Path(cfg["paths"]["splits_dir"])
    mapping: dict[str, str] = {}
    for name in SPLITS:
        p = d / f"{name}_source_ids.txt"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing - run `python src/dataset/split.py` before generating.")
        for line in p.read_text(encoding="utf-8").splitlines():
            sid = line.strip()
            if sid:
                mapping[sid] = name
    return mapping


def run(cfg: dict, sources_path: str | None = None) -> dict[str, list[str]]:
    force_utf8_stdout()
    src = sources_path or cfg["paths"]["sources"]
    ids = [r["source_id"] for r in read_jsonl(src)]
    if not ids:
        raise SystemExit(f"No sources in {src}; run clean.py first.")

    splits = make_split(ids, cfg)
    out_dir = Path(cfg["paths"]["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"SPLIT REPORT   (seed={cfg['seed']}, {len(set(ids))} unique sources)")
    print("=" * 66)
    for name in SPLITS:
        p = out_dir / f"{name}_source_ids.txt"
        p.write_text("\n".join(splits[name]) + "\n", encoding="utf-8")
        print(f"  {name:<6s} {len(splits[name]):6d}  "
              f"({len(splits[name])/len(set(ids)):.2%})  -> {p}")

    # Integrity: partitions must be disjoint and cover everything.
    sets = {k: set(v) for k, v in splits.items()}
    assert not (sets["train"] & sets["dev"]), "train/dev overlap"
    assert not (sets["train"] & sets["test"]), "train/test overlap"
    assert not (sets["dev"] & sets["test"]), "dev/test overlap"
    assert set().union(*sets.values()) == set(ids), "split does not cover sources"
    print("\n  integrity: partitions disjoint and exhaustive  OK")

    # Which generators write into which split (held-out policy).
    print("\n  generator -> splits:")
    for gname, g in cfg["generators"].items():
        print(f"    {gname:<16s} role={g['role']:<9s} splits={g['splits']}")
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sources", default=None)
    a = ap.parse_args()
    run(load_config(a.config), a.sources)


if __name__ == "__main__":
    main()
