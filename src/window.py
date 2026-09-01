"""Contiguous passage windowing.

Long articles are cut down to a contiguous window of ~6-12 sentences rather
than used whole, so that generation targets stay small and comparable. The
source article id travels with every window, so all derivatives trace back.

Deterministic: the window offset is drawn from a RNG seeded per source_id, so
the same source always yields the same window for a given global seed.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment import word_count  # noqa: E402
from utils import force_utf8_stdout, load_config, read_jsonl, write_jsonl  # noqa: E402


def _rng_for(source_id: str, seed: int) -> random.Random:
    """Per-source RNG: stable across runs and independent of iteration order."""
    return random.Random(f"{seed}:{source_id}")


def make_windows(source: dict, cfg: dict) -> list[dict]:
    """Cut one cleaned source into contiguous, non-overlapping windows."""
    w = cfg["window"]
    sents: list[str] = source["sentences"]
    lo, hi = w["min_sentences"], w["max_sentences"]
    if len(sents) < lo:
        return []

    rng = _rng_for(source["source_id"], cfg["seed"])
    out: list[dict] = []
    max_windows = w["max_windows_per_source"]

    # Window length drawn once per source, then laid down contiguously.
    size = rng.randint(lo, min(hi, len(sents)))
    n_possible = len(sents) // size
    if n_possible == 0:
        return []

    # Random starting offset so we do not always sample the article opening
    # (leads are stylistically distinct and would bias the dataset).
    max_offset = len(sents) - size * min(n_possible, max_windows)
    offset = rng.randint(0, max_offset) if max_offset > 0 else 0

    for k in range(min(n_possible, max_windows)):
        start = offset + k * size
        end = start + size
        chunk = sents[start:end]
        text = " ".join(chunk)
        if word_count(text) < w["min_words"]:
            continue
        out.append({
            "window_id": f"{source['source_id']}_w{k}",
            "source_id": source["source_id"],
            "title": source["title"],
            "url": source.get("url"),
            "domain": source["domain"],
            "sent_start": start,
            "sent_end": end,
            "sentences": chunk,
            "n_sentences": len(chunk),
            "n_words": word_count(text),
            "text": text,
        })
    return out


def run(cfg: dict, sources_path: str | None = None,
        out_path: str | None = None) -> list[dict]:
    force_utf8_stdout()
    src = sources_path or cfg["paths"]["sources"]
    out = out_path or str(Path(cfg["paths"]["sources"]).parent / "windows.jsonl")

    windows: list[dict] = []
    n_src = 0
    n_skipped = 0
    for s in read_jsonl(src):
        n_src += 1
        ws = make_windows(s, cfg)
        if not ws:
            n_skipped += 1
        windows.extend(ws)

    write_jsonl(out, windows)

    print("=" * 66)
    print("WINDOWING REPORT")
    print("=" * 66)
    print(f"  sources read        {n_src}")
    print(f"  sources with no window {n_skipped}")
    print(f"  windows written     {len(windows)}")
    if windows:
        ns = sorted(x["n_sentences"] for x in windows)
        nw = sorted(x["n_words"] for x in windows)
        pct = lambda a, p: a[min(len(a) - 1, int(len(a) * p / 100))]  # noqa: E731
        print(f"  sentences/window    min={ns[0]} p50={pct(ns,50)} max={ns[-1]}")
        print(f"  words/window        min={nw[0]} p50={pct(nw,50)} max={nw[-1]}")
        cons = cfg["constructions"]
        for name in ("type1_single_boundary", "type2_single_internal_segment",
                     "type3_multiple_internal_segments"):
            m = cons[name]["min_sentences"]
            n = sum(1 for x in windows if x["n_sentences"] >= m)
            print(f"  eligible {name:<34s} (>={m}): {n}")
    print(f"\n  wrote -> {out}")
    return windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sources", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(load_config(a.config), a.sources, a.out)


if __name__ == "__main__":
    main()
