"""Export Sinhala-BOUND as a Hugging Face dataset repository.

Writes a ready-to-upload folder (default: hf_release/said-bound/) laid out
like its companion, said-sinhala-ai-dataset/said-hat:

    data/mixed/{continuation,internal_span,multiple_spans}/{split}.parquet
    data/human_originals/{split}.parquet     aligned all-human twin of each window
    data/unrelated_human/{split}.parquet     human windows from unused articles
    prompts/generation_prompts.md
    README.md, LICENSE, CITATION.bib         (written by hand, not by this script)

Every twin is verified against its mixed passages before it is written, the
same check data.load_twins applies at training time. Prints the statistics the
dataset card quotes.

    python src/dataset/export_huggingface.py [--out hf_release/said-bound]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import REPO_ROOT, force_utf8_stdout, load_config, read_jsonl  # noqa: E402

SPLITS = {"train": "train", "dev": "validation", "test": "test"}
TYPES = {"type1_single_boundary": "continuation",
         "type2_single_internal_segment": "internal_span",
         "type3_multiple_internal_segments": "multiple_spans"}
GENERATORS = {"gpt_4o": "gpt-4o", "deepseek_v3": "deepseek-v3",
              "gemini_2_5_pro": "gemini-2.5-pro"}   # said-hat naming


def split_of_source(cfg: dict) -> dict[str, str]:
    out = {}
    for ours, hf in SPLITS.items():
        path = Path(cfg["paths"]["splits_dir"]) / f"{ours}_source_ids.txt"
        for sid in path.read_text(encoding="utf-8").split():
            out[sid] = hf
    return out


def ai_spans(labels: list[int]) -> list[list[int]]:
    """Half-open [start, end) runs of AI sentences, derived from the labels."""
    spans, start = [], None
    for i, y in enumerate(labels + [0]):
        if y and start is None:
            start = i
        elif not y and start is not None:
            spans.append([start, i])
            start = None
    return spans


def mixed_rows(cfg: dict) -> list[dict]:
    rows = []
    for r in read_jsonl(Path(cfg["paths"]["generated_dir"]) / "combined.jsonl"):
        labels = [int(x) for x in r["labels"]]
        if len(labels) != len(r["sentences"]):
            raise ValueError(f"{r['record_id']}: sentence/label length mismatch")
        rows.append({
            "passage_id": r["record_id"],
            "text": r["text"],
            "sentences": list(r["sentences"]),
            "labels": labels,
            "boundaries": [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]],
            "ai_spans": ai_spans(labels),
            "construction_type": TYPES[r["construction_type"]],
            "generator": GENERATORS[r["generator"]],
            "generator_role": r["generator_role"],
            "window_id": r["window_id"],
            "source_id": r["source_id"],
            "title": r["title"],
            "url": r["url"],
            "n_sentences": len(labels),
            "n_words": r["n_words"],
            "ai_sentence_ratio": sum(labels) / len(labels),
            "prompt_id": r["prompt_id"],
            "split": SPLITS[r["split"]],
        })
    return rows


def window_row(w: dict, split: str) -> dict:
    return {"window_id": w["window_id"], "source_id": w["source_id"],
            "title": w["title"], "url": w["url"], "text": w["text"],
            "sentences": list(w["sentences"]), "labels": [0] * len(w["sentences"]),
            "n_sentences": len(w["sentences"]), "n_words": w["n_words"],
            "split": split}


def human_rows(cfg: dict, mixed: list[dict]):
    windows_path = Path(cfg["paths"]["sources"]).parent / "windows.jsonl"
    windows = {w["window_id"]: w for w in read_jsonl(windows_path)}
    by_window = defaultdict(list)
    for m in mixed:
        by_window[m["window_id"]].append(m)

    originals = []
    for wid, ms in sorted(by_window.items()):
        w = windows[wid]
        for m in ms:   # verified alignment: every human sentence is the original
            if len(w["sentences"]) != m["n_sentences"] or any(
                    a != b for a, b, y in zip(w["sentences"], m["sentences"], m["labels"]) if y == 0):
                raise ValueError(f"{m['passage_id']}: does not align with window {wid}")
        splits = {m["split"] for m in ms}
        if len(splits) != 1:
            raise ValueError(f"window {wid} spans splits {splits}")
        row = window_row(w, splits.pop())
        row["passage_ids"] = sorted(m["passage_id"] for m in ms)
        originals.append(row)

    split_of = split_of_source(cfg)
    used_sources = {m["source_id"] for m in mixed}
    unrelated = [window_row(w, split_of[w["source_id"]])
                 for wid, w in sorted(windows.items())
                 if w["source_id"] in split_of and w["source_id"] not in used_sources]
    return originals, unrelated


def write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def prompts_markdown(cfg: dict) -> str:
    spec = yaml.safe_load((REPO_ROOT / cfg["paths"]["prompts"]).read_text(encoding="utf-8"))
    wiki = spec["wikipedia"]
    used = [("Continuation (`continuation`)", wiki["single_boundary_prefix"]),
            ("Span replacement (`internal_span`, `multiple_spans`)", wiki["span_replacement"])]
    out = ["# Sinhala-BOUND generation prompts", "",
           "Each AI span was generated with one of the two prompt templates below; "
           "the `prompt_id` column of the `mixed` configs names the template used. "
           "Instructions are in Sinhala, followed by a compact English constraint "
           "block. Fields in `{braces}` are filled per passage.", "",
           "All three generators were called through OpenRouter with temperature 0.8 "
           "and top-p 0.95. Every output was checked automatically (exact sentence "
           "count, word count within 25% of the replaced text, no preamble, no "
           "Markdown, Sinhala script, not a copy of the original, numbers unchanged); "
           "failing outputs were regenerated, never hand-edited.", ""]
    for heading, p in used:
        out += [f"## {heading}", "", f"`prompt_id`: `{p['prompt_id']}`", "",
                "### System prompt", "", "```text", p["system"].rstrip(), "```", "",
                "### User prompt", "", "```text", p["user"].rstrip(), "```", ""]
    return "\n".join(out)


def main() -> None:
    force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "hf_release" / "said-bound"))
    args = ap.parse_args()
    cfg = load_config()
    out = Path(args.out)

    mixed = mixed_rows(cfg)
    originals, unrelated = human_rows(cfg, mixed)

    for ctype in TYPES.values():
        for split in SPLITS.values():
            rows = [m for m in mixed if m["construction_type"] == ctype and m["split"] == split]
            write_parquet(rows, out / "data" / "mixed" / ctype / f"{split}.parquet")
    for name, rows in (("human_originals", originals), ("unrelated_human", unrelated)):
        for split in SPLITS.values():
            write_parquet([r for r in rows if r["split"] == split],
                          out / "data" / name / f"{split}.parquet")
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "prompts" / "generation_prompts.md").write_text(prompts_markdown(cfg), encoding="utf-8")

    # ---- statistics quoted in the dataset card ----
    n_sent = sum(m["n_sentences"] for m in mixed)
    n_ai = sum(sum(m["labels"]) for m in mixed)
    print(f"mixed passages: {len(mixed)}  sentences: {n_sent}  AI: {n_ai} ({n_ai / n_sent:.1%})"
          f"  source articles: {len({m['source_id'] for m in mixed})}"
          f"  windows: {len({m['window_id'] for m in mixed})}")
    print("by type x split:")
    for ctype in TYPES.values():
        c = Counter(m["split"] for m in mixed if m["construction_type"] == ctype)
        print(f"  {ctype:15s}", {s: c[s] for s in SPLITS.values()}, sum(c.values()))
    print("by generator x split:")
    for g in GENERATORS.values():
        c = Counter(m["split"] for m in mixed if m["generator"] == g)
        print(f"  {g:15s}", {s: c[s] for s in SPLITS.values()}, sum(c.values()))
    for name, rows in (("human_originals", originals), ("unrelated_human", unrelated)):
        c = Counter(r["split"] for r in rows)
        print(f"{name}: ", {s: c[s] for s in SPLITS.values()}, len(rows))
    print("sentences per passage:", Counter(m["n_sentences"] for m in mixed).most_common())
    print(f"written to {out}")


if __name__ == "__main__":
    main()
