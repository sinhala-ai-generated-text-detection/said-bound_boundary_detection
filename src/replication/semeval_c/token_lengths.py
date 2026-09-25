"""Token-length distribution of every document set under a tokenizer.

Shows how many documents exceed one encoder window and must be chunked.

    python src/replication/semeval_c/token_lengths.py \\
        --out results/replication/semeval_c/token_lengths
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import read, words_of  # noqa: E402

SETS = ("train", "dev", "test", "human_train_asap", "human_train_twins",
        "eval_asap", "eval_outfox", "eval_twins_peerread", "eval_twins_outfox")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    budget = a.max_length - 2
    res = {}
    for s in SETS:
        docs = read(s)
        n = np.array([len(tok(words_of(d["text"]), is_split_into_words=True,
                              add_special_tokens=False)["input_ids"])
                      for d in docs])
        w = np.array([len(words_of(d["text"])) for d in docs])
        res[s] = {"docs": len(docs),
                  **{f"p{q}": float(np.percentile(n, q))
                     for q in (5, 25, 50, 75, 95, 99)},
                  "max": int(n.max()), "mean": float(n.mean()),
                  "over_window": float((n > budget).mean()),
                  "tokens_per_word": float(n.sum() / w.sum()),
                  "words_median": float(np.median(w))}
    L = [f"# Token lengths under `{a.model}`", "",
         f"Tokens per document (no special tokens). *Over window*: share of "
         f"documents longer than {budget} tokens, which are split into "
         f"chunks on word boundaries.", "",
         "| set | docs | median words | p5 | p50 | p95 | p99 | max | "
         "tokens/word | over window |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in res.items():
        L.append(f"| {s} | {r['docs']} | {r['words_median']:.0f} | "
                 f"{r['p5']:.0f} | {r['p50']:.0f} | {r['p95']:.0f} | "
                 f"{r['p99']:.0f} | {r['max']} | {r['tokens_per_word']:.2f} | "
                 f"{r['over_window']:.1%} |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
