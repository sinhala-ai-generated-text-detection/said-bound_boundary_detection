"""Probe candidate generators on the real Sinhala prompts.

Used to choose a replacement when a configured generator fails the pilot's
fluency gate. Runs the actual Type-1 and span-replacement prompts against
arbitrary OpenRouter slugs and reports validator outcomes plus the raw text,
so the Sinhala can be read and judged rather than guessed at.

    python src/dataset/probe_models.py --models google/gemma-3-27b-it qwen/qwen3-32b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import TYPE1, TYPE2, load_prompts, plan_type1, plan_type2  # noqa: E402
from openrouter import OpenRouterClient  # noqa: E402
from segment import word_count  # noqa: E402
from utils import force_utf8_stdout, load_config, read_jsonl  # noqa: E402
from validate import Validator, clean_output  # noqa: E402


def run(cfg: dict, models: list[str], n_docs: int = 3) -> None:
    force_utf8_stdout()
    import random

    windows = list(read_jsonl(Path(cfg["paths"]["sources"]).parent / "windows.jsonl"))
    rng = random.Random(cfg["seed"])
    sample = [w for w in windows if w["n_sentences"] >= 8]
    rng.shuffle(sample)
    sample = sample[:n_docs]

    prompts = load_prompts(cfg)
    validator = Validator(cfg)

    # Register each probe slug as a temporary generator so the client can use it.
    for slug in models:
        cfg["generators"][slug] = {
            "slug": slug, "role": "probe", "splits": ["dev"],
            "temperature": 0.8, "top_p": 0.95, "max_tokens": 1200,
            "price_prompt": 0.0, "price_completion": 0.0, "reasoning": None,
        }

    client = OpenRouterClient(cfg)
    results: dict[str, dict] = {}

    for slug in models:
        print("\n" + "=" * 78)
        print(f"MODEL: {slug}")
        print("=" * 78)
        passed = total = 0
        for w in sample:
            for ctype, planner, fam_name in (
                (TYPE1, plan_type1, "single_boundary_prefix"),
                (TYPE2, plan_type2, "span_replacement"),
            ):
                plan = planner(w, cfg, random.Random(1))
                if plan is None:
                    continue
                fam = prompts[w["domain"]][fam_name]
                tol = cfg["validate"]["length_tolerance"]
                tw = plan["target_word_count"]
                base = {
                    "title": w["title"],
                    "target_sentence_count": plan["target_sentence_count"],
                    "target_word_count": tw,
                    "target_word_min": int(round(tw * (1 - tol))),
                    "target_word_max": int(round(tw * (1 + tol))),
                }
                if ctype == TYPE1:
                    base["prefix"] = " ".join(plan["human_prefix"])
                    original = None
                else:
                    s, e = plan["spans"][0]
                    sents = w["sentences"]
                    base["left"] = " ".join(sents[max(0, s - 3):s])
                    base["target"] = " ".join(plan["targets"][0])
                    base["right"] = " ".join(sents[e:e + 3])
                    base["target_sentence_count"] = len(plan["targets"][0])
                    base["target_word_count"] = word_count(base["target"])
                    base["target_word_min"] = int(round(base["target_word_count"] * (1 - tol)))
                    base["target_word_max"] = int(round(base["target_word_count"] * (1 + tol)))
                    original = base["target"]

                try:
                    comp = client.complete(slug, fam["system"],
                                           fam["user"].format(**base))
                except Exception as e:
                    print(f"  [{ctype}] API ERROR: {e}")
                    continue

                cleaned = clean_output(comp.text)
                res = validator.validate(
                    cleaned,
                    target_sentence_count=base["target_sentence_count"],
                    target_word_count=base["target_word_count"],
                    original_span=original,
                )
                total += 1
                passed += int(res.passed)
                print(f"\n  [{ctype}] {'PASS' if res.passed else 'FAIL'} "
                      f"{res.details}")
                if res.errors:
                    print(f"    errors: {res.errors}")
                print(f"    --- text ---\n    {cleaned[:600]}")
        results[slug] = {"passed": passed, "total": total}

    print("\n" + "=" * 78)
    print("PROBE SUMMARY")
    print("=" * 78)
    for slug, r in results.items():
        rate = r["passed"] / r["total"] if r["total"] else 0
        print(f"  {slug:<44s} {r['passed']}/{r['total']}  ({rate:.0%})")
    client.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--docs", type=int, default=3)
    a = ap.parse_args()
    run(load_config(a.config), a.models, a.docs)


if __name__ == "__main__":
    main()
